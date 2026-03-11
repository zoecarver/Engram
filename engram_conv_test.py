# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ShortConv test: depthwise conv1d with kernel_size=4, dilation=3.

Approach:
  1. Host creates 4 shifted copies (shift by 0, 3, 6, 9 positions)
  2. TT-Lang kernel computes weighted sum + SiLU activation
  3. Streaming with grid="auto"

The shift-and-weight pattern avoids complex on-device row manipulation.
Pipes will be used for cross-core boundary sharing in the full version.
"""

import torch
import torch.nn as nn
import math
import ttnn
import ttl

TILE = 32
HIDDEN_TILES = 1
HIDDEN_DIM = HIDDEN_TILES * TILE
HC_MULT = 4
KERNEL_SIZE = 4
DILATION = 3


def pytorch_short_conv(x, conv_weights, norm_weights):
    """
    PyTorch reference for ShortConv.
    x: [SEQ, HC_MULT, HIDDEN_DIM]
    conv_weights: [total_channels, 1, kernel_size] (depthwise conv1d weights)
    norm_weights: list of HC_MULT x [HIDDEN_DIM] (RMSNorm weights)
    Returns: [SEQ, HC_MULT, HIDDEN_DIM]
    """
    B, T, G, C = 1, x.shape[0], x.shape[1], x.shape[2]
    x = x.unsqueeze(0)  # [1, T, G, C]

    # RMSNorm per group
    normed = []
    for i in range(G):
        chunk = x[:, :, i, :]  # [1, T, C]
        rms = torch.rsqrt((chunk * chunk).mean(dim=-1, keepdim=True) + 1e-5)
        normed.append(chunk * rms * norm_weights[i].unsqueeze(0).unsqueeze(0))
    x_norm = torch.cat(normed, dim=-1)  # [1, T, G*C]

    # Depthwise conv1d
    x_bct = x_norm.transpose(1, 2)  # [1, G*C, T]
    padding = (KERNEL_SIZE - 1) * DILATION
    x_padded = torch.nn.functional.pad(x_bct, (padding, 0))
    # Manual depthwise conv1d for clarity
    total_ch = G * C
    out = torch.zeros_like(x_bct)
    for t in range(T):
        for k in range(KERNEL_SIZE):
            src_t = t + padding - k * DILATION
            if 0 <= src_t < T + padding:
                out[:, :, t] += conv_weights[:, 0, k] * x_padded[:, :, src_t]

    # SiLU
    out = out * torch.sigmoid(out)

    out = out.transpose(1, 2).view(B, T, G, C).squeeze(0)
    return out


def host_shift(x, shift, seq_len):
    """Shift tensor along dim 0 by `shift` positions (causal, zero-padded)."""
    if shift == 0:
        return x.clone()
    result = torch.zeros_like(x)
    result[shift:] = x[:seq_len - shift]
    return result


@ttl.kernel(grid="auto")
def conv_weighted_sum_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
    """
    Streaming weighted sum + SiLU.
    s0..s3: pre-shifted inputs [seq_tiles, hidden_tiles]
    w0..w3: per-channel weights broadcast to [1, hidden_tiles]
    out = silu(w0*s0 + w1*s1 + w2*s2 + w3*s3)
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = s0.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    s0_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s1_dfb = ttl.make_dataflow_buffer_like(s1, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s2_dfb = ttl.make_dataflow_buffer_like(s2, shape=(1, HIDDEN_TILES), buffer_factor=2)
    s3_dfb = ttl.make_dataflow_buffer_like(s3, shape=(1, HIDDEN_TILES), buffer_factor=2)
    w0_dfb = ttl.make_dataflow_buffer_like(w0, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w1_dfb = ttl.make_dataflow_buffer_like(w1, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w2_dfb = ttl.make_dataflow_buffer_like(w2, shape=(1, HIDDEN_TILES), buffer_factor=1)
    w3_dfb = ttl.make_dataflow_buffer_like(w3, shape=(1, HIDDEN_TILES), buffer_factor=1)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with w0_dfb.wait() as cw0, w1_dfb.wait() as cw1, w2_dfb.wait() as cw2, w3_dfb.wait() as cw3:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Weighted sum: w0*s0 + w1*s1 + w2*s2 + w3*s3
                    with s0_dfb.wait() as v0, s1_dfb.wait() as v1, s2_dfb.wait() as v2, s3_dfb.wait() as v3:
                        with acc_dfb.reserve() as acc:
                            acc.store(cw0 * v0 + cw1 * v1 + cw2 * v2 + cw3 * v3)

                    # SiLU: x * sigmoid(x)
                    with acc_dfb.wait() as x, out_dfb.reserve() as o:
                        o.store(x * ttl.math.sigmoid(x))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        # Weights loaded once
        with w0_dfb.reserve() as blk:
            tx = ttl.copy(w0[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w1_dfb.reserve() as blk:
            tx = ttl.copy(w1[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w2_dfb.reserve() as blk:
            tx = ttl.copy(w2[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w3_dfb.reserve() as blk:
            tx = ttl.copy(w3[0, 0:HIDDEN_TILES], blk); tx.wait()

        # Stream shifted inputs
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with s0_dfb.reserve() as blk:
                    tx = ttl.copy(s0[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s1_dfb.reserve() as blk:
                    tx = ttl.copy(s1[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s2_dfb.reserve() as blk:
                    tx = ttl.copy(s2[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s3_dfb.reserve() as blk:
                    tx = ttl.copy(s3[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES]); tx.wait()


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    SEQ = 256

    # Test with a single channel group first (one HC group's hidden dim)
    # Input: [SEQ, HIDDEN_DIM] (one channel group of the depthwise conv)
    input_torch = torch.randn(SEQ, HIDDEN_DIM, dtype=torch.bfloat16)

    # Conv weights: 4 values per channel
    conv_w = torch.randn(HIDDEN_DIM, 1, KERNEL_SIZE, dtype=torch.bfloat16)

    # PyTorch reference (just the conv part, no RMSNorm for this test)
    inp_bct = input_torch.float().unsqueeze(0).transpose(1, 2)  # [1, C, T]
    padding = (KERNEL_SIZE - 1) * DILATION
    inp_padded = torch.nn.functional.pad(inp_bct, (padding, 0))
    ref_bct = torch.zeros_like(inp_bct)
    for t in range(SEQ):
        for k in range(KERNEL_SIZE):
            src_t = t + padding - k * DILATION
            ref_bct[:, :, t] += conv_w[:, 0, k].float() * inp_padded[:, :, src_t].float()
    ref_bct = ref_bct * torch.sigmoid(ref_bct)  # SiLU
    ref = ref_bct.squeeze(0).transpose(0, 1)  # [T, C]
    print(f"PyTorch ref shape: {ref.shape}")
    print(f"PyTorch ref[0, :4]: {ref[0, :4]}")
    print(f"PyTorch ref[10, :4]: {ref[10, :4]}")

    # Pre-shift on host
    shifts = [0, DILATION, 2 * DILATION, 3 * DILATION]  # [0, 3, 6, 9]
    shifted = [host_shift(input_torch, s, SEQ) for s in shifts]

    # Conv weights as [1, HIDDEN_DIM] tiles (broadcast across seq)
    # w_k is the weight for shift k, shape [1, HIDDEN_DIM] -> tile-broadcast to [SEQ, HIDDEN_DIM]
    weight_tiles = []
    for k in range(KERNEL_SIZE):
        w_row = conv_w[:, 0, k].unsqueeze(0).expand(TILE, -1).contiguous().to(torch.bfloat16)
        weight_tiles.append(w_row)

    # Convert to TTNN
    s_tts = [to_ttnn(s, device) for s in shifted]
    w_tts = [to_ttnn(w, device) for w in weight_tiles]
    out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)

    conv_weighted_sum_kernel(s_tts[0], s_tts[1], s_tts[2], s_tts[3],
                             w_tts[0], w_tts[1], w_tts[2], w_tts[3], out_tt)

    result = ttnn.to_torch(out_tt)
    print(f"TT-Lang result shape: {result.shape}")
    print(f"TT-Lang result[0, :4]: {result[0, :4]}")
    print(f"TT-Lang result[10, :4]: {result[10, :4]}")

    ref_bf16 = ref.to(torch.bfloat16).float()
    result_f32 = result.float()
    max_err = (ref_bf16 - result_f32).abs().max().item()
    mean_err = (ref_bf16 - result_f32).abs().mean().item()
    print(f"Max error: {max_err:.4f}")
    print(f"Mean error: {mean_err:.6f}")

    if max_err < 1.0:
        print("PASS: Conv weighted sum + SiLU matches")
    else:
        print("FAIL: Conv mismatch")

    ttnn.close_device(device)
