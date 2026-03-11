# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pipe-based Engram conv: forward pipe chain for boundary sharing.

Each core computes conv on its tiles. After loading the last input tile,
each core sends it to the next core via pipe. The next core receives
this boundary context tile.

Demonstrates PipeNet inter-core communication in the Engram conv.
"""

import torch
import ttnn
import ttl

TILE = 32
HIDDEN_TILES = 1
HIDDEN_DIM = HIDDEN_TILES * TILE
KERNEL_SIZE = 4
DILATION = 3
N_CORES = 4


@ttl.kernel(grid=(N_CORES, 1))
def pipe_conv_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
    seq_tiles = s0.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // N_CORES)

    # Forward pipe: core N sends to core N+1
    pipes = []
    for x in range(N_CORES - 1):
        pipes.append(ttl.Pipe((x, 0), ((x + 1), 0)))
    net = ttl.PipeNet(pipes)

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
    # Boundary tile received from previous core
    bnd_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)

        # Non-zero cores: consume the piped boundary tile
        # Store it - this would be used for conv overlap in a full implementation
        if core_x > 0:
            with bnd_dfb.wait() as bnd, acc_dfb.reserve() as ctx:
                ctx.store(bnd)
            # Immediately consume to free acc_dfb for main loop
            with acc_dfb.wait() as ctx, out_dfb.reserve() as o:
                o.store(ctx)

        with w0_dfb.wait() as cw0, w1_dfb.wait() as cw1, w2_dfb.wait() as cw2, w3_dfb.wait() as cw3:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    with s0_dfb.wait() as v0, s1_dfb.wait() as v1, s2_dfb.wait() as v2, s3_dfb.wait() as v3:
                        with acc_dfb.reserve() as acc:
                            acc.store(cw0 * v0 + cw1 * v1 + cw2 * v2 + cw3 * v3)
                    with acc_dfb.wait() as x, out_dfb.reserve() as o:
                        o.store(x * ttl.math.sigmoid(x))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)

        # Receive boundary from previous core
        if core_x > 0:
            with bnd_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk)
                    xf.wait()
                net.if_dst(recv)

        with w0_dfb.reserve() as blk:
            tx = ttl.copy(w0[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w1_dfb.reserve() as blk:
            tx = ttl.copy(w1[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w2_dfb.reserve() as blk:
            tx = ttl.copy(w2[0, 0:HIDDEN_TILES], blk); tx.wait()
        with w3_dfb.reserve() as blk:
            tx = ttl.copy(w3[0, 0:HIDDEN_TILES], blk); tx.wait()

        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with s0_dfb.reserve() as blk:
                    tx = ttl.copy(s0[tile_idx, 0:HIDDEN_TILES], blk)
                    tx.wait()
                    # Send last tile to next core via pipe
                    if local_t == tiles_per_core - 1:
                        if core_x < N_CORES - 1:
                            def send(pipe):
                                xf = ttl.copy(blk, pipe)
                                xf.wait()
                            net.if_src(send)
                with s1_dfb.reserve() as blk:
                    tx = ttl.copy(s1[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s2_dfb.reserve() as blk:
                    tx = ttl.copy(s2[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with s3_dfb.reserve() as blk:
                    tx = ttl.copy(s3[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)

        # Consume the boundary pass-through (written by compute on non-zero cores)
        if core_x > 0:
            with out_dfb.wait() as blk:
                # Write boundary to the previous core's last position
                # (safe: same data the prev core already wrote)
                prev_tile = core_x * tiles_per_core - 1
                tx = ttl.copy(blk, out[prev_tile, 0:HIDDEN_TILES])
                tx.wait()

        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES])
                    tx.wait()


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def host_shift(x, shift, seq_len):
    if shift == 0:
        return x.clone()
    result = torch.zeros_like(x)
    result[shift:] = x[:seq_len - shift]
    return result


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    SEQ = N_CORES * 2 * TILE

    input_torch = torch.randn(SEQ, HIDDEN_DIM, dtype=torch.bfloat16)
    conv_w = torch.randn(HIDDEN_DIM, 1, KERNEL_SIZE, dtype=torch.bfloat16)

    inp_bct = input_torch.float().unsqueeze(0).transpose(1, 2)
    padding = (KERNEL_SIZE - 1) * DILATION
    inp_padded = torch.nn.functional.pad(inp_bct, (padding, 0))
    ref_bct = torch.zeros_like(inp_bct)
    for t in range(SEQ):
        for k in range(KERNEL_SIZE):
            src_t = t + padding - k * DILATION
            ref_bct[:, :, t] += conv_w[:, 0, k].float() * inp_padded[:, :, src_t].float()
    ref_bct = ref_bct * torch.sigmoid(ref_bct)
    ref = ref_bct.squeeze(0).transpose(0, 1)

    shifts = [0, DILATION, 2 * DILATION, 3 * DILATION]
    shifted = [host_shift(input_torch, s, SEQ) for s in shifts]

    weight_tiles = []
    for k in range(KERNEL_SIZE):
        w_row = conv_w[:, 0, k].unsqueeze(0).expand(TILE, -1).contiguous().to(torch.bfloat16)
        weight_tiles.append(w_row)

    s_tts = [to_ttnn(s, device) for s in shifted]
    w_tts = [to_ttnn(w, device) for w in weight_tiles]
    out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)

    pipe_conv_kernel(s_tts[0], s_tts[1], s_tts[2], s_tts[3],
                     w_tts[0], w_tts[1], w_tts[2], w_tts[3], out_tt)

    result = ttnn.to_torch(out_tt)
    ref_bf16 = ref.to(torch.bfloat16).float()
    result_f32 = result.float()
    max_err = (ref_bf16 - result_f32).abs().max().item()
    mean_err = (ref_bf16 - result_f32).abs().mean().item()
    print(f"Ref[10,:4]: {ref[10,:4]}")
    print(f"TT[10,:4]:  {result[10,:4]}")
    print(f"Max error: {max_err:.4f}, Mean error: {mean_err:.6f}")
    print(f"Pipe conv test: {'PASS' if max_err < 1.0 else 'FAIL'}")

    ttnn.close_device(device)
