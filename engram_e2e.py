# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Engram forward pass: gating + ShortConv + residual.

Flow:
  1. TT-Lang gating kernel: key_proj, RMSNorm, dot-product gate, value_proj
  2. Host: RMSNorm + pre-shift for conv (depthwise conv1d decomposition)
  3. TT-Lang conv kernel: streaming weighted sum + SiLU
  4. Host: residual add (value + conv_output)

Validates against full PyTorch Engram.forward() output.
"""

import torch
import torch.nn as nn
import math
import ttnn
import ttl

TILE = 32
ENGRAM_TILES = 1
HIDDEN_TILES = 1
ENGRAM_DIM = ENGRAM_TILES * TILE
HIDDEN_DIM = HIDDEN_TILES * TILE
HC_MULT = 4
KERNEL_SIZE = 4
DILATION = 3


class EngramModule(nn.Module):
    """Full Engram gating + ShortConv (no hash/embed)."""
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(ENGRAM_DIM, HIDDEN_DIM, bias=True)
        self.key_projs = nn.ModuleList(
            [nn.Linear(ENGRAM_DIM, HIDDEN_DIM, bias=True) for _ in range(HC_MULT)])
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(HIDDEN_DIM) for _ in range(HC_MULT)])
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(HIDDEN_DIM) for _ in range(HC_MULT)])
        total_ch = HIDDEN_DIM * HC_MULT
        self.conv = nn.Conv1d(
            total_ch, total_ch, KERNEL_SIZE, groups=total_ch, bias=False,
            padding=(KERNEL_SIZE - 1) * DILATION, dilation=DILATION)
        self.conv_norms = nn.ModuleList(
            [nn.RMSNorm(HIDDEN_DIM) for _ in range(HC_MULT)])
        self.silu = nn.SiLU()

    def forward(self, embeddings, hidden_states):
        # Gating
        gates = []
        for hc_idx in range(HC_MULT):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(HIDDEN_DIM)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        value = self.value_proj(embeddings).unsqueeze(2)
        gated = gates * value  # [B, L, HC_MULT, D]

        # ShortConv
        B, T, G, C = gated.shape
        normed_chunks = []
        for i in range(G):
            normed_chunks.append(self.conv_norms[i](gated[:, :, i, :]))
        x_norm = torch.cat(normed_chunks, dim=-1)  # [B, T, G*C]
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)[..., :T]
        y_bct = self.silu(y_bct)
        conv_out = y_bct.transpose(1, 2).view(B, T, G, C)

        return gated + conv_out


# Reuse validated kernels
@ttl.kernel(grid="auto")
def engram_gate_kernel(emb_a, emb_b, query, key_weight, key_bias,
                       value_weight, value_bias, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = emb_a.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    emb_a_dfb = ttl.make_dataflow_buffer_like(emb_a, shape=(1, ENGRAM_TILES), buffer_factor=2)
    emb_b_dfb = ttl.make_dataflow_buffer_like(emb_b, shape=(1, ENGRAM_TILES), buffer_factor=2)
    query_dfb = ttl.make_dataflow_buffer_like(query, shape=(1, HIDDEN_TILES), buffer_factor=2)
    kw_dfb = ttl.make_dataflow_buffer_like(key_weight, shape=(ENGRAM_TILES, HIDDEN_TILES), buffer_factor=1)
    kb_dfb = ttl.make_dataflow_buffer_like(key_bias, shape=(1, HIDDEN_TILES), buffer_factor=2)
    vw_dfb = ttl.make_dataflow_buffer_like(value_weight, shape=(ENGRAM_TILES, HIDDEN_TILES), buffer_factor=1)
    vb_dfb = ttl.make_dataflow_buffer_like(value_bias, shape=(1, HIDDEN_TILES), buffer_factor=2)
    knw_dfb = ttl.make_dataflow_buffer_like(key_norm_w, shape=(1, HIDDEN_TILES), buffer_factor=2)
    qnw_dfb = ttl.make_dataflow_buffer_like(query_norm_w, shape=(1, HIDDEN_TILES), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    isd_dfb = ttl.make_dataflow_buffer_like(inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=(1, 1), buffer_factor=1)

    mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    key_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    reduce_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    nk_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    nq_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    gb_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, isd_dfb.wait() as isd, eps_dfb.wait() as eps:
            with kw_dfb.wait() as kw, vw_dfb.wait() as vw:
                for local_t in range(tiles_per_core):
                    tile_idx = core_x * tiles_per_core + local_t
                    if tile_idx < seq_tiles:
                        with emb_a_dfb.wait() as ea:
                            with mm_dfb.reserve() as mm:
                                mm.store(ea @ kw)
                        with mm_dfb.wait() as kraw, kb_dfb.wait() as kb:
                            with key_dfb.reserve() as k:
                                k.store(kraw + kb)
                        with key_dfb.wait() as kv, knw_dfb.wait() as knw:
                            with sq_dfb.reserve() as sq:
                                sq.store(kv * kv)
                            with sq_dfb.wait() as sqv, reduce_dfb.reserve() as red:
                                red.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                            with reduce_dfb.wait() as sumv, reduce_dfb.reserve() as scaled:
                                scaled.store(sumv * ms)
                            with reduce_dfb.wait() as msq, reduce_dfb.reserve() as rsq:
                                rsq.store(ttl.math.rsqrt(msq))
                            with reduce_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(rsqv, dims=[1]))
                            with bcast_dfb.wait() as rbc, nk_dfb.reserve() as nk:
                                nk.store(kv * rbc * knw)
                        with query_dfb.wait() as qv, qnw_dfb.wait() as qnw:
                            with sq_dfb.reserve() as sq:
                                sq.store(qv * qv)
                            with sq_dfb.wait() as sqv, reduce_dfb.reserve() as red:
                                red.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                            with reduce_dfb.wait() as sumv, reduce_dfb.reserve() as scaled:
                                scaled.store(sumv * ms)
                            with reduce_dfb.wait() as msq, reduce_dfb.reserve() as rsq:
                                rsq.store(ttl.math.rsqrt(msq))
                            with reduce_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                                bc.store(ttl.math.broadcast(rsqv, dims=[1]))
                            with bcast_dfb.wait() as rbc, nq_dfb.reserve() as nq:
                                nq.store(qv * rbc * qnw)
                        with nk_dfb.wait() as nkv, nq_dfb.wait() as nqv:
                            with dot_dfb.reserve() as d:
                                d.store(nkv * nqv)
                        with dot_dfb.wait() as dv, reduce_dfb.reserve() as red:
                            red.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
                        with reduce_dfb.wait() as ds, reduce_dfb.reserve() as sd:
                            sd.store(ds * isd)
                        with reduce_dfb.wait() as sdv, gate_dfb.reserve() as g:
                            clamped = ttl.math.max(ttl.math.abs(sdv), eps)
                            g.store(ttl.math.sigmoid(sdv * ttl.math.rsqrt(clamped)))
                        with gate_dfb.wait() as gv, gb_dfb.reserve() as gb:
                            gb.store(ttl.math.broadcast(gv, dims=[1]))
                        with emb_b_dfb.wait() as eb:
                            with mm_dfb.reserve() as mm:
                                mm.store(eb @ vw)
                        with mm_dfb.wait() as vraw, vb_dfb.wait() as vb:
                            with value_dfb.reserve() as v:
                                v.store(vraw + vb)
                        with gb_dfb.wait() as gbv, value_dfb.wait() as val:
                            with out_dfb.reserve() as o:
                                o.store(gbv * val)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        with isd_dfb.reserve() as blk:
            tx = ttl.copy(inv_sqrt_d[0, 0], blk); tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_tile[0, 0], blk); tx.wait()
        with kw_dfb.reserve() as blk:
            tx = ttl.copy(key_weight[0:ENGRAM_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with vw_dfb.reserve() as blk:
            tx = ttl.copy(value_weight[0:ENGRAM_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with emb_a_dfb.reserve() as blk:
                    tx = ttl.copy(emb_a[tile_idx, 0:ENGRAM_TILES], blk); tx.wait()
                with emb_b_dfb.reserve() as blk:
                    tx = ttl.copy(emb_b[tile_idx, 0:ENGRAM_TILES], blk); tx.wait()
                with query_dfb.reserve() as blk:
                    tx = ttl.copy(query[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with kb_dfb.reserve() as blk:
                    tx = ttl.copy(key_bias[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with vb_dfb.reserve() as blk:
                    tx = ttl.copy(value_bias[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with knw_dfb.reserve() as blk:
                    tx = ttl.copy(key_norm_w[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                with qnw_dfb.reserve() as blk:
                    tx = ttl.copy(query_norm_w[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES]); tx.wait()


@ttl.kernel(grid="auto")
def conv_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
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
                    with s0_dfb.wait() as v0, s1_dfb.wait() as v1, s2_dfb.wait() as v2, s3_dfb.wait() as v3:
                        with acc_dfb.reserve() as acc:
                            acc.store(cw0 * v0 + cw1 * v1 + cw2 * v2 + cw3 * v3)
                    with acc_dfb.wait() as x, out_dfb.reserve() as o:
                        o.store(x * ttl.math.sigmoid(x))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
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


def tile_broadcast_1d(vec, seq_len):
    return vec.unsqueeze(0).expand(seq_len, -1).contiguous()


def host_shift(x, shift, seq_len):
    if shift == 0:
        return x.clone()
    result = torch.zeros_like(x)
    result[shift:] = x[:seq_len - shift]
    return result


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    SEQ = 256
    seq_tiles = SEQ // TILE

    module = EngramModule()
    embeddings_torch = torch.randn(1, SEQ, ENGRAM_DIM)
    hidden_states_torch = torch.randn(1, SEQ, HC_MULT, HIDDEN_DIM)

    with torch.no_grad():
        ref = module(embeddings_torch, hidden_states_torch)
    ref = ref.squeeze(0)
    print(f"PyTorch ref shape: {ref.shape}")

    # Scalar tiles
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    ms_torch = torch.full((32, 32), 1.0 / HIDDEN_DIM, dtype=torch.bfloat16)
    isd_torch = torch.full((32, 32), 1.0 / math.sqrt(HIDDEN_DIM), dtype=torch.bfloat16)
    eps_torch = torch.full((32, 32), 1e-6, dtype=torch.bfloat16)

    scaler_tt = to_ttnn(scaler_torch, device)
    ms_tt = to_ttnn(ms_torch, device)
    isd_tt = to_ttnn(isd_torch, device)
    eps_tt = to_ttnn(eps_torch, device)

    emb_2d = embeddings_torch.squeeze(0).to(torch.bfloat16)
    vw = module.value_proj.weight.data.T.to(torch.bfloat16).contiguous()
    vb = tile_broadcast_1d(module.value_proj.bias.data.to(torch.bfloat16), SEQ)

    # Step 1: Gating for all 4 HC groups
    gated_outputs = []
    for hc_idx in range(HC_MULT):
        kw = module.key_projs[hc_idx].weight.data.T.to(torch.bfloat16).contiguous()
        kb = tile_broadcast_1d(module.key_projs[hc_idx].bias.data.to(torch.bfloat16), SEQ)
        knw = tile_broadcast_1d(module.norm1[hc_idx].weight.data.to(torch.bfloat16), SEQ)
        qnw = tile_broadcast_1d(module.norm2[hc_idx].weight.data.to(torch.bfloat16), SEQ)
        query_slice = hidden_states_torch.squeeze(0)[:, hc_idx, :].to(torch.bfloat16).contiguous()

        out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)
        engram_gate_kernel(
            to_ttnn(emb_2d, device), to_ttnn(emb_2d, device),
            to_ttnn(query_slice, device),
            to_ttnn(kw, device), to_ttnn(kb, device),
            to_ttnn(vw, device), to_ttnn(vb, device),
            to_ttnn(knw, device), to_ttnn(qnw, device),
            scaler_tt, ms_tt, isd_tt, eps_tt, out_tt)
        gated_outputs.append(ttnn.to_torch(out_tt))

    # Step 2: ShortConv per HC group (RMSNorm + depthwise conv + SiLU)
    conv_outputs = []
    for hc_idx in range(HC_MULT):
        gated = gated_outputs[hc_idx].float()  # [SEQ, HIDDEN_DIM]

        # RMSNorm on CPU (conv norm)
        norm_w = module.conv_norms[hc_idx].weight.data.float()
        rms = torch.rsqrt((gated * gated).mean(dim=-1, keepdim=True) + 1e-5)
        normed = (gated * rms * norm_w.unsqueeze(0)).to(torch.bfloat16)

        # Pre-shift on host
        shifts = [0, DILATION, 2 * DILATION, 3 * DILATION]
        shifted = [host_shift(normed, s, SEQ) for s in shifts]

        # Conv weights for this HC group's channels
        # nn.Conv1d correlation: out[t] = w[0]*in[t-9] + w[1]*in[t-6] + w[2]*in[t-3] + w[3]*in[t]
        # Our shifts: s0=shift0 (in[t]), s1=shift3 (in[t-3]), s2=shift6 (in[t-6]), s3=shift9 (in[t-9])
        # So: w[3] pairs with s0, w[2] with s1, w[1] with s2, w[0] with s3
        ch_start = hc_idx * HIDDEN_DIM
        ch_end = ch_start + HIDDEN_DIM
        conv_w = module.conv.weight.data[ch_start:ch_end, 0, :].to(torch.bfloat16)

        weight_tiles = []
        for k in range(KERNEL_SIZE):
            # Reverse: shift k*DILATION pairs with weight[KERNEL_SIZE-1-k]
            w_row = conv_w[:, KERNEL_SIZE - 1 - k].unsqueeze(0).expand(TILE, -1).contiguous()
            weight_tiles.append(w_row)

        s_tts = [to_ttnn(s, device) for s in shifted]
        w_tts = [to_ttnn(w, device) for w in weight_tiles]
        out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)

        conv_kernel(s_tts[0], s_tts[1], s_tts[2], s_tts[3],
                     w_tts[0], w_tts[1], w_tts[2], w_tts[3], out_tt)
        conv_outputs.append(ttnn.to_torch(out_tt))

    # Step 3: Residual add on CPU
    tt_result = torch.zeros(SEQ, HC_MULT, HIDDEN_DIM)
    for hc_idx in range(HC_MULT):
        tt_result[:, hc_idx, :] = gated_outputs[hc_idx].float() + conv_outputs[hc_idx].float()

    print(f"TT-Lang result shape: {tt_result.shape}")

    ref_bf16 = ref.to(torch.bfloat16).float()
    tt_f32 = tt_result.float()

    for hc_idx in range(HC_MULT):
        hc_err = (ref_bf16[:, hc_idx, :] - tt_f32[:, hc_idx, :]).abs().max().item()
        print(f"HC {hc_idx}: max_err={hc_err:.4f}")

    overall_max = (ref_bf16 - tt_f32).abs().max().item()
    overall_mean = (ref_bf16 - tt_f32).abs().mean().item()
    print(f"\nOverall: max_err={overall_max:.4f}, mean_err={overall_mean:.6f}")
    print(f"Seq={SEQ}, ENGRAM_DIM={ENGRAM_DIM}, HIDDEN_DIM={HIDDEN_DIM}")

    if overall_max < 2.0:
        print("PASS: Full Engram forward (gating + conv + residual)")
    else:
        print("FAIL: Full Engram mismatch")

    ttnn.close_device(device)
