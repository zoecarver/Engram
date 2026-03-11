# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full Engram forward-pass comparison: PyTorch nn.Module vs TT-Lang.

Uses real nn.Linear and nn.RMSNorm weights. Compares output (without
short_conv) for all 4 HC groups.

Gating simplified: sigmoid(sqrt(|x|)*sign(x)) = sigmoid(x * rsqrt(|x|))
This avoids long fusion chains that exhaust DST registers.
"""

import torch
import torch.nn as nn
import math
import ttnn
import ttl

SEQ_TILES = 1
ENGRAM_TILES = 1
HIDDEN_TILES = 1

SEQ = SEQ_TILES * 32
ENGRAM_DIM = ENGRAM_TILES * 32
HIDDEN_DIM = HIDDEN_TILES * 32
HC_MULT = 4


class EngramGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_proj = nn.Linear(ENGRAM_DIM, HIDDEN_DIM, bias=True)
        self.key_projs = nn.ModuleList(
            [nn.Linear(ENGRAM_DIM, HIDDEN_DIM, bias=True) for _ in range(HC_MULT)])
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(HIDDEN_DIM) for _ in range(HC_MULT)])
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(HIDDEN_DIM) for _ in range(HC_MULT)])

    def forward(self, embeddings, hidden_states):
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
        return gates * value


@ttl.kernel(grid=(1, 1))
def engram_gate_kernel(emb_a, emb_b, query, key_weight, key_bias,
                       value_weight, value_bias,
                       key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    emb_a_dfb = ttl.make_dataflow_buffer_like(emb_a, shape=(SEQ_TILES, ENGRAM_TILES), buffer_factor=1)
    emb_b_dfb = ttl.make_dataflow_buffer_like(emb_b, shape=(SEQ_TILES, ENGRAM_TILES), buffer_factor=1)
    query_dfb = ttl.make_dataflow_buffer_like(query, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    kw_dfb = ttl.make_dataflow_buffer_like(key_weight, shape=(ENGRAM_TILES, HIDDEN_TILES), buffer_factor=1)
    kb_dfb = ttl.make_dataflow_buffer_like(key_bias, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    vw_dfb = ttl.make_dataflow_buffer_like(value_weight, shape=(ENGRAM_TILES, HIDDEN_TILES), buffer_factor=1)
    vb_dfb = ttl.make_dataflow_buffer_like(value_bias, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    knw_dfb = ttl.make_dataflow_buffer_like(key_norm_w, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    qnw_dfb = ttl.make_dataflow_buffer_like(query_norm_w, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    isd_dfb = ttl.make_dataflow_buffer_like(inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=(1, 1), buffer_factor=1)

    mm_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    key_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    reduce_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    normed_key_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    normed_query_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    gate_bcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, isd_dfb.wait() as isd, eps_dfb.wait() as eps:
            # Key = emb @ key_weight
            with emb_a_dfb.wait() as ea, kw_dfb.wait() as kw:
                with mm_dfb.reserve() as mm:
                    mm.store(ea @ kw)
            # Key += bias
            with mm_dfb.wait() as kraw, kb_dfb.wait() as kb:
                with key_dfb.reserve() as k:
                    k.store(kraw + kb)

            # RMSNorm key
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
                with bcast_dfb.wait() as rbc, normed_key_dfb.reserve() as nk:
                    nk.store(kv * rbc * knw)

            # RMSNorm query
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
                with bcast_dfb.wait() as rbc, normed_query_dfb.reserve() as nq:
                    nq.store(qv * rbc * qnw)

            # Dot product
            with normed_key_dfb.wait() as nkv, normed_query_dfb.wait() as nqv:
                with dot_dfb.reserve() as d:
                    d.store(nkv * nqv)

            # Per-row sum, scale
            with dot_dfb.wait() as dv, reduce_dfb.reserve() as red:
                red.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
            with reduce_dfb.wait() as dot_sum, reduce_dfb.reserve() as sd:
                sd.store(dot_sum * isd)

            # Gate: sigmoid(x * rsqrt(max(|x|, eps)))
            # This is equivalent to sigmoid(sqrt(|x|) * sign(x))
            with reduce_dfb.wait() as sdv, gate_dfb.reserve() as g:
                clamped = ttl.math.max(ttl.math.abs(sdv), eps)
                g.store(ttl.math.sigmoid(sdv * ttl.math.rsqrt(clamped)))

            # Broadcast gate
            with gate_dfb.wait() as gv, gate_bcast_dfb.reserve() as gb:
                gb.store(ttl.math.broadcast(gv, dims=[1]))

            # Value = emb @ value_weight + bias
            with emb_b_dfb.wait() as eb, vw_dfb.wait() as vw:
                with mm_dfb.reserve() as mm:
                    mm.store(eb @ vw)
            with mm_dfb.wait() as vraw, vb_dfb.wait() as vb:
                with value_dfb.reserve() as v:
                    v.store(vraw + vb)

            # Output
            with gate_bcast_dfb.wait() as gbv, value_dfb.wait() as val:
                with out_dfb.reserve() as o:
                    o.store(gbv * val)

    @ttl.datamovement()
    def dm_read():
        with emb_a_dfb.reserve() as blk:
            tx = ttl.copy(emb_a[0:SEQ_TILES, 0:ENGRAM_TILES], blk); tx.wait()
        with emb_b_dfb.reserve() as blk:
            tx = ttl.copy(emb_b[0:SEQ_TILES, 0:ENGRAM_TILES], blk); tx.wait()
        with query_dfb.reserve() as blk:
            tx = ttl.copy(query[0:SEQ_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with kw_dfb.reserve() as blk:
            tx = ttl.copy(key_weight[0:ENGRAM_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with kb_dfb.reserve() as blk:
            tx = ttl.copy(key_bias[0:SEQ_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with vw_dfb.reserve() as blk:
            tx = ttl.copy(value_weight[0:ENGRAM_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with vb_dfb.reserve() as blk:
            tx = ttl.copy(value_bias[0:SEQ_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with knw_dfb.reserve() as blk:
            tx = ttl.copy(key_norm_w[0:SEQ_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with qnw_dfb.reserve() as blk:
            tx = ttl.copy(query_norm_w[0:SEQ_TILES, 0:HIDDEN_TILES], blk); tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk); tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
        with isd_dfb.reserve() as blk:
            tx = ttl.copy(inv_sqrt_d[0, 0], blk); tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_tile[0, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES]); tx.wait()


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def tile_broadcast_1d(vec, seq_len):
    return vec.unsqueeze(0).expand(seq_len, -1).contiguous()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    module = EngramGating()
    embeddings_torch = torch.randn(1, SEQ, ENGRAM_DIM)
    hidden_states_torch = torch.randn(1, SEQ, HC_MULT, HIDDEN_DIM)

    with torch.no_grad():
        ref = module(embeddings_torch, hidden_states_torch)
    ref = ref.squeeze(0)
    print(f"PyTorch ref shape: {ref.shape}")

    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    mean_scale_torch = torch.full((32, 32), 1.0 / HIDDEN_DIM, dtype=torch.bfloat16)
    inv_sqrt_d_torch = torch.full((32, 32), 1.0 / math.sqrt(HIDDEN_DIM), dtype=torch.bfloat16)
    eps_torch = torch.full((32, 32), 1e-6, dtype=torch.bfloat16)

    scaler_tt = to_ttnn(scaler_torch, device)
    ms_tt = to_ttnn(mean_scale_torch, device)
    isd_tt = to_ttnn(inv_sqrt_d_torch, device)
    eps_tt = to_ttnn(eps_torch, device)

    emb_2d = embeddings_torch.squeeze(0).to(torch.bfloat16)
    vw = module.value_proj.weight.data.T.to(torch.bfloat16).contiguous()
    vb = tile_broadcast_1d(module.value_proj.bias.data.to(torch.bfloat16), SEQ)

    tt_outputs = []
    for hc_idx in range(HC_MULT):
        kw = module.key_projs[hc_idx].weight.data.T.to(torch.bfloat16).contiguous()
        kb = tile_broadcast_1d(module.key_projs[hc_idx].bias.data.to(torch.bfloat16), SEQ)
        knw = tile_broadcast_1d(module.norm1[hc_idx].weight.data.to(torch.bfloat16), SEQ)
        qnw = tile_broadcast_1d(module.norm2[hc_idx].weight.data.to(torch.bfloat16), SEQ)
        query_slice = hidden_states_torch.squeeze(0)[:, hc_idx, :].to(torch.bfloat16).contiguous()

        emb_a_tt = to_ttnn(emb_2d, device)
        emb_b_tt = to_ttnn(emb_2d, device)
        query_tt = to_ttnn(query_slice, device)
        kw_tt = to_ttnn(kw, device)
        kb_tt = to_ttnn(kb, device)
        vw_tt = to_ttnn(vw, device)
        vb_tt = to_ttnn(vb, device)
        knw_tt = to_ttnn(knw, device)
        qnw_tt = to_ttnn(qnw, device)
        out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)

        engram_gate_kernel(emb_a_tt, emb_b_tt, query_tt, kw_tt, kb_tt,
                           vw_tt, vb_tt, knw_tt, qnw_tt,
                           scaler_tt, ms_tt, isd_tt, eps_tt, out_tt)

        tt_outputs.append(ttnn.to_torch(out_tt))

    tt_result = torch.stack(tt_outputs, dim=1)
    print(f"TT-Lang result shape: {tt_result.shape}")

    ref_bf16 = ref.to(torch.bfloat16).float()
    tt_f32 = tt_result.float()

    for hc_idx in range(HC_MULT):
        hc_err = (ref_bf16[:, hc_idx, :] - tt_f32[:, hc_idx, :]).abs().max().item()
        print(f"HC {hc_idx}: max_err={hc_err:.4f}  ref[0,:4]={ref_bf16[0, hc_idx, :4].tolist()}  tt[0,:4]={tt_f32[0, hc_idx, :4].tolist()}")

    overall_max = (ref_bf16 - tt_f32).abs().max().item()
    overall_mean = (ref_bf16 - tt_f32).abs().mean().item()
    print(f"\nOverall: max_err={overall_max:.4f}, mean_err={overall_mean:.6f}")

    if overall_max < 1.0:
        print("PASS: Full Engram gating matches PyTorch module")
    else:
        print("FAIL: Engram gating mismatch")

    ttnn.close_device(device)
