# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Engram gating kernel: one HC group, (1,1) tiles.

Fused compute path:
  key = embeddings @ key_weight
  normed_key = rmsnorm(key)
  normed_query = rmsnorm(query)
  dot = sum(normed_key * normed_query, dim=-1) / sqrt(D)   [per-row]
  gate = sigmoid(sqrt(abs(dot)) * sign(dot))                [per-row scalar]
  value = embeddings @ value_weight
  output = gate * value                                     [broadcast gate per row]
"""

import torch
import math
import ttnn
import ttl

SEQ_TILES = 1
EMBED_TILES = 1
HIDDEN_TILES = 1

SEQ = SEQ_TILES * 32
ENGRAM_DIM = EMBED_TILES * 32
HIDDEN_DIM = HIDDEN_TILES * 32


def pytorch_reference(embeddings, query, key_weight, value_weight,
                      key_norm_w, query_norm_w):
    """Full gating for one HC group, per-row RMSNorm and per-row dot."""
    # Key projection + per-row RMSNorm
    key = embeddings @ key_weight
    key_rms = torch.rsqrt((key * key).mean(dim=-1, keepdim=True) + 1e-5)
    normed_key = key * key_rms * key_norm_w.unsqueeze(0)

    # Per-row RMSNorm on query
    query_rms = torch.rsqrt((query * query).mean(dim=-1, keepdim=True) + 1e-5)
    normed_query = query * query_rms * query_norm_w.unsqueeze(0)

    # Per-row gating
    dot = (normed_key * normed_query).sum(dim=-1) / math.sqrt(HIDDEN_DIM)
    gate = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
    gate = gate.sigmoid().unsqueeze(-1)  # [SEQ, 1]

    # Value projection
    value = embeddings @ value_weight

    return gate * value


@ttl.kernel(grid=(1, 1))
def engram_gate_kernel(embeddings_a, embeddings_b, query, key_weight,
                       value_weight, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    """
    Full Engram gating for one HC group.

    Reduction strategy (at 1x1 tiles):
      - RMSNorm: dims=[0] reduces across columns (hidden dim), per-row result
      - Dot product sum: dims=[0] same, per-row sum
      - Broadcast back: dims=[1] replicates column 0 across all columns

    scaler: 1x1 tile of 1.0s
    mean_scale: 1x1 tile of 1/HIDDEN_DIM (for per-row mean)
    inv_sqrt_d: 1x1 tile of 1/sqrt(HIDDEN_DIM)
    eps_tile: 1x1 tile of 1e-6
    """
    emb_a_dfb = ttl.make_dataflow_buffer_like(
        embeddings_a, shape=(SEQ_TILES, EMBED_TILES), buffer_factor=1)
    emb_b_dfb = ttl.make_dataflow_buffer_like(
        embeddings_b, shape=(SEQ_TILES, EMBED_TILES), buffer_factor=1)
    query_dfb = ttl.make_dataflow_buffer_like(
        query, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    kw_dfb = ttl.make_dataflow_buffer_like(
        key_weight, shape=(EMBED_TILES, HIDDEN_TILES), buffer_factor=1)
    vw_dfb = ttl.make_dataflow_buffer_like(
        value_weight, shape=(EMBED_TILES, HIDDEN_TILES), buffer_factor=1)
    knw_dfb = ttl.make_dataflow_buffer_like(
        key_norm_w, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    qnw_dfb = ttl.make_dataflow_buffer_like(
        query_norm_w, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(
        mean_scale, shape=(1, 1), buffer_factor=1)
    inv_sqrt_dfb = ttl.make_dataflow_buffer_like(
        inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(
        eps_tile, shape=(1, 1), buffer_factor=1)

    key_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    sq_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    reduce_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    normed_key_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    normed_query_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    # Gate is per-row: after reduce it's (1,1) with values in col 0
    gate_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2)
    gate_bcast_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(SEQ_TILES, HIDDEN_TILES), buffer_factor=1)

    @ttl.compute()
    def compute():
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, inv_sqrt_dfb.wait() as isd, eps_dfb.wait() as eps:
            # === Key projection ===
            with emb_a_dfb.wait() as emb_a, kw_dfb.wait() as kw:
                with key_dfb.reserve() as k:
                    k.store(emb_a @ kw)

            # === RMSNorm key (per-row) ===
            with key_dfb.wait() as kv, knw_dfb.wait() as knw:
                with sq_dfb.reserve() as sq:
                    sq.store(kv * kv)
                # dims=[0]: per-row reduce (keeps rows, sums columns)
                with sq_dfb.wait() as sqv, reduce_dfb.reserve() as red:
                    red.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                with reduce_dfb.wait() as sumv, reduce_dfb.reserve() as scaled:
                    scaled.store(sumv * ms)
                with reduce_dfb.wait() as mean_sq, reduce_dfb.reserve() as rsq:
                    rsq.store(ttl.math.rsqrt(mean_sq))
                # dims=[1]: broadcast col 0 across all columns
                with reduce_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                    bc.store(ttl.math.broadcast(rsqv, dims=[1]))
                with bcast_dfb.wait() as rbc, normed_key_dfb.reserve() as nk:
                    nk.store(kv * rbc * knw)

            # === RMSNorm query (per-row) ===
            with query_dfb.wait() as qv, qnw_dfb.wait() as qnw:
                with sq_dfb.reserve() as sq:
                    sq.store(qv * qv)
                with sq_dfb.wait() as sqv, reduce_dfb.reserve() as red:
                    red.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                with reduce_dfb.wait() as sumv, reduce_dfb.reserve() as scaled:
                    scaled.store(sumv * ms)
                with reduce_dfb.wait() as mean_sq, reduce_dfb.reserve() as rsq:
                    rsq.store(ttl.math.rsqrt(mean_sq))
                with reduce_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                    bc.store(ttl.math.broadcast(rsqv, dims=[1]))
                with bcast_dfb.wait() as rbc, normed_query_dfb.reserve() as nq:
                    nq.store(qv * rbc * qnw)

            # === Gating ===
            with normed_key_dfb.wait() as nkv, normed_query_dfb.wait() as nqv:
                with dot_dfb.reserve() as d:
                    d.store(nkv * nqv)

            # Per-row sum of dot products
            with dot_dfb.wait() as dv, reduce_dfb.reserve() as red:
                red.store(ttl.math.reduce_sum(dv, sc, dims=[0]))

            # gate = sigmoid(sqrt(max(abs(x), eps)) * sign(x))
            with reduce_dfb.wait() as dot_sum, gate_dfb.reserve() as g:
                scaled_dot = dot_sum * isd
                abs_val = ttl.math.abs(scaled_dot)
                clamped = ttl.math.max(abs_val, eps)
                sqrt_val = ttl.math.sqrt(clamped)
                sign_val = scaled_dot * ttl.math.recip(clamped)
                g.store(ttl.math.sigmoid(sqrt_val * sign_val))

            # Broadcast gate per-row values across all columns
            with gate_dfb.wait() as gv, gate_bcast_dfb.reserve() as gb:
                gb.store(ttl.math.broadcast(gv, dims=[1]))

            # === Value projection ===
            with emb_b_dfb.wait() as emb_b, vw_dfb.wait() as vw:
                with value_dfb.reserve() as v:
                    v.store(emb_b @ vw)

            # === Output: gate * value ===
            with gate_bcast_dfb.wait() as gbv, value_dfb.wait() as val:
                with out_dfb.reserve() as o:
                    o.store(gbv * val)

    @ttl.datamovement()
    def dm_read():
        with emb_a_dfb.reserve() as blk:
            tx = ttl.copy(embeddings_a[0:SEQ_TILES, 0:EMBED_TILES], blk)
            tx.wait()
        with emb_b_dfb.reserve() as blk:
            tx = ttl.copy(embeddings_b[0:SEQ_TILES, 0:EMBED_TILES], blk)
            tx.wait()
        with query_dfb.reserve() as blk:
            tx = ttl.copy(query[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with kw_dfb.reserve() as blk:
            tx = ttl.copy(key_weight[0:EMBED_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with vw_dfb.reserve() as blk:
            tx = ttl.copy(value_weight[0:EMBED_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with knw_dfb.reserve() as blk:
            tx = ttl.copy(key_norm_w[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with qnw_dfb.reserve() as blk:
            tx = ttl.copy(query_norm_w[0:SEQ_TILES, 0:HIDDEN_TILES], blk)
            tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        with ms_dfb.reserve() as blk:
            tx = ttl.copy(mean_scale[0, 0], blk)
            tx.wait()
        with inv_sqrt_dfb.reserve() as blk:
            tx = ttl.copy(inv_sqrt_d[0, 0], blk)
            tx.wait()
        with eps_dfb.reserve() as blk:
            tx = ttl.copy(eps_tile[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:HIDDEN_TILES])
            tx.wait()


def to_ttnn(tensor, device):
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    embeddings_torch = torch.randn(SEQ, ENGRAM_DIM, dtype=torch.bfloat16)
    query_torch = torch.randn(SEQ, HIDDEN_DIM, dtype=torch.bfloat16)
    key_weight_torch = torch.randn(ENGRAM_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
    value_weight_torch = torch.randn(ENGRAM_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
    key_norm_w = torch.ones(HIDDEN_DIM, dtype=torch.bfloat16)
    query_norm_w = torch.ones(HIDDEN_DIM, dtype=torch.bfloat16)

    ref = pytorch_reference(
        embeddings_torch.float(), query_torch.float(),
        key_weight_torch.float(), value_weight_torch.float(),
        key_norm_w.float(), query_norm_w.float())
    print(f"PyTorch ref shape: {ref.shape}")
    print(f"PyTorch ref[0, :8]: {ref[0, :8]}")
    print(f"PyTorch ref[1, :8]: {ref[1, :8]}")

    # Tile-broadcast norm weights
    knw_tiled = key_norm_w.unsqueeze(0).expand(SEQ, -1).contiguous()
    qnw_tiled = query_norm_w.unsqueeze(0).expand(SEQ, -1).contiguous()

    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    mean_scale_torch = torch.full((32, 32), 1.0 / HIDDEN_DIM, dtype=torch.bfloat16)
    inv_sqrt_d_torch = torch.full((32, 32), 1.0 / math.sqrt(HIDDEN_DIM), dtype=torch.bfloat16)
    eps_torch = torch.full((32, 32), 1e-6, dtype=torch.bfloat16)
    out_torch = torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16)

    emb_a_tt = to_ttnn(embeddings_torch, device)
    emb_b_tt = to_ttnn(embeddings_torch, device)
    query_tt = to_ttnn(query_torch, device)
    kw_tt = to_ttnn(key_weight_torch, device)
    vw_tt = to_ttnn(value_weight_torch, device)
    knw_tt = to_ttnn(knw_tiled, device)
    qnw_tt = to_ttnn(qnw_tiled, device)
    scaler_tt = to_ttnn(scaler_torch, device)
    ms_tt = to_ttnn(mean_scale_torch, device)
    isd_tt = to_ttnn(inv_sqrt_d_torch, device)
    eps_tt = to_ttnn(eps_torch, device)
    out_tt = to_ttnn(out_torch, device)

    engram_gate_kernel(emb_a_tt, emb_b_tt, query_tt, kw_tt, vw_tt,
                       knw_tt, qnw_tt, scaler_tt, ms_tt, isd_tt, eps_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    print(f"TT-Lang result shape: {result.shape}")
    print(f"TT-Lang result[0, :8]: {result[0, :8]}")
    print(f"TT-Lang result[1, :8]: {result[1, :8]}")

    ref_bf16 = ref.to(torch.bfloat16).float()
    result_f32 = result.float()
    max_err = (ref_bf16 - result_f32).abs().max().item()
    mean_err = (ref_bf16 - result_f32).abs().mean().item()
    print(f"Max absolute error: {max_err:.4f}")
    print(f"Mean absolute error: {mean_err:.4f}")

    if max_err < 1.0:
        print("PASS: full gating kernel matches")
    else:
        print("FAIL: gating kernel mismatch")

    ttnn.close_device(device)
