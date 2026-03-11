# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Engram forward pass: PyTorch reference vs TT hardware.

Uses the same Engram module from engram_demo_v1.py with identical weights
and input text. Compares outputs at each stage to show numerical equivalence.

Flow:
  1. Tokenize + Hash + Embed (CPU, shared)
  2. Key/Value projection (CPU, shared)
  3. Gating: PyTorch vs TT-Lang kernel (grid=auto, (1,1) tile DFBs)
  4. ShortConv: PyTorch vs TT-Lang pipe conv kernel (grid=N_CONV_CORES)
  5. Residual add + final comparison

TT-Lang kernels use (1,1) tile DFBs with loops over HIDDEN_TILES
to work around non-square matmul compiler bug (#383).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import math
import ttnn
import ttl

from engram_demo_v1 import (
    EngramConfig, BackBoneConfig, engram_cfg, backbone_config,
    CompressedTokenizer, NgramHashMapping, MultiHeadEmbedding,
    ShortConv, Engram,
)
from transformers import AutoTokenizer

TILE = 32

HIDDEN_DIM = backbone_config.hidden_size          # 1024
ENGRAM_DIM = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram  # 1024
HIDDEN_TILES = HIDDEN_DIM // TILE                  # 32
ENGRAM_TILES = ENGRAM_DIM // TILE                  # 32
HC_MULT = backbone_config.hc_mult                  # 4
KERNEL_SIZE = engram_cfg.kernel_size               # 4
DILATION = engram_cfg.max_ngram_size               # 3
N_CONV_CORES = 4

C_MEAN_SCALE = 1.0 / HIDDEN_DIM
C_INV_SQRT_D = 1.0 / math.sqrt(HIDDEN_DIM)
C_EPS = 1e-6


# --- Gating kernel (streaming, grid=auto, (1,1) tile DFBs) ---
# Works around non-square matmul compiler bug (#383) by using (1,1) tiles
# and looping over HIDDEN_TILES for reduce/normalize.

@ttl.kernel(grid="auto")
def engram_gate_kernel(key, query, value, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    grid_cols, _ = ttl.grid_size(dims=2)
    seq_tiles = key.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // grid_cols)

    key_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    query_dfb = ttl.make_dataflow_buffer_like(query, shape=(1, 1), buffer_factor=2)
    value_dfb = ttl.make_dataflow_buffer_like(value, shape=(1, 1), buffer_factor=2)
    knw_dfb = ttl.make_dataflow_buffer_like(key_norm_w, shape=(1, 1), buffer_factor=2)
    qnw_dfb = ttl.make_dataflow_buffer_like(query_norm_w, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
    isd_dfb = ttl.make_dataflow_buffer_like(inv_sqrt_d, shape=(1, 1), buffer_factor=1)
    eps_dfb = ttl.make_dataflow_buffer_like(eps_tile, shape=(1, 1), buffer_factor=1)

    sq_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    key_rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    query_rsq_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    dot_dfb = ttl.make_dataflow_buffer_like(key, shape=(1, 1), buffer_factor=2)
    gate_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        with scaler_dfb.wait() as sc, ms_dfb.wait() as ms, isd_dfb.wait() as isd, eps_dfb.wait() as eps:
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # --- Key RMSNorm pass 1: sum of squares ---
                    with key_dfb.wait() as k0:
                        with sq_dfb.reserve() as sq:
                            sq.store(k0 * k0)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                        acc.store(rv)
                    for j in range(HIDDEN_TILES - 1):
                        with key_dfb.wait() as kj:
                            with sq_dfb.reserve() as sq:
                                sq.store(kj * kj)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                            new_acc.store(av + rv)
                    # Broadcast col 0 -> all cols before rsqrt (reduce leaves garbage)
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                        scaled.store(bv * ms)
                    with red_dfb.wait() as msq, red_dfb.reserve() as rsq:
                        rsq.store(ttl.math.rsqrt(msq))
                    with red_dfb.wait() as rsqv, key_rsq_dfb.reserve() as kr:
                        kr.store(rsqv)

                    # --- Query RMSNorm pass 1: sum of squares ---
                    with query_dfb.wait() as q0:
                        with sq_dfb.reserve() as sq:
                            sq.store(q0 * q0)
                    with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                    with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                        acc.store(rv)
                    for j in range(HIDDEN_TILES - 1):
                        with query_dfb.wait() as qj:
                            with sq_dfb.reserve() as sq:
                                sq.store(qj * qj)
                        with sq_dfb.wait() as sqv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(sqv, sc, dims=[0]))
                        with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                            new_acc.store(av + rv)
                    with acc_dfb.wait() as total, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(total, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as scaled:
                        scaled.store(bv * ms)
                    with red_dfb.wait() as msq, red_dfb.reserve() as rsq:
                        rsq.store(ttl.math.rsqrt(msq))
                    with red_dfb.wait() as rsqv, query_rsq_dfb.reserve() as qr:
                        qr.store(rsqv)

                    # --- Normalize + dot product (interleaved) ---
                    with key_rsq_dfb.wait() as key_rsq, query_rsq_dfb.wait() as query_rsq:
                        with key_dfb.wait() as k0, knw_dfb.wait() as wk0, query_dfb.wait() as q0, qnw_dfb.wait() as wq0:
                            with dot_dfb.reserve() as d:
                                d.store((k0 * key_rsq * wk0) * (q0 * query_rsq * wq0))
                        with dot_dfb.wait() as dv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for j in range(HIDDEN_TILES - 1):
                            with key_dfb.wait() as kj, knw_dfb.wait() as wkj, query_dfb.wait() as qj, qnw_dfb.wait() as wqj:
                                with dot_dfb.reserve() as d:
                                    d.store((kj * key_rsq * wkj) * (qj * query_rsq * wqj))
                            with dot_dfb.wait() as dv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                                new_acc.store(av + rv)

                    # --- Gate ---
                    with acc_dfb.wait() as ds, bcast_dfb.reserve() as bc:
                        bc.store(ttl.math.broadcast(ds, dims=[1]))
                    with bcast_dfb.wait() as bv, red_dfb.reserve() as sd:
                        sd.store(bv * isd)
                    with red_dfb.wait() as sdv, gate_dfb.reserve() as g:
                        clamped = ttl.math.max(ttl.math.abs(sdv), eps)
                        g.store(ttl.math.sigmoid(sdv * ttl.math.rsqrt(clamped)))

                    # --- Gate * Value ---
                    with gate_dfb.wait() as gv:
                        for j in range(HIDDEN_TILES):
                            with value_dfb.wait() as vj:
                                with out_dfb.reserve() as o:
                                    o.store(gv * vj)

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
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                for j in range(HIDDEN_TILES):
                    with query_dfb.reserve() as blk:
                        tx = ttl.copy(query[tile_idx, j], blk); tx.wait()
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                    with knw_dfb.reserve() as blk:
                        tx = ttl.copy(key_norm_w[tile_idx, j], blk); tx.wait()
                    with query_dfb.reserve() as blk:
                        tx = ttl.copy(query[tile_idx, j], blk); tx.wait()
                    with qnw_dfb.reserve() as blk:
                        tx = ttl.copy(query_norm_w[tile_idx, j], blk); tx.wait()
                for j in range(HIDDEN_TILES):
                    with value_dfb.reserve() as blk:
                        tx = ttl.copy(value[tile_idx, j], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                for j in range(HIDDEN_TILES):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[tile_idx, j]); tx.wait()


# --- Pipe conv kernel (grid=N_CONV_CORES, forward pipe chain) ---

@ttl.kernel(grid=(N_CONV_CORES, 1))
def pipe_conv_kernel(s0, s1, s2, s3, w0, w1, w2, w3, out):
    seq_tiles = s0.shape[0] // TILE
    tiles_per_core = -(-seq_tiles // N_CONV_CORES)

    pipes = [ttl.Pipe((x, 0), ((x + 1), 0)) for x in range(N_CONV_CORES - 1)]
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
    bnd_dfb = ttl.make_dataflow_buffer_like(s0, shape=(1, HIDDEN_TILES), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        if core_x > 0:
            with bnd_dfb.wait() as bnd, acc_dfb.reserve() as ctx:
                ctx.store(bnd)
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
        if core_x > 0:
            with bnd_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk); xf.wait()
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
                    tx = ttl.copy(s0[tile_idx, 0:HIDDEN_TILES], blk); tx.wait()
                    if local_t == tiles_per_core - 1:
                        if core_x < N_CONV_CORES - 1:
                            def send(pipe):
                                xf = ttl.copy(blk, pipe); xf.wait()
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
        if core_x > 0:
            prev_tile = core_x * tiles_per_core - 1
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[prev_tile, 0:HIDDEN_TILES]); tx.wait()
        for local_t in range(tiles_per_core):
            tile_idx = core_x * tiles_per_core + local_t
            if tile_idx < seq_tiles:
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[tile_idx, 0:HIDDEN_TILES]); tx.wait()


# --- Utilities ---

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

def pad_to_tile(t, dim, tile=TILE):
    """Pad tensor along `dim` to next multiple of tile."""
    size = t.shape[dim]
    pad_size = (tile - size % tile) % tile
    if pad_size == 0:
        return t
    pad_spec = [0] * (2 * t.dim())
    pad_spec[-(2 * dim + 1)] = pad_size
    return torch.nn.functional.pad(t, pad_spec)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    torch.manual_seed(42)

    # ================================================================
    # Setup: create Engram module, tokenize input
    # ================================================================
    LAYER_ID = engram_cfg.layer_ids[0]  # first engram layer
    engram = Engram(layer_id=LAYER_ID)

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    B, L = input_ids.shape
    L_padded = ((L + TILE - 1) // TILE) * TILE
    print(f"Text: {text!r}")
    print(f"Tokens: {L}, padded to: {L_padded}")

    # Mock hidden states (same seed for both paths)
    hidden_states = torch.randn(B, L_padded, HC_MULT, HIDDEN_DIM)

    # ================================================================
    # Step 1: Hash + Embed (CPU, shared between PyTorch and TT-Lang)
    # ================================================================
    print("\n=== Step 1: Hash + Embed (CPU) ===")
    with torch.no_grad():
        hash_ids = torch.from_numpy(
            engram.hash_mapping.hash(input_ids)[LAYER_ID])
        embeddings = engram.multi_head_embedding(hash_ids).flatten(start_dim=-2)
        embeddings = pad_to_tile(embeddings, dim=1)  # pad seq to L_padded
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  embeddings[0, 0, :4]: {embeddings[0, 0, :4]}")
    print(f"  embeddings[0, 1, :4]: {embeddings[0, 1, :4]}")

    # ================================================================
    # Step 2: Key/Value projection (CPU, shared)
    # ================================================================
    print("\n=== Step 2: Key/Value Projection (CPU) ===")
    with torch.no_grad():
        value_proj = engram.value_proj(embeddings.float())
        key_projs = [engram.key_projs[i](embeddings.float()) for i in range(HC_MULT)]
    print(f"  value shape: {value_proj.shape}")
    print(f"  value[0, 0, :4]: {value_proj[0, 0, :4]}")
    print(f"  key_0[0, 0, :4]: {key_projs[0][0, 0, :4]}")

    # ================================================================
    # Step 3: PyTorch gating (reference)
    # ================================================================
    print("\n=== Step 3: PyTorch Gating ===")
    with torch.no_grad():
        py_gates = []
        for hc in range(HC_MULT):
            nk = engram.norm1[hc](key_projs[hc])
            q = hidden_states[:, :, hc, :]
            nq = engram.norm2[hc](q)
            dot = (nk * nq).sum(dim=-1) / math.sqrt(HIDDEN_DIM)
            g = dot.abs().clamp_min(C_EPS).sqrt() * dot.sign()
            g = g.sigmoid().unsqueeze(-1)
            py_gates.append(g)
        py_gates = torch.stack(py_gates, dim=2)
        py_gated = py_gates * value_proj.unsqueeze(2)
    print(f"  gate[0, 0, 0]: {py_gates[0, 0, 0, 0].item():.6f}")
    print(f"  gate[0, 1, 0]: {py_gates[0, 1, 0, 0].item():.6f}")
    print(f"  gated[0, 0, 0, :4]: {py_gated[0, 0, 0, :4]}")

    # ================================================================
    # Step 4: TT-Lang gating
    # ================================================================
    print("\n=== Step 4: TT-Lang Gating ===")
    scaler_tt = to_ttnn(torch.ones(32, 32, dtype=torch.bfloat16), device)
    ms_tt = to_ttnn(torch.full((32, 32), C_MEAN_SCALE, dtype=torch.bfloat16), device)
    isd_tt = to_ttnn(torch.full((32, 32), C_INV_SQRT_D, dtype=torch.bfloat16), device)
    eps_tt = to_ttnn(torch.full((32, 32), C_EPS, dtype=torch.bfloat16), device)

    tt_gated_list = []
    for hc in range(HC_MULT):
        key_bf16 = key_projs[hc].squeeze(0).to(torch.bfloat16).contiguous()
        query_bf16 = hidden_states.squeeze(0)[:, hc, :].to(torch.bfloat16).contiguous()
        value_bf16 = value_proj.squeeze(0).to(torch.bfloat16).contiguous()
        knw = tile_broadcast_1d(
            engram.norm1[hc].weight.data.to(torch.bfloat16), L_padded)
        qnw = tile_broadcast_1d(
            engram.norm2[hc].weight.data.to(torch.bfloat16), L_padded)

        out_tt = to_ttnn(
            torch.zeros(L_padded, HIDDEN_DIM, dtype=torch.bfloat16), device)
        engram_gate_kernel(
            to_ttnn(key_bf16, device), to_ttnn(query_bf16, device),
            to_ttnn(value_bf16, device),
            to_ttnn(knw, device), to_ttnn(qnw, device),
            scaler_tt, ms_tt, isd_tt, eps_tt, out_tt)
        tt_gated_list.append(ttnn.to_torch(out_tt))

    tt_gated = torch.stack(tt_gated_list, dim=1)  # [L_padded, HC, HIDDEN]
    print(f"  gated[0, 0, :4]: {tt_gated[0, 0, :4]}")

    # Compare gating
    py_g2d = py_gated.squeeze(0).to(torch.bfloat16).float()  # [L, HC, D]
    tt_g2d = tt_gated[:L_padded].float()
    gate_err = (py_g2d - tt_g2d).abs().max().item()
    gate_close = torch.allclose(py_g2d, tt_g2d, atol=0.5, rtol=0.1)
    print(f"  PyTorch vs TT-Lang gating max_err: {gate_err:.4f}")
    print(f"  allclose(atol=0.5): {gate_close}")

    # ================================================================
    # Step 5: PyTorch ShortConv (reference)
    # ================================================================
    print("\n=== Step 5: PyTorch ShortConv ===")
    with torch.no_grad():
        py_conv = engram.short_conv(py_gated)
    py_output = py_gated + py_conv
    print(f"  conv[0, 0, 0, :4]: {py_conv[0, 0, 0, :4]}")
    print(f"  output[0, 0, 0, :4]: {py_output[0, 0, 0, :4]}")

    # ================================================================
    # Step 6: TT-Lang Pipe Conv
    # ================================================================
    print("\n=== Step 6: TT-Lang Pipe Conv ===")
    tt_conv_list = []
    for hc in range(HC_MULT):
        gated_f = tt_gated_list[hc].float()

        # RMSNorm on CPU
        norm_w = engram.short_conv.norms[hc].weight.data.float()
        rms = torch.rsqrt((gated_f * gated_f).mean(dim=-1, keepdim=True) + 1e-5)
        normed = (gated_f * rms * norm_w.unsqueeze(0)).to(torch.bfloat16)

        shifts = [0, DILATION, 2 * DILATION, 3 * DILATION]
        shifted = [host_shift(normed, s, L_padded) for s in shifts]

        ch_start = hc * HIDDEN_DIM
        ch_end = ch_start + HIDDEN_DIM
        conv_w = engram.short_conv.conv.weight.data[ch_start:ch_end, 0, :].to(torch.bfloat16)
        weight_tiles = []
        for k in range(KERNEL_SIZE):
            w_row = conv_w[:, KERNEL_SIZE - 1 - k].unsqueeze(0).expand(TILE, -1).contiguous()
            weight_tiles.append(w_row)

        s_tts = [to_ttnn(s, device) for s in shifted]
        w_tts = [to_ttnn(w, device) for w in weight_tiles]
        out_tt = to_ttnn(
            torch.zeros(L_padded, HIDDEN_DIM, dtype=torch.bfloat16), device)
        pipe_conv_kernel(
            s_tts[0], s_tts[1], s_tts[2], s_tts[3],
            w_tts[0], w_tts[1], w_tts[2], w_tts[3], out_tt)
        tt_conv_list.append(ttnn.to_torch(out_tt))

    # Residual add
    tt_output = torch.zeros(L_padded, HC_MULT, HIDDEN_DIM)
    for hc in range(HC_MULT):
        tt_output[:, hc, :] = tt_gated_list[hc].float() + tt_conv_list[hc].float()
    print(f"  conv[0, 0, :4]: {tt_conv_list[0][0, :4]}")
    print(f"  output[0, 0, :4]: {tt_output[0, 0, :4]}")

    # ================================================================
    # Final comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: PyTorch vs TT-Lang")
    print("=" * 60)

    py_out = py_output.squeeze(0)[:L_padded].to(torch.bfloat16).float()
    tt_out = tt_output.float()

    for hc in range(HC_MULT):
        err = (py_out[:, hc, :] - tt_out[:, hc, :]).abs().max().item()
        print(f"  HC {hc}: max_err={err:.4f}")

    overall_max = (py_out - tt_out).abs().max().item()
    overall_mean = (py_out - tt_out).abs().mean().item()
    print(f"\n  Overall max_err={overall_max:.4f}, mean_err={overall_mean:.6f}")
    print(f"  Config: L={L}, L_padded={L_padded}, HIDDEN_DIM={HIDDEN_DIM}")
    print(f"  Kernels: gating(grid=auto) + pipe_conv(grid={N_CONV_CORES})")

    print(f"\n  PyTorch  output[0, 0, :6]: {py_out[0, 0, :6]}")
    print(f"  TT-Lang  output[0, 0, :6]: {tt_out[0, 0, :6]}")
    print(f"  PyTorch  output[1, 0, :6]: {py_out[1, 0, :6]}")
    print(f"  TT-Lang  output[1, 0, :6]: {tt_out[1, 0, :6]}")

    close = torch.allclose(py_out, tt_out, atol=0.5, rtol=0.1)
    if overall_max < 5.0:
        print(f"\nPASS (max_err={overall_max:.4f}, allclose={close})")
    else:
        print(f"\nFAIL (max_err={overall_max:.4f})")

    ttnn.close_device(device)
