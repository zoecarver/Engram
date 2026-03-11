# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end Engram forward pass on TT hardware.

Flow:
  1. Host: key_proj, value_proj (linear projections)
  2. TT-Lang gating kernel (grid=auto): RMSNorm, dot-product gate, gate*value
     All (1,1) tile DFBs with loops over hidden dim.
  3. Host: RMSNorm + pre-shift for conv
  4. TT-Lang pipe conv kernel (grid=N_CONV_CORES): weighted sum + SiLU with
     inter-core boundary sharing via PipeNet
  5. Host: residual add

Validates against PyTorch Engram.forward() output.
"""

import torch
import torch.nn as nn
import math
import ttnn
import ttl

TILE = 32

ENGRAM_TILES = 32
HIDDEN_TILES = 32
ENGRAM_DIM = ENGRAM_TILES * TILE
HIDDEN_DIM = HIDDEN_TILES * TILE

HC_MULT = 4
KERNEL_SIZE = 4
DILATION = 3
N_CONV_CORES = 4

C_MEAN_SCALE = 1.0 / HIDDEN_DIM
C_INV_SQRT_D = 1.0 / math.sqrt(HIDDEN_DIM)
C_EPS = 1e-6


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
        gated = gates * value

        B, T, G, C = gated.shape
        normed_chunks = [self.conv_norms[i](gated[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.silu(self.conv(x_bct)[..., :T])
        conv_out = y_bct.transpose(1, 2).view(B, T, G, C)
        return gated + conv_out


# --- Gating kernel (streaming, grid=auto, (1,1) tile DFBs) ---

@ttl.kernel(grid="auto")
def engram_gate_kernel(key, query, value, key_norm_w, query_norm_w,
                       scaler, mean_scale, inv_sqrt_d, eps_tile, out):
    """RMSNorm key/query, dot-product gate, gate*value.

    All DFBs are (1,1) to work around non-square matmul compiler bug (#383).
    Loops over HIDDEN_TILES for reduce/normalize.
    Per position: 2 passes over key (squares + normalize),
    2 passes over query, 1 pass over value for output.
    """
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
                    # Broadcast col 0 -> all cols, then scale + rsqrt
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
                    # Broadcast col 0 -> all cols, then scale + rsqrt
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
                        # First tile
                        with key_dfb.wait() as k0, knw_dfb.wait() as wk0, query_dfb.wait() as q0, qnw_dfb.wait() as wq0:
                            with dot_dfb.reserve() as d:
                                d.store((k0 * key_rsq * wk0) * (q0 * query_rsq * wq0))
                        with dot_dfb.wait() as dv, red_dfb.reserve() as r:
                            r.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        # Remaining tiles
                        for j in range(HIDDEN_TILES - 1):
                            with key_dfb.wait() as kj, knw_dfb.wait() as wkj, query_dfb.wait() as qj, qnw_dfb.wait() as wqj:
                                with dot_dfb.reserve() as d:
                                    d.store((kj * key_rsq * wkj) * (qj * query_rsq * wqj))
                            with dot_dfb.wait() as dv, red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(dv, sc, dims=[0]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as av, acc_dfb.reserve() as new_acc:
                                new_acc.store(av + rv)

                    # --- Gate ---
                    # Broadcast col 0 -> all cols (reduce leaves garbage in cols 1-31)
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
                # Key pass 1: H tiles for sum of squares
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                # Query pass 1: H tiles for sum of squares
                for j in range(HIDDEN_TILES):
                    with query_dfb.reserve() as blk:
                        tx = ttl.copy(query[tile_idx, j], blk); tx.wait()
                # Normalize + dot: key, knw, query, qnw interleaved
                for j in range(HIDDEN_TILES):
                    with key_dfb.reserve() as blk:
                        tx = ttl.copy(key[tile_idx, j], blk); tx.wait()
                    with knw_dfb.reserve() as blk:
                        tx = ttl.copy(key_norm_w[tile_idx, j], blk); tx.wait()
                    with query_dfb.reserve() as blk:
                        tx = ttl.copy(query[tile_idx, j], blk); tx.wait()
                    with qnw_dfb.reserve() as blk:
                        tx = ttl.copy(query_norm_w[tile_idx, j], blk); tx.wait()
                # Value: H tiles
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

    scaler_tt = to_ttnn(torch.ones(32, 32, dtype=torch.bfloat16), device)
    ms_tt = to_ttnn(torch.full((32, 32), C_MEAN_SCALE, dtype=torch.bfloat16), device)
    isd_tt = to_ttnn(torch.full((32, 32), C_INV_SQRT_D, dtype=torch.bfloat16), device)
    eps_tt = to_ttnn(torch.full((32, 32), C_EPS, dtype=torch.bfloat16), device)

    emb_2d = embeddings_torch.squeeze(0)

    # Host: value projection (shared across HC groups)
    with torch.no_grad():
        value_projected = module.value_proj(emb_2d.float()).to(torch.bfloat16)

    # Step 1: Gating for all 4 HC groups
    print("Step 1: Gating...")
    gated_outputs = []
    for hc_idx in range(HC_MULT):
        # Host: key projection
        with torch.no_grad():
            key_projected = module.key_projs[hc_idx](emb_2d.float()).to(torch.bfloat16)

        query_slice = hidden_states_torch.squeeze(0)[:, hc_idx, :].to(torch.bfloat16).contiguous()
        knw = tile_broadcast_1d(module.norm1[hc_idx].weight.data.to(torch.bfloat16), SEQ)
        qnw = tile_broadcast_1d(module.norm2[hc_idx].weight.data.to(torch.bfloat16), SEQ)

        out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)
        engram_gate_kernel(
            to_ttnn(key_projected, device),
            to_ttnn(query_slice, device),
            to_ttnn(value_projected, device),
            to_ttnn(knw, device), to_ttnn(qnw, device),
            scaler_tt, ms_tt, isd_tt, eps_tt, out_tt)
        gated_outputs.append(ttnn.to_torch(out_tt))

    # Step 2: Pipe conv per HC group
    print("Step 2: Pipe conv...")
    conv_outputs = []
    for hc_idx in range(HC_MULT):
        gated = gated_outputs[hc_idx].float()

        # RMSNorm on CPU
        norm_w = module.conv_norms[hc_idx].weight.data.float()
        rms = torch.rsqrt((gated * gated).mean(dim=-1, keepdim=True) + 1e-5)
        normed = (gated * rms * norm_w.unsqueeze(0)).to(torch.bfloat16)

        # Pre-shift on host
        shifts = [0, DILATION, 2 * DILATION, 3 * DILATION]
        shifted = [host_shift(normed, s, SEQ) for s in shifts]

        # Conv weights: w[k] pairs with in[t - padding + k*D]
        # k=0->in[t-9]=s3, k=1->in[t-6]=s2, k=2->in[t-3]=s1, k=3->in[t]=s0
        # So weight for s_j is w[KERNEL_SIZE-1-j]
        ch_start = hc_idx * HIDDEN_DIM
        ch_end = ch_start + HIDDEN_DIM
        conv_w = module.conv.weight.data[ch_start:ch_end, 0, :].to(torch.bfloat16)

        weight_tiles = []
        for k in range(KERNEL_SIZE):
            w_row = conv_w[:, KERNEL_SIZE - 1 - k].unsqueeze(0).expand(TILE, -1).contiguous()
            weight_tiles.append(w_row)

        s_tts = [to_ttnn(s, device) for s in shifted]
        w_tts = [to_ttnn(w, device) for w in weight_tiles]
        out_tt = to_ttnn(torch.zeros(SEQ, HIDDEN_DIM, dtype=torch.bfloat16), device)

        pipe_conv_kernel(s_tts[0], s_tts[1], s_tts[2], s_tts[3],
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
    print(f"Config: SEQ={SEQ}, ENGRAM_DIM={ENGRAM_DIM}, HIDDEN_DIM={HIDDEN_DIM}")
    print(f"Kernels: gating(grid=auto) + pipe_conv(grid={N_CONV_CORES})")

    if overall_max < 5.0:
        print("PASS: Full E2E Engram (gating + pipe conv + residual)")
    else:
        print("FAIL: E2E mismatch")

    ttnn.close_device(device)
