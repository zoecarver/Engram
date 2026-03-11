# Engram on Tenstorrent

Port of the [Engram](https://github.com/deepseek-ai/Engram) conditional memory module to [TT-Lang](https://github.com/tenstorrent/tt-lang), running on Tenstorrent Wormhole hardware.

Uses the same weights, tokenizer, and input text as `engram_demo_v1.py` (the original PyTorch reference) and produces numerically matching outputs at every stage.

## Kernels

Both kernels are in `engram_demo_ttlang.py` and leverage the following patterns:

- **Streaming** -- data flows through small tile-sized dataflow buffers, so sequence length and hidden dimension are bounded only by DRAM, not L1. The same kernel handles 32 tokens or 32K tokens.
- **Automatic multicore** -- `grid="auto"` distributes sequence positions across all available Tensix cores. Each core streams its partition independently.
- **Inter-core pipes** -- the conv kernel uses `PipeNet` to forward boundary tiles between cores, enabling overlap-aware convolution without redundant DRAM reads.

### Gating Kernel (`engram_gate_kernel`)

Fused RMSNorm + dot-product gate + gated value projection, streaming one sequence position at a time across all cores (`grid="auto"`).

Each position loops over 32 hidden-dimension tiles (1024-dim) using `(1,1)` tile dataflow buffers:

1. **RMSNorm key** -- two passes: accumulate per-row sum-of-squares across 32 tiles, then normalize with broadcast rsqrt
2. **RMSNorm query** -- same pattern
3. **Dot-product gate** -- interleaved normalize + element-wise multiply + reduce, fused into a single loop over hidden tiles
4. **Gate * Value** -- broadcast scalar gate across all 32 value tiles

The original PyTorch reference (`engram_demo_v1.py`) does `key_proj` and `value_proj` as `nn.Linear` layers. These run on the host; the TT-Lang kernel handles everything after projection.

### Pipe Conv Kernel (`pipe_conv_kernel`)

Depthwise conv1d (kernel_size=4, dilation=3) with SiLU activation on a fixed 4-core grid with inter-core boundary sharing via `PipeNet`.

The host pre-shifts the input by `[0, 3, 6, 9]` positions (causal dilation offsets) and the kernel computes a streaming weighted sum + SiLU. After processing its last input tile, each core pipes the boundary tile to the next core for overlap handling.

The original reference uses `nn.Conv1d` with `groups=total_channels` (depthwise). The TT-Lang version decomposes this into explicit shift + weight + fuse, which maps naturally to the streaming tile model.

## Running

```
python engram_demo_ttlang.py
```

Requires `engram_demo_v1.py` (the original PyTorch reference) in the same directory.

## Output

```
Text: 'Only Alexander the Great could tame the horse Bucephalus.'
Tokens: 16, padded to: 32

=== Step 1: Hash + Embed (CPU) ===
  embeddings shape: torch.Size([1, 32, 1024])
  embeddings[0, 0, :4]: tensor([-0.5258, -0.1628,  1.4012, -0.0337])
  embeddings[0, 1, :4]: tensor([ 0.6422, -0.8383, -1.6465, -0.0594])

=== Step 2: Key/Value Projection (CPU) ===
  value shape: torch.Size([1, 32, 1024])
  value[0, 0, :4]: tensor([-0.9547,  0.4807, -0.1513, -1.5147])
  key_0[0, 0, :4]: tensor([ 0.4311, -0.5420,  0.3825, -0.5490])

=== Step 3: PyTorch Gating ===
  gate[0, 0, 0]: 0.311713
  gate[0, 1, 0]: 0.222689
  gated[0, 0, 0, :4]: tensor([-0.2976,  0.1498, -0.0472, -0.4722])

=== Step 4: TT-Lang Gating ===
  gated[0, 0, :4]: tensor([-0.2969,  0.1494, -0.0471, -0.4727], dtype=torch.bfloat16)
  PyTorch vs TT-Lang gating max_err: 0.0195
  allclose(atol=0.5): True

=== Step 5: PyTorch ShortConv ===
  conv[0, 0, 0, :4]: tensor([-0.2045, -0.1131, -0.0084, -0.0548])
  output[0, 0, 0, :4]: tensor([-0.5021,  0.0368, -0.0555, -0.5270])

=== Step 6: TT-Lang Pipe Conv ===
  conv[0, 0, :4]: tensor([-0.2021, -0.1118, -0.0083, -0.0547], dtype=torch.bfloat16)
  output[0, 0, :4]: tensor([-0.4990,  0.0376, -0.0554, -0.5273])

============================================================
FINAL COMPARISON: PyTorch vs TT-Lang
============================================================
  HC 0: max_err=0.0391
  HC 1: max_err=0.0399
  HC 2: max_err=0.0377
  HC 3: max_err=0.0548

  Overall max_err=0.0548, mean_err=0.002232
  Config: L=16, L_padded=32, HIDDEN_DIM=1024
  Kernels: gating(grid=auto) + pipe_conv(grid=4)

  PyTorch  output[0, 0, :6]: tensor([-0.5039,  0.0369, -0.0554, -0.5273, -0.1230,  0.1162])
  TT-Lang  output[0, 0, :6]: tensor([-0.4990,  0.0376, -0.0554, -0.5273, -0.1234,  0.1152])
  PyTorch  output[1, 0, :6]: tensor([-0.0889,  0.1035, -0.2168,  0.0260, -0.0354,  0.0874])
  TT-Lang  output[1, 0, :6]: tensor([-0.0881,  0.1025, -0.2180,  0.0260, -0.0354,  0.0859])

PASS (max_err=0.0548, allclose=True)
```
