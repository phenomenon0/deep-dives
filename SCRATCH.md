# Inference Engine Deep Dive — Scratch Pad

## Progress Tracker

| Chapter | Status | Commit | Notes |
|---------|--------|--------|-------|
| Part I tweaks | DONE | — | Already present in existing file |
| Ch 05 - Why Inference Is Slow | DONE | 03c07c8 | Interactive roofline D3, GPU specs table, decode/prefill analysis |
| Ch 06 - The KV Cache | DONE | pending | KV growth D3 viz, GQA table, PagedAttention deep-dive, KV quantization |
| Ch 07 - Quantization | pending | — | |
| Ch 08 - FlashAttention | pending | — | |
| Ch 09 - Continuous Batching | pending | — | |
| Ch 10 - Prefill vs Decode | pending | — | |
| Ch 11 - Speculative Decoding | pending | — | |
| Ch 12 - The Engines | pending | — | |
| Ch 13 - One User Becomes Many | pending | — | |
| Ch 14 - The Scheduler | pending | — | |
| Ch 15 - Prefix Caching | pending | — | |
| Ch 16 - Disaggregated P/D | pending | — | |
| Ch 17 - Parallelism | pending | — | |
| Ch 18 - LoRA & Multi-Tenancy | pending | — | |
| Ch 19 - Metrics That Matter | pending | — | |
| Ch 20 - Modern Serving Stack | pending | — | |

## Research Notes

### Chapter 05 — Why Inference Is Slow
- **GPU specs verified**: RTX 3090 (142 TFLOPS, 936 GB/s), RTX 4090 (165 TFLOPS, 1008 GB/s), A100 (312 TFLOPS, 2039 GB/s), H100 (989 TFLOPS dense FP16, 3350 GB/s)
- **Key insight**: Decode AI ≈ 1 FLOP/byte at batch=1, independent of model size. 153× below A100 ridge point.
- **H100 paradox**: 3.2× compute over A100, but only 1.6× bandwidth → less utilized during decode
- **Prefill crossover**: Ridge at ~153 seq_len for A100, ~295 for H100
- **Sources**: NVIDIA datasheets, kipply's transformer inference arithmetic, JAX Scaling Book roofline chapter, vLLM internals (PagedAttention, chunked prefill, CUDA graphs)
- **D3 viz**: Interactive roofline with GPU selector, model size, batch slider, prefill slider. Decode (red) and prefill (green) dots move on the roofline.

## Design Decisions
- Era markers: small badge at chapter top, same style as Part I's pipeline position indicator
- Part II file: complete rebuild, fresh CSS (copy from Part I base)
- Nav: cross-links between all 3 parts
