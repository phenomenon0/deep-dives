# Inference Optimization Research Notes

## 1. FlashAttention Tiling
- Loads Q/K/V tiles into SRAM, computes partial attention, accumulates with online softmax
- Uses cp_async for overlapping data transfer with compute
- Memory: O(N) vs O(N²) for standard attention
- 10× savings at 2K seq, 20× at 4K
- FA2: 2× speedup over FA1, better parallelism, causal masking bottom-right alignment

## 2. FlashDecoding
- Parallelizes over KV sequence length (new dimension)
- Splits KV cache into n segments, each thread block processes same Q with different KV
- Separate reduction kernel combines with LSE values
- Up to 8× speedup for very large sequences
- FlashDecoding++: 2.02× over FlashDecoding (async softmax, flat GEMM)

## 3. Speculative Decoding
- P(accept) = min(1, P_target/P_draft) per token
- If rejected: resample from max(0, p(x) - q(x)) / Z
- Typical speedup: 2-2.5× with draft models
- Acceptance rate 60-80% for well-matched pairs

## 4. KV Cache Compression
- MQA/GQA: architectural reduction
- SKVQ: clipped dynamic quantization at group level, preserves recent + sink tokens in FP16
- KVC-Q: recency priority + importance preservation + head-aware allocation
- PM-KVQ: variable precision across layers

## 5. Continuous vs Static Batching
- Static: constant batch size, better peak throughput at high batch, 30-60% GPU util
- Continuous: 23× throughput improvement, 80-95% GPU util, each sequence independent latency
- Real example: 50→450 tok/s throughput, 2.5→0.8s latency, 40% cost reduction

## 6. Prefill vs Decode
- Prefill: compute-bound for seq > ~480 tokens, quadratic complexity, high arithmetic intensity
- Decode: memory-bandwidth-bound, constant tiny FLOPS, massive KV loading, linear complexity
- Can run prefill+decode in parallel on same GPU (different resource bottlenecks)

## 7. Tensor vs Pipeline Parallelism
- TP: splits within layers, all-reduce communication, no bubble, needs NVLink
- PP: splits layers across GPUs, minimal communication at boundaries, bubble time ~20-40%
- Best practice: TP intra-node, PP inter-node

## 8. Kernel Fusion
- FlashAttention as case study: fuses all attention ops into single kernel
- FA2 on Hopper: 20-50% higher FLOPS/s vs Ampere
- Common: RMSNorm+MatMul, LayerNorm+Linear, element-wise+subsequent

## 9. GPTQ Algorithm
- One-shot weight quantization using approximate second-order info
- Each row independent, quantize column-by-column with error redistribution
- Calibration: 128 random 2048-token segments from C4
- 175B in ~4 GPU hours, down to 3-4 bits

## 10. AWQ Algorithm
- 1% salient weights identified by activation distribution
- Scale important channels up (increase quant resolution), compensate in activations
- No backpropagation needed, generalizes across domains

## 11. Roofline Model
- X-axis: arithmetic intensity (FLOPS/byte), Y-axis: performance (FLOPS)
- Ridge point: intersection of compute and memory roofs
- Llama 7B decode on A10: AI≈62 ops/byte, A10 ratio≈208.3 → memory-bound
- H100: 660 TFLOPS, 3.35 TB/s → ridge at ~197 FLOPS/byte
