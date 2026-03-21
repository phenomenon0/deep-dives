# llama.cpp Architecture Research Notes

## 1. GGML Tensor Library

**Core Data Structures:**
- `ggml_tensor`: type (enum ggml_type), n_dims (1-4), ne[GGML_MAX_DIMS] (elements per dim), nb[GGML_MAX_DIMS] (stride bytes), op (enum ggml_op), src[GGML_MAX_SRC] (source tensors), data pointer, name string
- `ggml_context`: custom memory arena for sequential tensor allocation, contiguous memory
- `ggml_graph`: computation DAG where operations are nodes

**Memory Management:**
- Context-based allocation with minimal overhead
- Each context owns contiguous memory arena
- Rich type system: float + quantized types
- Hardware abstraction through pluggable backends
- Extensive SIMD optimizations

## 2. GGUF File Format

**Structure (24-byte header + metadata + tensor data):**

1. **Header:** 4-byte ASCII "GGUF", 32-bit version, num tensors, num metadata KV pairs
2. **Metadata:** length-prefixed KV pairs — architecture params (n_embd, n_layer, n_head), tokenizer info, quant type, checksums (SHA256, CRC32), RoPE config
3. **Tensor Layout:** each tensor has metadata header (name, shape, quant type, block sizes, disk offsets), followed by quantized weights (blockwise packed), quant tables + scaling constants
4. **Alignment:** tensor metadata padded to 8 bytes, tensor data padded to 16 bytes (mmap/cache-line optimal), all multi-byte fields little-endian

## 3. Quantization Methods (Bit-Level)

**Q4_0 (symmetric):** one float scale per block, w = d * q, simple + fast to decode, good for CPU
**Q4_1 (affine/asymmetric):** adds per-block offset/zero-point, better for asymmetric distributions
**Q4_K_M (K-means, medium):** superblocks of 8 blocks × 32 weights = 4.5 bits/weight, K-means optimized, extra scaling factor per superblock for outliers
**Q5_K_M:** similar superblock structure, 5.5 bits/weight, higher precision
**Size variants:** S=minimal size, M=higher precision on essential tensors, L=higher precision on more tensors
**Best balance:** Q4_K_M

## 4. Inference Loop

**Two-Stage Processing:**
1. **Prompt Processing (PP):** entire prompt fed through model at once
2. **Token Generation (TG):** one token at a time (autoregressive)

**Main Functions:**
- `llama_eval()`: takes n_eval tokens from embd starting at position i, runs forward pass
- `llama_decode()`: modern API — forward-pass given inputs in `_batch`
- Uses `n_ctx` (context buffer) and `n_past` (context position)

**Loop:** input tokens → logits → sample token → feed back → repeat until EOS or max tokens

## 5. Backend System

**Architecture:**
- `ggml_backend` wraps `ggml_backend_i` interface with function pointers
- Backend registry for discovery + management
- Dynamic loading for runtime selection

**Backends:** CPU (AVX, AVX2, AVX512, AMX, ARM NEON), CUDA (+ HIP for AMD), Metal (Apple), Vulkan (cross-platform), SYCL (Intel), ROCm (AMD), MUSA (Moore Threads)

**Multi-Backend:** simultaneous backends, dynamic libraries, runtime selection via `--device`, scoring mechanism for auto-selection

## 6. Memory Mapping (mmap)

- Lazy loading — only needed parts loaded, kernel manages physical pages
- Avoids page copying (prevents kernel cache eviction)
- Dramatic memory reduction (40GB model → 20GB working set)
- Makes large models viable on modest hardware
- Disabling: slower loads, may reduce pageouts without --mlock

## 7. KV Cache

**Structure:** ring buffer by layers + streams, separate K and V tensors per layer, multiple sequences share cells
**Cells:** each = single token position, track sequences via bitsets, ring buffer allocation with head pointer
**Memory:** shift operation applies RoPE adjustments to K cache, supports per-sequence or unified caching

## 8. Batch Processing

**llama_batch:** arbitrary set of tokens, each with position + sequence ID(s), determines attention relationships, constructs KQ_mask
**Context:** ring buffer by layers/streams, each sequence has own context, tokens see only same-sequence tokens
**Params:** n_batch (max PP batch size), n_ctx (context token count)

## 9. Optimization Techniques

**SIMD:** weight layouts reordered for concurrent column access, fast per-group dequant via MSB toggling, dequant fused with matmul
**Quantized GEMM:** fused kernels on quantized data (no explicit dequant), CPU: activations→q8, vecdot(q4,q8), MUL_MAT = 95% of execution time
**Architecture-specific:** KleidiAI (ARM), Arm I8MM smmla instruction, significant gains on Neoverse-N2
**Flash Attention:** IO-aware tiling, reduces HBM reads/writes, supports on-the-fly dequant of Q/K/V, across CUDA/Metal/Vulkan/SYCL, example: 11→32 tok/sec on M3 Max
**CUDA Graphs:** up to 1.2x speedup (H100 Llama 7B), reduced scheduling overhead, updates via cudaGraphExecUpdate

## 10. Sampling Pipeline

**Default order:** penalties → dry → top_n_sigma → top_k → typ_p → top_p → min_p → xtc → temperature
**Temperature:** logits / temp → softmax → probabilities (applied last, after truncation filters noise)
**Top-P (nucleus):** cumulative probability threshold, 0.95=diverse, 0.5=focused
**Min-P:** relative minimum probability threshold vs most likely token

## 11. Key Data Structures

- `llama_model`: weights, metadata, vocabulary, device allocation, `llama_model::impl` has buffer lists + device mappings
- `llama_context`: runtime state — memory mgmt, computation graphs, backend coordination, KV cache init via `llama_new_context_with_model`

## 12. Attention Mechanism

- Three vectors per embedding: K, Q, V (via wk, wq, wv matrices)
- QK^T softmax V formula
- Self-attention = only cross-token computation
- Single-token inference MACs ≈ 2·S·d (S=seq length, d=model dim)

## 13. RoPE (Rotary Position Embedding)

- Polar coordinate system (angles + complex numbers)
- High-dim vectors decomposed into pairs for rotation
- Integrated into MHA layer (not embedding layer)
- GPT-J style: reshape to complex → multiply by frequency → convert back
- NTK/YaRN scaling for 256k+ tokens
- Rotation applied once when writing KV cache

## 14. Computation Engine

- `ggml_backend_sched` dispatches to hardware backends
- `llama_build_graph` + `ggml_backend_sched_split_graph` construct DAGs
- Translated to CUDA graphs for execution
- Graph updates via cudaGraphExecUpdate when context size changes

## 15. Supported Models

LLaMA 1/2/3, Mistral (7B, Nemo), Phi/PhiMoE, Qwen (2, 2.5, 3, multimodal), Gemma, Yi, DeepSeek, Falcon, MPT, Bloom, StableLM, many more
