# vLLM Architecture Research Notes

## 1. PagedAttention
- Divides KV cache into fixed-size "blocks" (pages), each stores KV for fixed number of tokens
- Block Table maps logical block indices to physical memory locations (like page tables)
- Key classes: `PagedAttention` in `vllm/v1/attention/ops/paged_attn.py`
  - `split_kv_cache()`, `write_to_paged_cache()` via `ops.reshape_and_cache`
- CUDA kernels: `paged_attention_v1_kernel`, `paged_attention_v2_kernel`

## 2. Scheduler
- Queues: waiting, running, swapped (preempted to CPU)
- Key methods: `schedule()` → `SchedulerOutput`, `update_from_output()`, `add_request()`
- Preemption: default RECOMPUTE (not SWAP in V1), selects lowest-priority running request
- Priority: configurable via `--scheduling-policy`, `PriorityRequestQueue` with FIFO tie-breaking
- States: WAITING → RUNNING → FINISHED_* or PREEMPTED → WAITING

## 3. Continuous Batching
- V1 unified scheduler: both prompt and output tokens treated similarly
- Chunked prefill: large prefills split, batched with decode requests
- Decode requests prioritized first, then prefill if token budget allows

## 4. Execution Engine
- `GPUModelRunner`: init model, manage KV cache, invoke forward pass, sample tokens
- `InputBatch`: batched representation of multiple requests (token IDs, positions, KV mappings, sampling params)
- CUDA Graph dispatch: `CudagraphDispatcher` → `BatchDescriptor` → `CUDAGraphWrapper` (capture/replay)
- Modes: NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE (default V1)

## 5. Parallelism
- TP: `RowParallelLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`
- PP: `Base.pipeline_parallel()` partitions `nn.ModuleList` across stages
- Workers: API Server → Engine Core → GPU Workers (TP×PP per engine core)
- Communication: ZMQ sockets, NCCL for tensor transfer, Ray for multi-node

## 6. Memory Management
- `KVCacheManager` → `KVCacheCoordinator` → `SingleTypeKVCacheManager`
- `BlockPool`: manages `KVCacheBlock` objects, doubly-linked `FreeKVCacheBlockQueue` (LRU)
- `allocate_slots()`: calculate needed, free unnecessary, allocate from pool
- Prefix caching: hash-based lookup (`cached_block_hash_to_block`), ref counting

## 7. Workers
- `WorkerBase` abstract → `GPUWorker` (one per GPU)
- Executors: `UniProcExecutor`, `MultiprocExecutor` (multiprocessing), `RayDistributedExecutor`

## 8. Speculative Decoding Proposers
- N-gram, Draft Model, EAGLE/EAGLE3, Medusa, Suffix Decoding
- `RejectionSampler`: accepted + recovered + bonus tokens

## 9. Quantization
- AWQ, GPTQ (2/3/4/8-bit), FP8 (W8A8, Hopper+), BitsAndBytes, GGUF, INT8
- KV cache quantization: `kv_cache_dtype="fp8"`, per-tensor or per-head schemes

## 10. API Server
- FastAPI, OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings
- `OpenAIServingChat`, `OpenAIServingCompletion` → `AsyncLLMEngine`
- SSE streaming via `StreamingResponse` + `AsyncGenerator`

## 11. torch.compile Kernel Fusions
- AllReduce + RMSNorm (`fuse_allreduce_rms`)
- Attention + Quantization (`fuse_attn_quant`)
- RoPE + KV-Cache Update (`fuse_rope_kvcache`)
- Sequence Parallelism (`enable_sp`)
- AsyncTP GEMM + Collective Overlap (`fuse_gemm_comms`)
- QK Norm + RoPE (`enable_qk_norm_rope_fusion`)
- RMSNorm + Quantization (`fuse_norm_quant`)
- SiLU+Mul + Quantization (`fuse_act_quant`)

## 12. V1 vs V0
- V1: multi-process architecture (API Server, Engine Core, GPU Workers)
- V1: unified scheduler (no separate prefill/decode phases)
- V1: RECOMPUTE preemption (not SWAP)
- V1: chunked prefill default, prefix caching default, torch.compile default
- Removed: best_of, per-request logits processors, GPU↔CPU KV cache swapping
