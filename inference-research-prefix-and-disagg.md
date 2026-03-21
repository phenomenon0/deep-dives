# LLM Serving at Scale: Research Notes (Prefix Caching & Disaggregated Serving)

Research for Ch 15 (Prefix Caching) and Ch 16 (Disaggregated Prefill/Decode).

---

## TOPIC 1: Prefix Caching

### How Shared Prefixes Arise Naturally

Every production LLM deployment has massive prefix overlap. The patterns:

1. **System prompts**: every request starts with the same 500-2000 token instructions.
   10,000 requests/hour * 1,000 system prompt tokens = 10M redundant prefill tokens.
2. **Few-shot examples**: same demonstrations prepended to every request in a pipeline.
3. **RAG contexts**: retrieved documents overlap across users asking similar questions.
4. **Multi-turn chat**: turn N's prompt is turn N-1's prompt + response + new message.
   The entire conversation history is a prefix of the next request.
5. **Tool-use scaffolds**: function definitions (often 2,000-5,000 tokens) repeated on
   every API call for agents using tool calling.
6. **Agentic workflows**: massive static contexts (tools, step-history) with 100:1+
   input-to-output ratios. Prefix reuse is essential for computational viability.

**The economics**: cached tokens cost ~10x less than uncached tokens in production
pricing. Claude Sonnet: $0.30/M cached vs $3.00/M uncached. This isn't charity --
it reflects real compute savings.

### vLLM Automatic Prefix Caching (APC)

**Core mechanism**: hash KV cache blocks by their token content, reuse blocks on
cache hits, skip prefill computation for the matched prefix.

**Hash-based block identification** (SHA-256):

Each KV cache block gets a `BlockHash` computed from:
1. **Parent block hash** -- forming a hash chain (so "hello world" at position 0 and
   position 1000 hash differently even though the tokens are identical)
2. **Current block token IDs** -- the actual content of this block
3. **Extra keys**: multimodal identifiers, LoRA request names, cache salt for
   request isolation, prompt embeddings hash

The hash chain is critical: it means block identity depends on *position in the
sequence*, not just token content. Two identical token subsequences at different
positions won't collide.

**Block pooling architecture** (`BlockPool`):
- All physical blocks indexed by block ID
- `FreeKVCacheBlockQueue` -- available blocks in LRU order
- `BlockHashToBlockMap` -- maps `BlockHashWithGroupId` to `KVCacheBlock`

**Cache hit behavior**:
1. New request arrives with token IDs
2. Divide into blocks, compute hash chain for each block
3. Look up each block hash in `BlockHashToBlockMap`
4. If found: reuse that KV cache block (skip prefill for those tokens)
5. If not found: compute normally, register the new blocks for future reuse
6. Partial hits work -- first N blocks might be cached, remaining computed

**LRU eviction**:
- When free queue is exhausted, evict least recently used blocks
- Priority: evict blocks at the *end* of a chain first (leaf blocks) when access
  times are equal
- Only blocks with `ref_cnt == 0` (no active request using them) are eligible
- Eviction doesn't cascade -- if a leaf block is evicted, its parent stays
  until it too becomes LRU

**Performance characteristics**:
- APC reduces prefilling time only -- decode phase unchanged
- vLLM calls it "a free lunch": no measurable overhead when cache misses
- TTFT improvement on repeated requests with ~10K token prompts:
  **4.3 seconds --> 0.6 seconds** (7x speedup)
- Enable with: `enable_prefix_caching=True` in engine config
- Default block size: 16 tokens per block

**When APC helps vs. doesn't help**:
- Helps: shared system prompts, multi-turn chat, RAG with document overlap
- Doesn't help: unique prompts with no shared prefixes, decode-dominated workloads
  (long outputs), very short prompts where prefill is already fast

### SGLang RadixAttention

**Core difference from vLLM**: token-level radix tree vs. block-level hashing.

**Radix tree data structure**:
A radix tree (compact prefix tree) where edges can be labeled with sequences of
tokens of varying lengths. The KV cache tensors are stored on the GPU in a paged
layout, one token per page.

Instead of discarding KV cache after completing a request, SGLang retains both
prompt and generation KV cache in the radix tree. This means:
- Generated output tokens from request A can be reused if request B's prompt
  starts with request A's prompt + output
- This is especially powerful for multi-turn chat and tree-of-thought

**How it works**:
1. New request arrives
2. Traverse the radix tree, matching the request's token sequence
3. Match found: reuse KV cache for the matched prefix
4. Match ends: compute from the divergence point onward
5. After generation: insert the new tokens (prompt + output) into the tree

**Eviction**: LRU on leaf nodes, recursively. CPU-based with "small maintenance
overhead." No noticeable overhead even when cache misses are dominant, so
RadixAttention stays permanently enabled.

**Fork/merge semantics**: parallel generation requests that share a prefix
"branch" from the same node. This enables efficient tree-of-thought, where
multiple continuations share a common reasoning prefix.

**Performance**: up to 5x higher throughput vs. vLLM v0.2.5 and Guidance v0.1.8
on benchmarks including MMLU, HellaSwag, ReAct Agent, Tree-of-Thought, JSON
extraction, chat, DSPy RAG, and LLaVA vision tasks. Tested on Llama-7B (A10G)
and Mixtral-8x7B (8 GPUs).

### SGLang vs vLLM: The Granularity Tradeoff

| Aspect | vLLM (block-level hashing) | SGLang (token-level radix tree) |
|--------|---------------------------|-------------------------------|
| Granularity | Fixed blocks (e.g., 16 tokens) | Token-level |
| Match type | Exact prefix match per block | Partial overlap detection |
| Cache output tokens? | No (prompt only by default) | Yes (prompt + generation) |
| Best for | Templated batch inference | Multi-turn chat, agents |
| Overhead | Near-zero on miss | Near-zero on miss |

**Benchmark comparison** (DeepSeek-R1-Distill-Llama-70B, dual H100 SXM):
- Fresh 7K context: SGLang 29.5 tok/s vs vLLM 28.6 tok/s
- Cached 7K context: SGLang ~35.0 tok/s (20% gain) vs vLLM ~32.8 tok/s (15% gain)
- Net advantage: SGLang ~10% faster on cached multi-turn scenarios

**Decision heuristic**:
- SGLang: conversational AI, unpredictable dialog flows, agentic loops
- vLLM: batch inference with predictable templates, structured workflows

### Prefix Caching at Cluster Scale (llm-d)

Single-instance prefix caching breaks at cluster scale because standard load
balancers scatter related requests across pods, destroying cache locality.

**llm-d's solution** (Kubernetes-native):

1. **KVEvents**: continuous event streams from each pod reporting which KV blocks
   they hold. Two layers:
   - `kvevents.Pool`: maintains a KV-Block Index mapping block-hashes to pods
   - `kvcache.Index`: higher-level index for what % of a request's prefix exists
     on each accessible pod

2. **Precise prefix-cache scorer**: queries cache index, outputs "cache affinity
   score" per pod, combines with load-aware metrics for routing.

3. **Memory efficiency**: managing a **365 GB** cache pool requires only **339 KB**
   of scheduler metadata -- a 1,000,000:1 data-to-metadata ratio.

**Performance** (150 enterprise customers, 6K-token contexts, 5 concurrent users each):
- Precise scheduling: P90 TTFT = **0.542s** (mean 0.298s)
- Approximate scheduling: P90 TTFT = 31.083s (**57x slower**)
- Random scheduling: **170x slower**
- Throughput: 8,730 output tokens/s (25% over approximate, 2x over cache-blind)

**Adoption**: Alibaba Cloud integrates this routing into Container Service for
Kubernetes. DaoCloud uses it for their MaaS platform with P/D disaggregation.

### Economics Summary

Without prefix caching for a system prompt workload:
- 10,000 requests * 1,000 system prompt tokens = **10M prefill tokens**

With prefix caching:
- 1 * 1,000 tokens prefill + 10,000 * (unique suffix only)
- If average unique suffix is 200 tokens: 1K + 2M = **2.001M tokens** (5x reduction)

Production cache hit rates: **60-90%** for chat applications. Higher for agents
with fixed tool definitions.

---

## TOPIC 2: Disaggregated Prefill/Decode

### Why Disaggregate: The Phase Mismatch

Prefill and decode have fundamentally different hardware profiles:

| Property | Prefill | Decode |
|----------|---------|--------|
| Bottleneck | Compute (matrix multiply over full prompt) | Memory bandwidth (read weights for 1 token) |
| GPU utilization | High (can saturate FLOPs) | Low (waiting on memory reads) |
| Parallelism | Lower TP sufficient | Higher TP / more instances needed |
| Latency metric | TTFT (time to first token) | ITL (inter-token latency) |
| Batch behavior | Few large prefills | Many small decodes |

**The interference problem**: when a long prefill joins a batch of decodes on the
same GPU, all ongoing decodes stall. Decode latency inflates by **2x-30x**,
especially under bursty workloads. A 4K-token prefill can stall 50 ongoing
decodes for hundreds of milliseconds.

**Coupled scaling problem**: if prefill and decode share GPUs, you must provision
for the worst case of *both* phases simultaneously. This leads to over-provisioning
and poor utilization.

### How Disaggregation Works

```
Client request
    |
    v
[Router/Scheduler]
    |
    v
[Prefill Instance] --process prompt--> KV cache generated
    |
    | KV cache transfer (RDMA / NVLink / TCP)
    v
[Decode Instance] --generate tokens--> stream back to client
```

1. Request arrives at router
2. Router sends to a prefill-optimized GPU pool
3. Prefill instance processes the full prompt, produces KV cache
4. KV cache is transferred to a decode-optimized GPU pool
5. Decode instance generates tokens using the transferred KV cache
6. Tokens stream back to the client

### KV Cache Transfer: The Critical Path

**KV cache size per token** (Llama-3.1-70B):
- 80 layers * 8 KV heads * 128 head_dim * 2 (K and V) * 2 bytes (FP16)
- = **327,680 bytes per token** (~320 KB)
- For a 4K token prompt: **~1.34 GB** of KV cache to transfer

**Transfer speeds by interconnect**:

| Connection | Bandwidth | 1.34 GB transfer time |
|-----------|-----------|----------------------|
| 1 GbE | 125 MB/s | 10.7 seconds |
| 10 GbE | 1.25 GB/s | 1.07 seconds |
| 25 GbE | 3.125 GB/s | 430 ms |
| 100 GbE | 12.5 GB/s | 110 ms |
| InfiniBand HDR | 25 GB/s | 54 ms |
| NVLink (intra-node) | 600 GB/s | 2.2 ms |

**Practical requirement**: for ~500ms target TTFT with 200ms prefill, you have
~300ms for KV transfer, requiring **~4.5 GB/s minimum** -- meaning 100 GbE or
InfiniBand is effectively required for production disaggregation.

**Transfer libraries**:
- **NVIDIA NIXL**: unifies NVLink, InfiniBand, PCIe, and SSD under one abstraction.
  Hardware-agnostic point-to-point communication via UCX, GPUDirect Storage, S3.
- **DeepSeek 3FS**: combines thousands of SSDs and network bandwidth for
  locality-oblivious storage access
- **P/D-Serve**: shifts from block-fixed to contiguous buffer transfers, enabling
  single-burst device-to-device migration over RDMA (RoCE). 46% reduction in
  transfer time, 60% throughput improvement at scale.

### DistServe (OSDI 2024)

The foundational paper. Zhong et al., UC Berkeley.

**Key idea**: assign prefill and decode to different GPUs, co-optimize resource
allocation and parallelism strategy independently for each phase.

**Results** (A100-80GB, synthetic workloads, 512-token input, 64-token output):
- **7.4x more requests** served within SLO constraints
- **12.6x tighter SLO compliance** (P90 latency targets)
- **4.48x goodput improvement** over co-located systems
- **20x reduction in latency variance** between phases

**Placement algorithm**: considers interconnect bandwidth between prefill and
decode nodes to minimize KV transfer overhead. Places communicating pairs on
nodes with high-bandwidth links.

**Independent parallelism**: prefill pool might use TP=2 for compute efficiency
while decode pool uses TP=4 for memory bandwidth. This is impossible with
co-located serving.

### Splitwise (ISCA 2024, Microsoft)

**Key insight**: use phase-appropriate hardware. Prefill benefits from
compute-dense GPUs; decode benefits from memory-bandwidth-rich machines.

**Results** (Llama-2-70B and BLOOM-176B):
- **1.4x higher throughput at 20% lower cost**
- Or: **2.35x more throughput** with the same cost/power budget
- Efficient request state transfer via optimized network libraries on fast
  back-plane interconnects

**Relationship to DistServe**: DistServe focuses on optimal placement and
parallelism strategy per phase. Splitwise focuses on cost efficiency by matching
hardware to phase characteristics. Complementary approaches.

### Current Engine Support (2026)

**NVIDIA Dynamo** (announced GTC 2025, Dynamo 1.0 in production 2026):
- Disaggregated serving is the core design principle
- Smart Router tracks KV cache locations across GPU fleet using a radix tree
- Dynamo Planner: dynamically decides disaggregated vs aggregated based on load
- Distributed KV Cache Manager: offloads to CPU memory, local SSDs, or object
  storage (petabyte-scale)
- Performance: **30x throughput** on DeepSeek-R1 671B (GB200 NVL72), **>2x** on
  Llama 70B (Hopper)
- Compatible with PyTorch, SGLang, TensorRT-LLM, and vLLM backends

**vLLM** (experimental disaggregated prefill):
- Architecture: 2 vLLM instances (prefill + decode) connected by a KV transfer
  connector
- Connectors: `NixlConnector` (fully async), `P2pNcclConnector`, `MooncakeConnector`
- Located under `vllm/distributed/kv_transfer`
- Enables separate tuning of TTFT and ITL without mutual interference
- Controlled tail latency: more reliable than chunked prefill for P99 ITL
- Production users: Meta, LinkedIn, Mistral, HuggingFace
- Recent RFC: bidirectional KV transfer between P and D nodes to eliminate
  redundant prefill computations

**SGLang**:
- Prefill/decode disaggregation as a core feature
- Production results on DeepSeek-V3/R1:
  - 52.3K input TPS, 22.3K output TPS per node
  - Configuration: 3 prefill nodes (24 GPUs) + 9 decode nodes (72 GPUs) on 96 H100s
  - GB200 NVL72 follow-up: 3.8x prefill and 4.8x decode throughput gains

**llm-d** (Kubernetes-native):
- Policy-driven routing and autoscaling with failure isolation
- Supports wide expert parallelism for MoE models
- 2.2K tokens/s per H200 GPU on 32-way expert parallel

**Ray Serve LLM**:
- Integrates NIXL and LMCache connectors
- Independent autoscaling of each phase based on load characteristics

### Industry Adoption (2026)

"Almost every production-grade LLM serving framework -- NVIDIA Dynamo, llm-d,
Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake -- runs on disaggregation."

The approach went from research paper (Jan 2024) to industry standard in ~18
months. Driving factors:
1. Latency control > raw throughput for production SLOs
2. Composable architecture: specialized optimizations per phase
3. Cost efficiency: right-size hardware for each phase
4. MoE models (DeepSeek-V3/R1) make disaggregation even more attractive

### Emerging: Attention-FFN Disaggregation

Beyond prefill/decode, the next frontier: disaggregate attention layers (memory-bound)
from FFN layers (compute-bound) within the same forward pass. Viable for MoE models
where communication patterns align naturally. Dense models remain an open challenge.

### KV Cache Storage Layer

Two notable systems for decoupling KV cache from inference:

**LMCache** (University of Chicago): middleware layer that stores/retrieves KV cache
independently of the inference engine. Works with vLLM and Dynamo.

**MoonCake** (Kimi AI, FAST'25 best paper): pools underexploited storage as a
centralized KV cache abstraction. 525% improvement with KV cache disaggregation.

---

## Key Formulas Summary

| Formula | Meaning |
|---------|---------|
| KV bytes/token = 2 * layers * kv_heads * head_dim * dtype_bytes | KV cache memory per token |
| Block hash = SHA-256(parent_hash, token_ids, extra_keys) | vLLM prefix cache block identity |
| Required bandwidth = KV_size / (TTFT_target - prefill_time) | Minimum interconnect for disagg |

---

## Sources

- [vLLM Automatic Prefix Caching Docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) -- official APC documentation
- [vLLM KV Cache Management (DeepWiki)](https://deepwiki.com/vllm-project/vllm/3.4-kv-cache-management-and-prefix-caching) -- SHA-256 hashing, BlockPool, LRU eviction internals
- [SGLang RadixAttention Blog (LMSYS)](https://lmsys.org/blog/2024-01-17-sglang/) -- radix tree design, 5x throughput, LRU eviction, fork/merge
- [SGLang vs vLLM KV Cache (RunPod)](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) -- token-level vs block-level, DeepSeek-R1 benchmarks
- [KV-Cache Wins: Prefix Caching to llm-d (llm-d.ai)](https://llm-d.ai/blog/kvcache-wins-you-can-see) -- cluster-scale routing, 57x TTFT improvement, 365GB pool in 339KB metadata
- [Disaggregated Inference: 18 Months Later (Hao AI Lab)](https://haoailab.com/blogs/distserve-retro/) -- industry adoption, SGLang/vLLM/Dynamo production numbers
- [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670) -- 7.4x requests, 12.6x SLO, placement algorithm
- [Splitwise (ISCA 2024, Microsoft)](https://arxiv.org/abs/2311.18677) -- 1.4x throughput at 20% lower cost, phase-appropriate hardware
- [NVIDIA Dynamo Introduction](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) -- Smart Router, Planner, NIXL, 30x throughput on GB200
- [vLLM Disaggregated Prefill Docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) -- experimental P/D, connector types, architecture
- [Disaggregated Prefill-Decode Architecture (JarvisLabs)](https://docs.jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode) -- KV transfer sizes, interconnect bandwidth table, production results
- [vLLM + PyTorch Disaggregated Inference (PyTorch Blog)](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) -- Meta production deployment
