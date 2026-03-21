# LLM Serving: Final Chapters Research Notes

Research for Ch 17 (Parallelism Across GPUs), Ch 18 (LoRA & Multi-Tenancy),
Ch 19 (Metrics That Matter), Ch 20 (The Modern Serving Stack).

---

## TOPIC 1: Parallelism Across GPUs (Ch 17)

### Tensor Parallelism (TP)

Splits individual weight matrices across GPUs. Each GPU holds a horizontal or
vertical slice of every layer's parameters. Two fundamental patterns from
Megatron-LM:

**Column-parallel linear**: weight matrix W split column-wise into [W₁, W₂, ...Wₙ].
Each GPU computes Y_i = X @ W_i independently. Results are used directly as input
to the next layer (no communication needed before the row-parallel layer).

**Row-parallel linear**: weight matrix W split row-wise. Each GPU computes a partial
result, then an **all-reduce** (sum) synchronizes across GPUs. This happens after
every transformer block (two all-reduces per layer: one after attention, one after MLP).

Communication pattern per layer:
- Forward pass: 2 all-reduce operations
- Backward pass: 2 all-reduce operations
- All-reduce volume: O(batch_size * seq_len * hidden_dim) per operation

**Bandwidth requirement**: TP is latency-sensitive because every layer needs
synchronization. NVLink provides 600-900 GB/s bidirectional (A100/H100), vs PCIe
Gen4 at ~32 GB/s. TP across PCIe is 20-30x slower per synchronization step.

**Rule of thumb**: TP only works well within a single node on NVLink. Never use TP
across nodes unless you have InfiniBand with RDMA (and even then, pipeline
parallelism is usually better cross-node).

**Memory math for Llama-70B with TP**:
- FP16: 70B * 2 bytes = 140 GB. TP=4 → 35 GB/GPU (doesn't fit 24GB cards)
- TP=8 → 17.5 GB/GPU (fits A100-40GB with room for KV cache)
- Q4 quantized: 70B * 0.5 bytes = 35 GB. TP=2 → 17.5 GB/GPU (fits 24GB cards)
- Q4 TP=4 → 8.75 GB/GPU (fits consumer GPUs with plenty of KV cache room)

### Pipeline Parallelism (PP)

Splits model layers across GPUs sequentially. GPU 0 gets layers 0-N, GPU 1 gets
layers N+1-2N, etc. Requests flow through the pipeline.

**The bubble problem**: with naive PP, only one GPU is active at a time. The others
idle. Microbatching reduces this:

- Pipeline bubble fraction = (p - 1) / (p + m - 1)
  where p = pipeline stages, m = number of microbatches
- With p=4 stages, m=4 microbatches: bubble = 3/7 = 43% wasted
- With p=4 stages, m=32 microbatches: bubble = 3/35 = 8.6% wasted
- Need m >> p to minimize bubbles

**Communication**: only activation tensors pass between stages (much less than
all-reduce). A single point-to-point transfer per microbatch per stage boundary.
Volume = batch_size * seq_len * hidden_dim * bytes. Works fine over slower
interconnects (PCIe, InfiniBand across nodes).

**When to use PP**:
- Cross-node deployments where NVLink isn't available
- When model fits on one node with TP but you need more throughput
- Combine with TP: TP within node, PP across nodes (3D parallelism)

**Memory advantage**: each GPU only loads its assigned layers. For Llama-70B with
PP=4: each GPU holds ~17.5B params = 35 GB FP16. But KV cache must exist on every
stage (each stage needs the full KV cache for its layers), so memory savings are
real for weights, partial for KV.

### Data Parallelism (DP)

Duplicates the full model on each GPU. Each replica handles independent requests.
Zero cross-GPU communication during inference (unlike training, where gradients
must sync).

**When to use**: model fits on one GPU and you need more throughput. Simplest
scaling strategy. Each GPU is completely independent -- no coordination overhead,
no communication bottleneck.

**Scaling**: throughput scales linearly with GPU count. 4 GPUs = 4x throughput
(minus load balancing overhead). This is what most production deployments use when
the model fits on a single GPU (7B-13B models on 80GB GPUs, or quantized 70B on
80GB GPUs).

**Load balancing**: a router/gateway distributes requests across replicas. Can use
round-robin, least-connections, or KV-cache-aware routing (prefix cache affinity).

### Expert Parallelism (EP)

For Mixture-of-Experts models. Routes different experts to different GPUs. Only
activated experts consume compute per token.

**DeepSeek-V3 architecture**:
- 671B total parameters, but only 37B activated per token
- Uses DeepSeekMoE architecture with Multi-head Latent Attention (MLA)
- Auxiliary-loss-free load balancing for expert routing
- Trained on 14.8T tokens using 2.788M H800 GPU hours

**Mixtral 8x7B architecture**:
- 46.7B sparse (total) parameters across 8 experts
- 12.8B active parameters per token (2 experts selected per token)
- Must load all 46.7B params onto GPU(s), but only 12.8B compute per forward pass

**Communication pattern -- all-to-all**:
- Router selects top-k experts per token
- Tokens must be dispatched to the GPUs holding their selected experts
- After expert computation, results must be gathered back
- This is an all-to-all communication pattern (every GPU may need to send to every other)
- Far more irregular than all-reduce (traffic pattern depends on routing decisions)

**EP strategy**: assign expert subsets to different GPUs. With 8 experts and 4 GPUs,
each GPU holds 2 experts. Tokens routed to experts on a remote GPU must be
transferred. Load balance depends entirely on the router's expert selection
distribution.

**The load balancing problem**: if certain experts are "popular" (selected more
often), the GPUs hosting them become bottlenecks. DeepSeek-V3's auxiliary-loss-free
balancing addresses this at training time. At inference time, uneven routing
creates GPU utilization imbalance.

### Sequence Parallelism (SP)

Three approaches for long contexts:

**Megatron SP**: partitions along sequence dimension for non-tensor-parallel ops
(LayerNorm, Dropout). Pairs naturally with TP -- TP handles linear layers, SP
handles everything else. Replaces all-reduce with reduce-scatter + all-gather.

**DeepSpeed-Ulysses**: all-to-all communication where each GPU handles a subset of
attention heads across the full sequence.

**Ring Attention**: KV blocks passed between GPUs in a ring topology via peer-to-peer
transfers. Enables arbitrarily long sequences by distributing the sequence across
GPUs. Each GPU holds a chunk of the sequence and rotates KV blocks around the ring.

### Combined Strategies (3D Parallelism)

Production deployments combine strategies:

| Setup | Strategy |
|-------|----------|
| Model fits on 1 GPU | DP across GPUs (simplest) |
| Model needs 2-8 GPUs, same node | TP within node |
| Model needs 2-8 GPUs + throughput | TP within node + DP across nodes |
| Very large model, multi-node | TP within node + PP across nodes |
| MoE model, multi-node | EP + TP within node |

HuggingFace strategy table:
- Single node, fits on 1 GPU → DDP or ZeRO
- Single node, doesn't fit → PP, ZeRO, or TP
- Single node, largest layer doesn't fit → TP or ZeRO
- Multi-node, fast interconnect → ZeRO or 3D parallelism (TP+PP+DP)
- Multi-node, slow interconnect → ZeRO or PP-heavy 3D parallelism

---

## TOPIC 2: LoRA and Multi-Tenancy (Ch 18)

### LoRA Mechanics Recap

Low-Rank Adaptation: instead of fine-tuning all parameters, add low-rank delta
matrices. For a weight matrix W ∈ R^(d×k), LoRA adds:

    W' = W + B @ A    where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

- Rank r typically 8-64 (most common: 16 or 32)
- Adapter size = 2 * d * r * num_adapted_layers * bytes_per_element
- For Llama-7B with rank 16, adapting all linear layers:
  - ~10-20M parameters = 20-40 MB in FP16
  - That's 0.2-0.3% of the base model's 14 GB
- Base model stays frozen. Only adapter weights are trained.

### S-LoRA: Serving Thousands of Adapters

Paper: "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" (Sheng et al., 2023)

**Core insight**: store all adapters in main memory (CPU RAM), fetch only the
adapters needed by current batch into GPU memory on demand.

**Three key innovations**:

1. **Unified Paging**: a single memory pool manages two types of dynamic data:
   - Adapter weights (varying ranks across different adapters)
   - KV cache tensors (varying sequence lengths across requests)
   - Same paging mechanism handles both, reducing fragmentation
   - Inspired by PagedAttention but extended to adapter weight management

2. **Custom CUDA kernels for heterogeneous batching**: within a single batch,
   different requests may use different adapters (different ranks, different
   weight values). Custom kernels handle this heterogeneity efficiently --
   each request's forward pass applies its own adapter weights without
   serializing the computation.

3. **Adapter-aware tensor parallelism**: when using TP across GPUs, adapter
   weights must be split consistently with the base model's tensor parallel
   layout. S-LoRA implements this correctly so multi-GPU serving works with
   multi-adapter batches.

**Performance**:
- Up to 4x throughput improvement vs HuggingFace PEFT and baseline vLLM
- Serves thousands of adapters on a single GPU (or across multiple)
- Increases number of concurrently servable adapters by "several orders of magnitude"
- Negligible overhead when adapter is already in GPU memory

### Multi-LoRA in vLLM

vLLM has native LoRA support with key configuration:

- `--enable-lora`: activates LoRA serving
- `--max-loras N`: maximum number of adapters loaded in GPU memory simultaneously
- `--max-lora-rank R`: maximum adapter rank (bounds memory allocation)
- `--lora-modules name=path`: register named adapters at startup

**Batching**: different requests in the same continuous batch can use different
adapters. The base model forward pass is shared; only the adapter delta computation
differs per request. This is the key efficiency: base model matmuls are not
duplicated.

**Memory**: each loaded adapter consumes (2 * rank * hidden_dim * num_layers * 2)
bytes for FP16. With max_loras=4 and rank 16 on a 7B model, adapter overhead is
roughly 80-160 MB total -- trivial compared to the base model's 14 GB.

**Hot-swapping**: when a request needs an adapter not currently in GPU memory, vLLM
evicts the least-recently-used adapter and loads the new one. This is fast because
adapters are small (tens of MB vs GB for models).

**Quantization compatibility**: LoRA works with quantized base models (AWQ, GPTQ).
The base model is quantized, adapters remain in FP16/BF16. Triton and torch
kernels handle the mixed-precision computation.

### Multi-LoRA Batching: How It Works

Within a single batch of B requests, some may use adapter-A, others adapter-B,
others no adapter at all. The computation flow:

1. **Shared base computation**: all B requests go through the base model's
   frozen weights simultaneously (standard batched matmul)
2. **Per-adapter delta**: for each adapted layer, compute B_adapter @ A_adapter
   for each request, using that request's specific adapter weights
3. **Addition**: base output + adapter delta = final output per request

The per-adapter delta step is where S-LoRA's custom CUDA kernels shine -- they
batch the heterogeneous adapter computations efficiently instead of looping over
adapters serially.

### KV Cache Implications

All adapters share the same KV cache layout because they share the same base model
architecture. This means:
- Prefix caching works across adapters: if two requests (using different adapters)
  share the same prompt prefix, the prefix KV cache can be reused
- The cache key includes the adapter ID, so cached KV blocks are correctly
  associated with their adapter
- Memory planning is simpler: KV cache size depends only on sequence length and
  model architecture, not on which adapter is active

### Multi-Tenancy Architecture

Production multi-tenant LoRA serving:

```
User-A (adapter: customer-support-v3) ─┐
User-B (adapter: code-review-v2)       ─┼─→ [Gateway] → [Scheduler] → [Engine]
User-C (no adapter, base model)        ─┘                                │
                                                                    [GPU Memory]
                                                                    ┌───────────┐
                                                                    │ Base Model │ (14 GB)
                                                                    │ Adapter A  │ (40 MB)
                                                                    │ Adapter B  │ (40 MB)
                                                                    │ KV Cache   │ (shared pool)
                                                                    └───────────┘
```

Hundreds or thousands of adapters in CPU RAM, a handful hot in GPU memory at any
time. The gateway routes requests by adapter name. The scheduler batches requests
across adapters. The engine applies per-request adapter weights during forward pass.

---

## TOPIC 3: Metrics That Matter (Ch 19)

### Latency Metrics

**TTFT (Time to First Token)**:
- Time from request arrival to first generated token
- Dominated by: queue wait + prefill computation
- Prefill is compute-bound: proportional to prompt length
- Users perceive this as "how fast the response starts"
- Good: <500ms. Acceptable: <2s. Bad: >5s.
- Under load, TTFT degrades first because of queuing

**ITL / TBT (Inter-Token Latency / Time Between Tokens)**:
- Time between consecutive generated tokens
- Dominated by: decode step time (memory-bandwidth-bound)
- For one user: ~10-30ms per token on modern hardware
- Under load: shared GPU means each decode step takes longer
- Users perceive this as "streaming speed" -- how fast text appears
- Good: <50ms (20+ tok/s streaming). Bad: >200ms (choppy, noticeable pauses)

**TPOT (Time Per Output Token)**:
- Average of ITL across all output tokens
- vLLM v0.6.0 benchmarks: 2x improvement in TPOT on Llama 70B vs prior versions
- Measured differently than ITL: TPOT = total_generation_time / num_output_tokens

**End-to-End Latency**:
- TTFT + (num_output_tokens * average_ITL)
- What the user actually experiences from request to completion
- Heavily dependent on output length (which is unpredictable)

### Throughput Metrics

**Token throughput (tokens/second)**:
- Total tokens generated per second across ALL concurrent requests
- The primary capacity metric for a serving system
- vLLM on Llama-8B: up to ~2,300-4,000 tok/s depending on backend and concurrency
- LMDeploy on Llama-8B: up to ~4,000 tok/s at 100 concurrent users (A100)
- Scales with batch size until GPU saturates

**Request throughput (requests/second)**:
- Complete requests served per second
- Depends on average output length: short outputs = more req/s
- More meaningful for business metrics than raw token throughput

**Goodput**:
- Throughput of requests that actually meet SLO targets
- A system doing 1000 tok/s but with 40% of requests exceeding latency SLO
  has lower goodput than one doing 800 tok/s with 5% SLO violations

### Percentile Latencies

Why P50/P90/P99 matter more than averages:

- P50: the median experience. Half of users see this or better.
- P90: the "unlucky" experience. 1 in 10 users hit this.
- P99: the tail. 1 in 100 users. Often 3-8x worse than P50.
- SLO contracts specify P99: "99% of requests < X seconds"

**What drives tail latency in LLM serving**:
- Long prefills blocking decode batches (without chunked prefill)
- KV cache preemption and recomputation
- Adapter loading/swapping
- GC pauses in Python runtime
- Cross-GPU communication stalls (TP/PP)
- Queue depth spikes from bursty traffic

From prior research (Sarathi-Serve):
- P50 TTFT: 0.76s, but P99 TBT: up to 1.76s
- Preemption accounts for 70% of P99 request latency

### GPU Utilization vs Effective Utilization

**GPU utilization** (nvidia-smi): percentage of time at least one GPU kernel is
running. Can be 99% while serving terribly.

**Why high GPU% doesn't mean good serving**:
- A long prefill pegging the GPU at 100% while 50 decode requests starve
- Memory-bandwidth-bound decode saturating DRAM bandwidth but barely using
  compute units
- Kernel launch overhead eating cycles between tiny operations
- "Token generation rate is strongly correlated with GPU utilization" (BentoML
  benchmark finding), but the reverse isn't true -- high utilization doesn't
  guarantee high token rate

**Effective utilization**: tokens produced per GPU-second, normalized by model size.
This captures whether the GPU is doing *useful work* toward completing requests.

**MFU (Model FLOPs Utilization)**: actual FLOPS achieved / theoretical peak FLOPS.
More informative than GPU utilization. Typical inference MFU:
- Prefill: 40-60% MFU (compute-bound, reasonably efficient)
- Decode: 1-5% MFU (memory-bandwidth-bound, GPU compute massively underutilized)
- This is why batching matters: larger decode batches increase MFU by amortizing
  weight reads across more tokens

### Cache Metrics

**Prefix cache hit rate**: fraction of prompt tokens that match cached KV blocks.
- High hit rate = skip prefill for those tokens = lower TTFT
- SGLang achieves 2-5x speedup on TTFT with high cache hit rates
- Meaningful for: chatbots (system prompt reuse), RAG (shared document chunks),
  code completion (file context reuse)

**Preemption rate**: how often running requests get evicted to free KV cache memory.
- Each preemption = wasted compute (must recompute KV cache later)
- High preemption rate signals: memory too small, batch too large, or
  scheduling policy too aggressive
- Target: <1% of requests preempted

### Economic Metrics

**$/million tokens (cost per million tokens)**:
- The metric that determines business viability
- Factors: GPU cost ($/GPU-hour), throughput (tokens/s), utilization
- Example: A100 at $2/hr generating 2000 tok/s = $0.28/million tokens
  (2000 tok/s * 3600s = 7.2M tokens/hr, $2/7.2M = $0.28/M)
- H100 at $3/hr generating 5000 tok/s = $0.17/million tokens
- Quantization (Q4) can 2x throughput → halves $/M tokens
- Batching efficiency is the biggest lever on cost

**Tokens per dollar per second (normalized throughput)**:
- Accounts for different GPU price points
- Enables apples-to-apples comparison across hardware generations

### Common Benchmark Traps

1. **Local tok/s is not a service metric**: single-user tok/s (llama.cpp on
   desktop) measures peak single-stream speed. Under concurrent load,
   per-user tok/s drops dramatically. Never quote single-user numbers for
   production capacity.

2. **TTFT and ITL are separate concerns**: a system can have great TTFT (fast
   prefill) but terrible ITL (slow decode under load), or vice versa. Always
   report both.

3. **Concurrency changes everything**: benchmark at 1 user, 10 users, 50, 100.
   Every metric looks different at each level. A system that's "fastest" at
   concurrency 1 may be slowest at 100.

4. **Average latency hides sins**: always report P50, P90, P99. A system with
   P50=100ms but P99=10s is terrible despite a "good average."

5. **Input/output length distribution matters**: benchmarking with uniform 100-token
   prompts and 100-token outputs is unrealistic. Use ShareGPT or similar
   real-world distributions (heavy-tailed output lengths).

6. **Don't tune per-backend**: BentoML benchmark insight -- tuning each backend's
   config (KV cache size, max sequences, block size) is "not scalable." Use
   default configs for fair comparison.

7. **Throughput at what latency?**: reporting throughput without latency constraints
   is meaningless. "10,000 tok/s" means nothing if P99 latency is 30 seconds.
   Always report throughput at a specific latency target.

### vLLM v0.6.0 Benchmark Methodology (Reference)

Tested against TensorRT-LLM r24.07, SGLang v0.3.0, lmdeploy v0.6.0a0:
- **ShareGPT dataset**: 500 prompts, avg 202 input / 179 output tokens
- **Prefill-heavy synthetic**: ~462 input, 16 output tokens
- **Decode-heavy synthetic**: ~462 input, 256 output tokens
- Hardware: A100 and H100 GPUs
- Models: Llama 3 8B and 70B
- Results: vLLM v0.6.0 achieved 2.7x throughput + 5x TPOT improvement on 8B,
  1.8x throughput + 2x TPOT on 70B

---

## TOPIC 4: The Modern Serving Stack (Ch 20)

### Full Stack Architecture

```
Client Request
    │
    ▼
┌─────────────────────┐
│   Gateway / Router   │  ← Load balancing, auth, rate limiting
│   (prefix-aware)     │  ← Route to replica with best cache hit
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Scheduler          │  ← Continuous batching, chunked prefill
│                      │  ← Priority queues, SLO-aware admission
│                      │  ← Preemption policy (FCFS, priority, SJF)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Inference Engine   │  ← PagedAttention, FlashAttention
│   (vLLM/SGLang/     │  ← TP/PP across GPUs
│    TRT-LLM)         │  ← Quantized kernels (W4A16, W8A8)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   GPU(s)             │  ← Actual matrix multiplications
│   + KV Cache         │  ← PagedAttention block tables
│   + Adapter Weights  │  ← Hot LoRA adapters
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Metrics / Autoscale│  ← TTFT, ITL, throughput, cache hit rate
│                      │  ← Scale replicas based on queue depth
└─────────────────────┘
```

### NVIDIA Dynamo: The Orchestration Layer

Dynamo sits above inference engines as an orchestration framework. Not a replacement
for vLLM/SGLang/TRT-LLM -- it coordinates them.

**Key components**:

1. **Frontend & KV-Aware Router**: routes requests based on worker load AND cache
   overlap. If worker-3 already has the prefix cached, route there -- eliminates
   redundant prefill computation.

2. **Disaggregated Serving**: separates prefill and decode into independent GPU pools.
   Prefill pool: optimized for compute (high FLOPS utilization). Decode pool:
   optimized for memory bandwidth (high batch sizes, low per-request compute).
   Each pool can be sized independently based on workload mix.

3. **Planner**: SLA-driven autoscaler. Profiles workloads, right-sizes prefill and
   decode pools, meets latency targets at minimum cost. This is the "brain" that
   decides how many prefill GPUs vs decode GPUs to allocate.

4. **KV Block Manager (KVBM)**: offloads KV cache across a hierarchy:
   GPU VRAM → CPU RAM → SSD → remote storage. Extends effective context length
   beyond GPU memory. Enables longer conversations without preemption.

5. **ModelExpress**: streams model weights GPU-to-GPU via NIXL/NVLink. Enables 7x
   faster cold-starts for new replicas vs loading from disk. Critical for
   autoscaling responsiveness.

6. **Grove**: Kubernetes operator for topology-aware gang scheduling. Places
   TP groups on GPUs that share NVLink, PP stages on GPUs with fast interconnects.
   Aware of rack, host, and NUMA topology.

**Supported backends**: vLLM, SGLang, TensorRT-LLM. All three support disaggregated
serving and KV-aware routing through Dynamo's coordination.

### When to Use Each Architecture

**Local-first (llama.cpp on consumer hardware)**:
- Single GPU, single user, maximum simplicity
- Quantized models (Q4_K_M, Q5_K_M) on consumer GPUs
- Best for: developers, hobbyists, privacy-sensitive use cases
- No batching, no scheduling, no multi-user support needed
- Throughput: 30-100+ tok/s single-stream depending on model/GPU

**Single-node production (vLLM or SGLang)**:
- One multi-GPU node (2-8 GPUs with NVLink)
- TP across GPUs for models that don't fit on one
- Continuous batching, PagedAttention, prefix caching
- Best for: startups, medium-traffic APIs, internal tools
- Handles 10-100+ concurrent users per node
- Example: Llama-70B on 4x A100-80GB with TP=4

**Multi-node scale-out (Dynamo orchestrating backends)**:
- Multiple nodes, disaggregated prefill/decode
- KV-aware routing across nodes
- Autoscaling based on SLOs
- Best for: high-traffic production, variable load, strict SLOs
- Example: 8 prefill GPUs + 32 decode GPUs, independently scaled

**Adapter-heavy enterprise (multi-LoRA with SGLang/vLLM)**:
- Hundreds of customer-specific adapters
- Shared base model, hot-swapped adapters
- S-LoRA-style unified paging
- Best for: SaaS platforms offering fine-tuned models per customer
- Example: one base Llama-70B, 500 customer adapters in CPU RAM

**Low-latency assistant (disaggregated P/D with prefix caching)**:
- Chatbot/assistant with system prompts that cache well
- Prefix caching for system prompt + conversation history
- Disaggregated serving to prevent prefill-decode interference
- Speculative decoding for interactive speed
- Best for: consumer chat products, coding assistants
- Target: TTFT <300ms, ITL <30ms

**High-throughput batch (maximize tokens/dollar)**:
- Offline processing: summarization, classification, extraction
- Maximize batch size, sacrifice latency
- No streaming needed, can buffer large queues
- Best for: document processing pipelines, eval suites
- Example: 256-512 batch size, P99 latency irrelevant, $/M tokens is king

### The Evolution Arc

The full progression through the trilogy:

| Era | Innovation | Chapter | Key Metric Unlocked |
|-----|-----------|---------|-------------------|
| 1 - Just Run | Basic transformer inference | Ch 1-3 | "It works" |
| 2 - Fit | Quantization (Q4/Q8), reduced precision | Ch 4-6 | Model fits on GPU |
| 3 - IO | FlashAttention, fused kernels, memory bandwidth | Ch 7-9 | Single-user tok/s |
| 4 - KV | PagedAttention, KV cache management | Ch 10-12 | Memory efficiency |
| 5 - Batch | Continuous batching, scheduling | Ch 13-14 | Multi-user throughput |
| 6 - Split | Disaggregated P/D, prefix caching | Ch 15-16 | Tail latency (P99) |
| 7 - Parallel | TP, PP, EP across GPUs | Ch 17 | Scale beyond 1 GPU |
| 8 - Adapt | LoRA multi-tenancy | Ch 18 | $/customer |
| 9 - Measure | Metrics, benchmarking | Ch 19 | SLO compliance |
| 10 - Orchestrate | Full stack (Dynamo, routing, autoscale) | Ch 20 | Production readiness |

Each era builds on the previous. You can't orchestrate well (era 10) without
understanding what to measure (era 9). You can't measure meaningfully without
multi-user workloads (era 5). The stack is deeply layered, and every optimization
in earlier chapters compounds into the final serving architecture.

### The Serving Stack Landscape (2026)

**Inference engines** (the GPU workers):
- vLLM: PagedAttention, broad model support, LoRA, speculative decoding
- SGLang: RadixAttention (prefix tree), constrained decoding, multi-LoRA
- TensorRT-LLM: NVIDIA-optimized kernels, FP8, best raw performance on NVIDIA HW
- llama.cpp: CPU/consumer GPU, GGUF quantization, widest hardware support
- LMDeploy: strong throughput on quantized models (AWQ), competitive with vLLM

**Orchestration** (the coordination layer):
- NVIDIA Dynamo: disaggregated serving, KV-aware routing, autoscaling
- Ray Serve: general-purpose scaling, used by Anyscale
- Kubernetes + custom operators: DIY orchestration

**Benchmarking results** (BentoML, Llama 3 8B on A100, 100 concurrent):
- LMDeploy: ~4,000 tok/s, lowest TTFT
- vLLM: ~2,300-2,500 tok/s, most consistent TTFT under load
- TRT-LLM: matched LMDeploy on 70B, but TTFT degraded to >6s at 100 users
- TGI: ~2,300+ tok/s, good documentation/ease of use

---

## Sources

- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) -- unified paging, custom CUDA kernels, 4x throughput
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) -- SOSP 2023, 2-4x throughput, <4% memory waste
- [vLLM v0.6.0 Performance Update](https://vllm.ai/blog/perf-update) -- 2.7x throughput on 8B, benchmark methodology
- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo) -- orchestration, disaggregated serving, KV-aware routing, ModelExpress
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) -- 671B/37B MoE, MLA, 14.8T tokens
- [Colossal-AI Parallelism Concepts](https://colossalai.org/docs/concepts/paradigms_of_parallelism/) -- TP/PP/DP/SP/ZeRO overview
- [HuggingFace Multi-GPU Training Guide](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many) -- parallelism strategy table, 3D parallelism
- [BentoML Inference Backend Benchmark](https://www.bentoml.com/blog/benchmarking-llm-inference-backends) -- LMDeploy vs vLLM vs TRT-LLM vs TGI
- [Anyscale Continuous Batching Blog](https://www.anyscale.com/blog/continuous-batching-llm-inference) -- 8-23x throughput, iteration-level scheduling
- [MoE Visual Guide (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) -- Mixtral 46.7B/12.8B, routing mechanisms
