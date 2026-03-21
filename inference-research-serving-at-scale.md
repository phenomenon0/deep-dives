# LLM Serving at Scale: Research Notes

Research for Ch 13 (One User Becomes Many) and Ch 14 (The Scheduler).

---

## TOPIC 1: One User Becomes Many

### The Queuing Problem

Real LLM traffic is nothing like a single-user demo. Requests arrive as a Poisson
process (memoryless, random arrivals), but the work each request requires varies wildly:

| User | Prompt tokens | Output tokens | KV cache held | Wall time |
|------|--------------|---------------|---------------|-----------|
| A    | 20           | 20            | ~40 tokens    | ~200ms    |
| B    | 4,096        | 2             | ~4,098 tokens | ~800ms (prefill-dominated) |
| C    | 200          | 800           | ~1,000 tokens | ~8s (decode-dominated) |

Prompt lengths follow a **heavy-tailed distribution** -- most are short, but a
meaningful fraction are very long. Output lengths are even worse: completely
unpredictable until the model emits EOS. Research models this with a lognormal
distribution (log mean ~7, std dev ~0.7) for output token counts.

Key insight from queuing theory paper (Xin et al., 2024): the heavy tail of output
token lengths in a small fraction of requests **significantly extends the average
queuing delay** for everyone. A slight reduction in max output tokens for a very
small percentage of requests can reduce mean queuing delay by ~59%.

### Little's Law Applied to LLM Serving

**L = lambda * W**

- L = average number of requests in the system (concurrency)
- lambda = arrival rate (requests/second)
- W = average time a request spends in the system (end-to-end latency)

Concrete example:
- lambda = 50 req/s, W = 2 seconds --> L = 100 concurrent requests
- Each concurrent request holds KV cache memory
- KV cache per token per layer = 2 * num_kv_heads * head_dim * bytes_per_element
- For Llama-70B (80 layers, 8 KV heads GQA, head_dim 128, FP16):
  - Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = ~320 KB
  - 100 concurrent requests, avg 2K tokens each = 100 * 2048 * 320KB = ~62 GB of KV cache alone
- This is why gpu_memory_utilization (default 0.9 in vLLM) allocates 90% of VRAM to KV cache + model

**The feedback loop**: under load, queuing delay increases W, which increases L
(more concurrent requests), which increases memory pressure, which may trigger
preemptions, which increases W further. Systems can tip into a death spiral.

### Tail Latency

The P99 is often 3-8x worse than P50. Sarathi-Serve measurements:
- P50 TTFT: 0.76s, but prefill-decode interference causes generation stalls lasting
  **multiple seconds** for tail requests
- vLLM P99 TBT (time between tokens): up to **1.76 seconds** on internal workloads
- Preemption loss accounts for **70%** of P99 request latency

Why tail latency dominates user experience:
- Users notice the *worst* response, not the average
- SLO contracts specify P99: "99% of requests complete in < X seconds"
- One long prefill blocks decode for *everyone* in the batch (without chunked prefill)
- A 4K-token prefill can stall 50 ongoing decodes for hundreds of milliseconds

### Prefill-Decode Interference: The Core Tension

Prefill is **compute-bound** (matrix multiplications over full prompt). Decode is
**memory-bandwidth-bound** (reading model weights for one token at a time, per request).

When a long prefill joins a batch of decodes:
- The GPU spends its compute budget on the prefill
- All ongoing decodes stall until the prefill completes
- This is a "generation stall" -- users see tokens stop streaming, then resume

Without mitigation, this makes the system fundamentally unpredictable. The solution
is chunked prefill (see Scheduler section below).

### Throughput vs Latency: The Fundamental Tradeoff

| Batch size | Throughput     | Per-request latency | GPU utilization |
|-----------|----------------|---------------------|-----------------|
| 1         | Low            | Lowest              | Low (idle between ops) |
| 8         | Good           | Moderate            | Moderate        |
| 64        | High           | Higher              | High            |
| 256       | Maximum        | Highest             | Saturated       |

- Batch size 1 = demo mode. Fast for that one user, GPU mostly idle.
- Batch size 256 = production mode. Maximum throughput, but every request shares
  the GPU with 255 others.

Key finding: **after a certain batch size, you cross from memory-bound to
compute-bound**, and further increases just add latency without improving throughput.
But recent research shows large-batch LLM inference often stays memory-bound
(DRAM bandwidth saturated), meaning the crossover point is higher than expected.

Continuous batching achieves **23x throughput improvement** over static batching
(vLLM/Anyscale benchmarks on OPT-13B). This is the single biggest architectural
win for multi-user serving.

### The M/G/1 Queue Model for LLM Serving

The standard model: M/G/1 queue (Poisson arrivals, General service time, 1 server).

- Service time S = a*n + c (linear in output token count n, plus constant overhead)
- Mean queuing delay: **E[W] = lambda * E[S^2] / 2(1 - rho)** where rho = lambda/mu
- The E[S^2] term means variance in service time *directly* increases delay
- Heavy-tailed output distributions inflate E[S^2] dramatically
- This is why a few long-output requests hurt *everyone's* latency

Practical implication: setting max_tokens limits isn't just about controlling cost --
it's a queuing theory necessity for maintaining reasonable latency.

---

## TOPIC 2: The Scheduler

### Overview: What the Scheduler Does Each Step

The vLLM V1 scheduler runs a two-phase loop every iteration:

```
Phase 1: RUNNING REQUESTS
  For each request in running list:
    Try to allocate KV cache blocks for new tokens
    If allocation fails (OOM):
      Preempt this request (free KV, move to waiting queue)

Phase 2: WAITING REQUESTS
  While token_budget > 0 AND running_count < max_num_running_reqs:
    Take next request from waiting queue
    Allocate KV cache blocks
    Add to running batch
    Subtract tokens from budget

Execute forward pass on the assembled batch
```

### Admission Control

Before accepting a new request, the scheduler checks:
1. Is there enough KV cache memory to hold at least the prompt?
2. Will accepting it breach max_num_seqs (default 128)?
3. Will it exceed max_num_batched_tokens budget (default 2048)?

If any check fails, the request stays in the waiting queue.

The token budget (max_num_batched_tokens) is the master throttle. It limits the
total tokens processed in one forward pass. This is critical because:
- Too high → long prefills dominate, decode latency spikes
- Too low → GPU underutilized, throughput suffers
- vLLM recommends > 8192 for small models on large GPUs

### Scheduling Policies

**FCFS (First Come First Served)** -- vLLM default (`--scheduling-policy fcfs`)
- Simple, fair, predictable
- Problem: head-of-line blocking. A 32K-token prefill blocks everything behind it.
- Preemption order: LIFO (most recently arrived gets preempted first -- "fair" in
  the sense that the request that has done the least work loses the least)

**Priority-based** (`--scheduling-policy priority`)
- Requests assigned numeric priorities
- Higher priority values are preempted first (lower number = higher priority)
- Same-priority tie-breaking: FIFO
- Use cases: premium users, time-sensitive requests, retry prioritization

**SJF (Shortest Job First)** -- not natively in vLLM, but studied in research
- Optimal for minimizing average latency (provably)
- Unfair to long requests (starvation risk)
- Hard to implement because output length is unknown until EOS

**Decode-first** -- vLLM V1 with chunked prefill enabled
- Always schedule ALL pending decodes before any prefills
- Remaining token budget goes to prefill chunks
- Rationale: decodes are cheap (1 token each) and latency-sensitive (user is
  waiting for streaming tokens). Prefills are expensive but can be chunked.

### Chunked Prefill: The Key Innovation

Without chunked prefill: a 4K-token prompt runs as one giant prefill, blocking all
decodes for that iteration.

With chunked prefill: the 4K prompt is split into chunks (e.g., 512 tokens each)
processed across 8 iterations, interleaved with ongoing decodes.

Tuning the chunk size (via max_num_batched_tokens):
- **Smaller chunks (e.g., 512-2048)**: better ITL (inter-token latency), worse TTFT
  (time to first token) because prefills take more iterations
- **Larger chunks (e.g., 8192+)**: better TTFT, worse ITL because more prefill
  work per iteration
- If max_num_batched_tokens == max_model_len: almost equivalent to V0 behavior
  (but still decode-first)

Overhead of chunking: even with the smallest chunk size of 512, overhead is at most
~25% compared to non-chunked execution (Sarathi-Serve measurements).

### Preemption: When KV Cache Fills Up

Trigger: a running request needs a new KV cache block but none are available.

**RECOMPUTE** (V1 default):
- Free all KV cache blocks for the preempted request
- Move request back to waiting queue (status: PREEMPTED)
- When rescheduled: recompute the entire KV cache from scratch
- Why this is fast: FlashAttention-3 can recompute 4K tokens faster than
  transferring ~10 GB of KV cache over PCIe Gen5
- Simpler implementation, lower variance

**SWAP** (legacy, required for beam search):
- Copy KV cache blocks from GPU VRAM to CPU RAM via PCIe
- When rescheduled: copy back from CPU to GPU
- Higher overhead (PCIe bandwidth bottleneck, memory pinning)
- Required when multiple sequences share KV history (beam search)
  because recomputation of shared state is algorithmically complex

Which request to preempt?
- FCFS policy: most recently arrived (LIFO preemption) -- fairness to older requests
- Priority policy: lowest priority first (highest numeric value)
- Preempted requests are prepended to waiting queue (rescheduled next)

### Key vLLM Parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `max_num_seqs` | 128 | Max concurrent sequences in a batch |
| `max_num_batched_tokens` | 2048 | Token budget per iteration |
| `gpu_memory_utilization` | 0.9 | Fraction of VRAM for KV cache (after model weights) |
| `enable_chunked_prefill` | True (V1) | Split large prefills into chunks |
| `scheduling_policy` | "fcfs" | FCFS or priority-based scheduling |
| `preemption_mode` | "recompute" (V1) | Recompute or swap on preemption |

### Request Lifecycle

```
WAITING ──(admitted, KV allocated)──> RUNNING ──(EOS)──> FINISHED
   ^                                     |
   |                                     |
   └──(preempted, KV freed)──────────────┘
```

Special states in V1:
- WAITING_FOR_FSM: waiting for structured output FSM initialization
- WAITING_FOR_REMOTE_KVS: waiting for KV transfer in disaggregated prefill/decode

### The Throughput-Latency Operating Point

The scheduler's job is to find the sweet spot:

```
                Throughput
                    ^
                    |          ___________
                    |         /
                    |        /
                    |       /    <-- optimal operating point
                    |      /         (SLO-constrained)
                    |     /
                    |    /
                    |   /
                    +---+-------------------> Latency (P99)
                        |
                    SLO boundary
```

- Small batch → low latency, low throughput, GPU underutilized
- Large batch → high throughput, high latency, good GPU utilization
- The optimal point is the **maximum throughput that still meets the P99 SLO**

Sarathi-Serve achieves 2.6x-6.3x higher serving capacity than vLLM under strict SLOs
by eliminating generation stalls through chunked prefill + decode-first scheduling.

### Disaggregated Prefill/Decode (Emerging Architecture)

Separate GPU pools for prefill and decode:
- Prefill instances: compute-bound, benefit from high batch sizes
- Decode instances: memory-bandwidth-bound, benefit from many concurrent requests
- KV cache transferred between pools after prefill completes

Tradeoffs:
- Eliminates prefill-decode interference entirely
- Adds network transfer overhead for KV cache
- Complexity: routing, load balancing, KV transfer protocols
- Active research area: PPD, DuetServe, TaiChi, ARES, Llumnix

---

## Key Formulas Summary

| Formula | Meaning |
|---------|---------|
| L = lambda * W | Little's Law: concurrency = arrival rate * latency |
| E[W] = lambda*E[S^2] / 2(1-rho) | M/G/1 mean queuing delay |
| KV bytes/token = 2 * layers * kv_heads * head_dim * dtype_bytes | KV cache memory per token |
| Total KV = batch * seq_len * KV_bytes/token | Total KV cache memory |

---

## Sources

- [Queueing Theoretic Perspective on Low-Latency LLM Inference (Xin et al.)](https://arxiv.org/html/2407.05347) -- M/G/1 model, heavy-tailed distributions, 59% delay reduction
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff (OSDI'24)](https://arxiv.org/html/2403.02310) -- chunked prefill, stall-free batching, 2.6-6.3x capacity gains
- [vLLM Scheduler and Resource Allocation (DeepWiki)](https://deepwiki.com/vllm-project/vllm/3.3-multimodal-models) -- two-phase loop, LIFO preemption, token budget
- [vLLM Request Scheduling (DeepWiki)](https://deepwiki.com/vllm-project/vllm/2.5-request-scheduling) -- SchedulerConfig defaults, policy options
- [vLLM Optimization and Tuning Docs](https://docs.vllm.ai/en/stable/configuration/optimization/) -- parameter defaults, chunked prefill tuning
- [vLLM Scheduler API Reference](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/) -- V1 scheduler classes
- [vLLM Preemption Forum Discussion](https://discuss.vllm.ai/t/request-preemption-option/1672) -- recompute vs swap tradeoffs
- [KV Cache Memory Calculation (Brenndoerfer)](https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu) -- per-token formulas
- [Continuous Batching: 23x Throughput (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference) -- static vs continuous batching benchmarks
- [Little's Law and Concurrency (Medium)](https://medium.com/@rajesh.sgr/littles-law-and-concurrency-why-your-system-gets-slow-when-it-s-busy-a0fbee7f303b) -- L = lambda * W applied to systems
- [DuetServe: Harmonizing Prefill and Decode](https://arxiv.org/pdf/2511.04791) -- disaggregated P/D architecture
- [POD-Attention: Prefill-Decode Overlap (Microsoft)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf) -- overlap strategies
- [NVIDIA Inference Optimization Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) -- prefill compute-bound, decode memory-bound
- [vLLM Scheduler Source (GitHub)](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
