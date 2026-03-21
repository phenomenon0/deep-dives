# Inference Engine Landscape -- March 2026

Research notes for the "every engine is a worldview" payoff chapter.

---

## 1. llama.cpp -- The Portability Engine

**Philosophy:** One binary, zero dependencies, runs anywhere. The GCC of inference.

**Backends (as of 2026):**
- CUDA, Metal, Vulkan (1.2+), OpenCL, SYCL, MUSA, HIP (AMD), CANN (Ascend), RPC, CPU (x86 AVX/AVX-512, ARM NEON, zDNN, ZenDNN, BLAS, BLIS)
- Full Android/ChromeOS acceleration added December 2025 via new GUI binding -- native app development, not just adb cross-compile
- RPC backend enables distributed inference: one controller orchestrates remote workers over the network, layers placed on any device (local or remote) proportional to available memory. In 2026, a 4-node AMD Ryzen AI Max+ cluster ran Kimi K2.5 (1T params) via RPC.

**GGUF format:** Version 3. Self-contained (weights + metadata + tokenizer). Quantization baked into the file. Docker Hub integration added -- GGUF models stored/pulled as OCI artifacts.

**Key recent features (2025-2026):**
- Flash Attention on CUDA and Metal backends
- CUDA graphs for reduced kernel launch overhead (NVIDIA published optimization guide)
- Speculative decoding via `--model-draft` + `--draft N` (draft model generates candidates, main model verifies). Profile-guided speculative decoding in development.
- libmtmd (April 2025): revived multimodal model support that had been stagnant
- Docker Hub native GGUF pulls

**Strength:** Runs anywhere. Literally one binary. Consumer GPUs, Apple Silicon, Raspberry Pi, phones, Jetson, AMD, Intel, distributed clusters over RPC. No Python, no CUDA toolkit, no framework dependency.

**Weakness:** Single-user focused. The server mode (`llama-server`) handles multiple requests but lacks the sophisticated scheduling of vLLM/SGLang (no PagedAttention, no continuous batching at the vLLM level, no prefix caching trees). Limited multi-request scheduling is the fundamental gap.

**Who uses it:** Local inference, edge deployment, hobbyists, embedded systems, Apple Silicon users, anyone who wants "just run the model." Ollama wraps it. GPUStack wraps it (llama-box). It's the universal backend for local AI.

---

## 2. ExLlamaV2/V3 -- The Consumer NVIDIA Specialist

**Philosophy:** Maximum performance on exactly the hardware you have. One GPU, one user, ruthlessly optimized.

**Critical update: ExLlamaV3 exists (2025-2026).**
- turboderp-org/exllamav3 on GitHub, latest release v0.0.25 (March 2026)
- Supports 30+ model architectures including multimodal (vision-language)
- New EXL3 quantization format: streamlined QTIP variant (Cornell RelaxML). Near-lossless at 4 bpw, coherent down to 1.6 bpw. Simple conversion: just input model + target bitrate.
- Day-0 support for Qwen3.5 and Qwen3.5 MoE (March 2026)

**ExLlamaV2 (still active, not deprecated):**
- EXL2 format: GPTQ-based, per-layer mixed quantization (2-8 bit), any average bitrate. More important weights get more bits -- sparse quantization effect.
- Dynamic generator: paged attention via Flash Attention 2.5.7+, continuous batching, smart prompt caching, KV cache deduplication, speculative decoding, all in one consolidated API.
- v0.3.2 on PyPI

**TabbyAPI:** The official server for both V2 and V3. OpenAI-compatible API, HF model downloading, embedding support, Jinja2 chat templates, parallel batching with paged attention (Ampere+).

**Strength:** Best single-GPU performance on consumer NVIDIA. EXL2/EXL3 quantization quality is best-in-class for mixed precision. TabbyAPI makes it a real server.

**Weakness:** NVIDIA-only (no ROCm in V3 yet as of early peek). Consumer-focused -- not designed for datacenter multi-node. Community is smaller than llama.cpp or vLLM.

**Who uses it:** Prosumers, text-generation-webui users, anyone running a 70B model on a single 3090/4090 and wanting maximum quality-per-bit.

---

## 3. vLLM -- The Production Default

**Philosophy:** The Kubernetes of inference. General-purpose, well-documented, extensible, the safe choice.

**V1 engine timeline (confirmed):**
- v0.8.0 (Feb 2025): V1 becomes default engine
- v0.9: Last version with V0 engine code (frozen, bugs-only)
- v0.10: V0 code removal begins
- v0.11.0: V0 fully removed. V1 is the only engine. AsyncLLMEngine, LLMEngine, MQLLMEngine, all V0 attention backends -- gone.

**Core tech:**
- PagedAttention (the innovation that started it all -- virtual memory for KV cache)
- Continuous batching with chunked prefill (default in V1)
- Automatic prefix caching (APC)
- Speculative decoding (EAGLE-3 support being tested, Q1 2026)
- Distributed: Tensor Parallelism, Pipeline Parallelism, Data Parallelism, Expert Parallelism (wide EP mature as of Q3 2025)
- Prefill/decode disaggregation (mature as of Q3 2025)
- OpenAI-compatible API

**Q1 2026 roadmap focus:**
- PyTorch compilation integration: custom compile/fusion passes, vLLM IR for kernel registration, compile caching
- EAGLE-3 support and testing
- Performance dashboard for priority models (DeepSeek-V3.2, K2, GPT-OSS, Qwen3-Next, Gemma3) on priority hardware (GB200, H200, MI355)

**Recent additions (2026):**
- NVIDIA Nemotron 3 Super support (March 2026)
- vLLM Semantic Router: v0.1 Iris (Jan 2026), v0.2 Athena -- routing for safety, semantic caching, memory, retrieval

**Strength:** General-purpose. Largest community. Best documentation. Widest model support. The "you can't get fired for choosing vLLM" option. Mature disaggregated serving.

**Weakness:** Not the fastest on any single benchmark. SGLang and LMDeploy both beat it by ~29% on raw throughput (H100, 2026 benchmarks). The compatibility layer that makes it general also limits depth of optimization.

**Performance (2026 benchmarks):**
- ~12,500 tok/s on H100 (vs SGLang/LMDeploy ~16,200 tok/s)
- The gap is not kernel speed -- it's scheduler/orchestration overhead. When vLLM uses the same FlashInfer kernels as SGLang, the bottleneck is internal plumbing.

**Who uses it:** Startups, mid-tier deployments, anyone who wants the default recommendation. Cloud providers. HuggingFace recommends it as TGI replacement.

---

## 4. SGLang -- The Serving-First Runtime

**Philosophy:** The workload IS the optimization target. Build the runtime around how LLMs are actually used (shared prefixes, structured output, multi-turn).

**Core innovation -- RadixAttention:**
A radix tree (prefix tree) for KV cache management. Every completed prefix is stored in a tree. New requests that share a prefix (system prompt, few-shot examples, chat history) get instant KV cache reuse. Not just "automatic prefix caching" -- it's a data structure designed for the access patterns of real LLM serving.

**Key features (2025-2026):**
- Continuous batching, multi-LoRA support
- Structured output (grammar-based constrained decoding)
- Prefill/decode disaggregation
- SGLang Diffusion (Jan 2026): up to 1.5x faster image/video generation
- Native TPU support via SGLang-Jax backend (Oct 2025)
- Day-0 model support: DeepSeek-V3.2 sparse attention (Sep 2025), MiMo-V2-Flash, Nemotron 3 Nano, Mistral Large 3, LLaDA 2.0, MiniMax M2 (Dec 2025)
- Model Gateway v0.2.4: optimized radix tree for cache-aware load balancing, tokenizer CPU/memory optimization
- a16z Open Source AI Grant recipient (June 2025) -- "trillions of tokens daily"

**Performance (2026 benchmarks):**
- ~16,200 tok/s on H100 (tied with LMDeploy, 29% ahead of vLLM)
- 10-20% speed boost over vLLM in multi-turn conversations (DeepSeek-R1, dual H100)
- The advantage grows with prefix sharing: agent workflows, chat with system prompts, few-shot examples

**Strength:** Prefix reuse (RadixAttention), structured generation, competitive throughput. Best for workloads with heavy prefix sharing.

**Weakness:** Younger ecosystem than vLLM. Thinner documentation, smaller community. Edge cases require more debugging. Stability vs peak performance tradeoff.

**Who uses it:** Applications with heavy prefix sharing, structured output needs, agent frameworks, anyone who benchmarked and found vLLM leaving performance on the table. HuggingFace recommends it alongside vLLM as TGI replacement.

---

## 5. TensorRT-LLM -- The NVIDIA Performance Stack

**Philosophy:** NVIDIA silicon, NVIDIA software, NVIDIA optimization. The vertical stack.

**Key features (2025-2026):**
- Optimized CUDA kernels, FP8 native, NVFP4 (new) for Blackwell/RTX 50-series
- In-flight batching, KV cache quantization
- B200 GPU support, GeForce RTX 50-series via WSL (limited models)
- Disaggregated serving with three approaches: standalone trtllm-serve, MPI/UCX backend, Triton Inference Server ensemble
- EAGLE-3 and multi-token prediction speculative decoding
- Guided decoding via XGrammar backend
- Sparse attention support
- Helix Parallelism (new), wide expert parallelism (EP)
- KV Cache Connector for disaggregated setups
- HMAC encryption for IPC sockets (security, enabled by default)
- Latest docker: pytorch:25.12-py3, PyTorch 2.9.1, transformers 4.57.3

**NVIDIA Dynamo (GTC 2025, production March 2026):**
- Open-source distributed inference framework that wraps TRT-LLM, vLLM, and SGLang
- Disaggregated prefill/decode, dynamic GPU scheduling, LLM-aware request routing, accelerated async data transfer between GPUs
- Up to 30x more requests served on Blackwell (DeepSeek-R1)
- Entered production March 2026, called "inference operating system for AI factories"
- ai-dynamo/dynamo on GitHub
- This is NVIDIA's answer to the serving layer -- not replacing TRT-LLM but orchestrating it

**Strength:** Highest raw performance on NVIDIA datacenter GPUs. FP8/NVFP4 native. Deep hardware integration. Dynamo adds the orchestration layer that TRT-LLM alone lacked.

**Weakness:** NVIDIA-only. More complex setup than vLLM/SGLang. Smaller open-source community. Build pipeline is heavier.

**Who uses it:** Large-scale NVIDIA datacenter deployments, cloud inference providers, NVIDIA NIM microservices.

---

## 6. TGI -- Historical Context (Confirmed Dead)

**Status: Maintenance mode as of December 11, 2025.** Confirmed by Lysandre Debut (HuggingFace) on X/Twitter.

**What happened:**
- Only minor bug fixes, documentation improvements, and lightweight maintenance accepted
- HuggingFace explicitly recommends vLLM or SGLang for new deployments
- Migration guidance published for existing HF Inference Endpoints users
- TGI's legacy: "initiated the movement for optimized inference engines to rely on a transformers" codebase

**Historical significance:**
- Was the default for HF Inference Endpoints
- Pioneered multi-backend support (added TRT-LLM and vLLM backends before going maintenance)
- Helped establish the pattern of optimized serving engines that now dominates

**Who still uses it:** Existing deployments that haven't migrated. No new deployments should use TGI.

---

## 7. Notable Mention: LMDeploy (The Dark Horse)

**Not in the original brief but shows up in every 2026 benchmark.**

- From InternLM (Shanghai AI Lab)
- TurboMind engine: custom CUDA kernels, persistent batch inference, LRU KV cache manager
- MXFP4 quantization support (V100+), 1.5x faster than vLLM on H800 for GPT-OSS models
- DeepSeek PD disaggregation via DLSlime and Mooncake integration
- Integrates deepseek-ai techniques: FlashMLA, DeepGemm, DeepEP, MicroBatch, eplb
- ~16,200 tok/s on H100 -- tied with SGLang, 29% ahead of vLLM
- "Dominates quantized model serving" per 2026 benchmarks

---

## The Landscape Map

```
                    SINGLE USER                    MULTI USER
                    ─────────────────────────────────────────────
  PORTABLE     │  llama.cpp                    │  llama.cpp (limited)
               │  ExLlamaV2/V3                 │  TabbyAPI
               │                               │
  NVIDIA       │  ExLlamaV2/V3 (best)          │  vLLM (default)
  CONSUMER     │                               │  SGLang (if prefix-heavy)
               │                               │
  NVIDIA       │  TensorRT-LLM                 │  TensorRT-LLM + Dynamo
  DATACENTER   │                               │  vLLM / SGLang
               │                               │  LMDeploy (dark horse)
               │                               │
  MULTI-VENDOR │  llama.cpp (RPC)              │  vLLM (broadest HW)
               │                               │  SGLang (TPU via Jax)
```

## The Worldview Summary

| Engine | Worldview | One-liner |
|--------|-----------|-----------|
| llama.cpp | Portability above all | "If it has a CPU, it can think" |
| ExLlamaV2/V3 | Squeeze every bit | "Your one GPU is enough" |
| vLLM | Production reliability | "The safe choice at scale" |
| SGLang | Workload-aware serving | "Your access pattern IS the optimization" |
| TensorRT-LLM | Vertical integration | "NVIDIA silicon deserves NVIDIA software" |
| Dynamo | Orchestration layer | "The inference OS for GPU clusters" |
| LMDeploy | Quantization + speed | "The fastest engine nobody talks about" |
| TGI | (historical) | "Pioneered the category, then stepped aside" |
