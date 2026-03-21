# Wasm + AI/LLMs Research Notes (Chapters 13-16)

Compiled 2026-03-20. Purely research -- every number has a source or clear provenance.

---

## CHAPTER 13: INFERENCE IN THE TAB (Running LLMs in Browser via Wasm)

### 13.1 wllama (llama.cpp Wasm bindings)

**What it is:** WebAssembly binding for llama.cpp. Runs GGUF models directly in browser via Wasm SIMD. No backend, no GPU. TypeScript, zero runtime dependencies. Inference runs inside a Web Worker (doesn't block UI).

**Models:** Any GGUF model. Recommended: Q4, Q5, Q6 quantizations for browser. Supports split models (chunks up to 2GB each for parallel loading). Tested with TinyLlama stories260K, and in production with SmolLM2-360M (369MB).

**Architecture:** Two builds -- single-thread and multi-thread, auto-switches based on browser support (multi-thread requires SharedArrayBuffer + COOP/COEP headers). Uses Wasm SIMD 128-bit intrinsics for vectorized operations.

**Performance (Wasm-only, CPU path):**
- TinyLlama-1.1B Q4 via Wasm: **2-5 tok/s** on modern hardware. Source: SitePoint WebGPU vs WebASM benchmarks, corroborated by multiple independent reports. This is the CPU-only ceiling for a 1B model in browser.
- wllama release notes mention "x2 speed for Qx_K and Qx_0 quantization" in recent versions.
- No official benchmark suite published by wllama project itself.

**Production deployment:** Firefox uses wllama as inference engine for Link Preview feature (Beta/Nightly). Model: SmolLM2-360M from HuggingFace (369MB download). "Most people see the first key point within 4 seconds and each additional point within a second." Source: Mozilla blog, 2025. Firefox 142 also exposes wllama to extension developers.

**FOSDEM 2025:** wllama presented in a 15-minute talk, introducing the project.

Sources:
- https://github.com/ngxson/wllama
- https://blog.mozilla.org/en/firefox/firefox-ai/ai-link-previews-firefox/
- https://www.sitepoint.com/webgpu-vs-webasm-transformers-js/ (benchmark numbers)

---

### 13.2 WebLLM / MLC LLM

**What it is:** High-performance in-browser LLM inference engine from MLC AI. Uses WebGPU for GPU compute + Wasm for CPU operations. Companion to MLC-LLM (universal deployment across hardware).

**Architecture (the key insight):**
1. **Compilation pipeline:** MLC-LLM + Apache TVM compile model Python code into two artifacts: (a) converted weights, (b) a Wasm library containing WebGPU kernels + non-kernel functions.
2. **WebGPU kernels:** GPU-accelerated matmuls, compiled ahead-of-time via MLC-LLM. Graph-level optimizations (kernel fusion) + operator-level optimizations (GEMM tiling).
3. **Wasm handles:** Grammar engine for structured generation, tensor manipulation, CPU-side orchestration. Compiled from C++ via Emscripten.
4. **Frontend:** ServiceWorkerMLCEngine exposes OpenAI-style API. Background MLCEngine runs in a Web Worker via message-passing (UI never blocked).

**Performance numbers (from arXiv 2412.15803, WebLLM paper):**

| Model | WebLLM tok/s | Native MLC-LLM tok/s | % of Native |
|-------|-------------|----------------------|-------------|
| Llama-3.1-8B (Q4) | 41.1 | 57.7 | 71.2% |
| Phi-3.5-mini (3.8B) | 71.1 | 89.3 | 79.6% |

Hardware: Apple MacBook Pro M3 Max, Chrome Canary 133.0.6870.0 (arm64).

**Key claim:** "WebLLM can retain up to **80% native performance** on the same device, with room to further close the gap." Source: arXiv 2412.15803 abstract.

**Model support:** Llama 3/3.1, Phi 3/3.5, Gemma, Mistral, Qwen, and many others.

**WebGPU support coverage:** ~70-75% of mobile users, 90%+ desktop. Source: BuildMVPFast 2026 analysis.

Sources:
- https://arxiv.org/abs/2412.15803 (WebLLM paper)
- https://arxiv.org/html/2412.15803v1 (full HTML with tables)
- https://github.com/mlc-ai/web-llm
- https://webllm.mlc.ai/

---

### 13.3 whisper.cpp in Browser

**What it is:** Wasm port of whisper.cpp (OpenAI Whisper in C/C++). Runs speech recognition entirely in browser.

**Real-time factor:** For tiny and base models on modern CPU + browser: **2x-3x real-time** (60 seconds of audio transcribed in ~20-30 seconds). Source: whisper.cpp Wasm README.

**Model sizes:**
- tiny.en: 75MB (standard), 31MB (Q5_1 quantized)
- base.en: 57MB (Q5_1 quantized)
- Can run all models up to "small" size; beyond that, memory and performance are "unsatisfactory"

**Architecture:** Uses Wasm SIMD 128-bit intrinsics. Browser must support WASM SIMD. Audio processed locally, never leaves user's device.

**Real-time streaming:** Dedicated "stream" example provides real-time transcription via Wasm in browser. Not true real-time (2-3x real-time factor means slight lag), but usable for near-real-time transcription of the tiny model.

**Answer to "can it transcribe in real-time?":** Not quite for the Wasm path alone. At 2-3x real-time factor, a 60s clip takes 20-30s. True real-time would require 1x or better. For the tiny.en model on fast hardware, it approaches usable for streaming (with buffering), but is not instantaneous.

Sources:
- https://github.com/ggml-org/whisper.cpp/blob/master/examples/whisper.wasm/README.md
- https://ggml.ai/whisper.cpp/ (live demo)
- https://ggml.ai/whisper.cpp/stream.wasm/ (streaming demo)

---

### 13.4 WebGPU + Wasm Split

**Why Wasm alone isn't enough:**
- Wasm SIMD runs on CPU only. For TinyLlama-1.1B token generation: **2-5 tok/s** via Wasm CPU path.
- WebGPU on discrete NVIDIA RTX GPU: **25-40 tok/s** for same model. That's 5-20x faster.
- For small models with short inputs (<128 tokens), Wasm can actually match or beat WebGPU (GPU dispatch overhead: buffer upload, shader dispatch, readback exceeds computation). M2 MacBook Air: Wasm ~8-12ms vs WebGPU ~15-25ms for single embedding inference.
- **Crossover:** WebGPU wins decisively for larger models and token generation. Wasm wins for tiny inference calls.

**The architecture that emerged:** WebGPU handles the heavy matmuls (attention, FFN). Wasm handles everything else -- tokenization, sampling, grammar engine, orchestration logic. This is exactly what WebLLM does.

**Relaxed SIMD:** Shipped in Chrome 114. Speeds up existing workloads 1.5-3x by reducing strict non-determinism requirements. New dot product and FMA instructions.

**ONNX Runtime Web comparison:**
- Segment Anything encoder: WebGPU is **19x faster** than Wasm backend
- Segment Anything decoder: WebGPU is **3.8x faster**
- Hardware: NVIDIA RTX 3060, Intel Core i9 laptop
- Source: Microsoft blog, Feb 2024

Sources:
- https://developer.chrome.com/blog/io24-webassembly-webgpu-1
- https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/
- SitePoint WebGPU vs WebASM benchmarks

---

### 13.5 The 4GB Limit

**The constraint:** Wasm uses 32-bit pointers (wasm32). Maximum addressable memory: 2^32 bytes = ~4.3GB. This is a hard ceiling in the spec.

**What fits:**
- TinyLlama-1.1B Q4: ~700MB -- fits easily
- Llama-3.1-8B Q4: ~4.5GB -- does NOT fit in Wasm linear memory alone
- SmolLM2-360M: 369MB -- comfortable

**How WebLLM gets around it:** WebGPU has its own memory space separate from Wasm linear memory. Model weights live in GPU memory (via WebGPU), not Wasm memory. Wasm handles orchestration/CPU logic, which uses far less than 4GB. The 4GB limit applies to Wasm's linear memory, not to the total system.

**Google's innovation (MediaPipe, Gemma 7B in browser):**
- Problem: Gemma 1.1 7B = 8.6GB, "several times larger than any model we've run in browser previously"
- Solution: Layer-by-layer async streaming. Instead of loading all 28 transformer layers into Wasm memory at once, load each layer's weight buffer sequentially, transfer to WebGPU device memory, free Wasm buffer.
- Result: Peak Wasm memory usage dropped to **<1% of synchronous loading requirements**
- C++ engine calls out to JavaScript to request each weight buffer on demand
- Four memory layers: File reading (JS) -> JS memory -> Wasm memory -> WebGPU device memory
- Source: Google Research blog, "Unlocking 7B+ language models in your browser"

**Memory64 proposal:** Removes the 4GB heap limit by enabling 64-bit pointers. Particularly important for "large models like we have today." Chrome implementation shipping (as of mid-2024 Google I/O talk).

**Browser per-tab limit:** Chrome has a "rather generous per-tab limit (about 16GB)" for JavaScript memory. Source: Google Research blog.

Sources:
- https://v8.dev/blog/4gb-wasm-memory
- https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/
- https://developer.chrome.com/blog/io24-webassembly-webgpu-1

---

### 13.6 Performance Gap: Browser vs Native

**WebLLM (best case, WebGPU path):** 71-80% of native. Source: arXiv 2412.15803.
- Llama-3.1-8B: 71.2% of native
- Phi-3.5-mini: 79.6% of native

**Academic study (broader picture, less favorable):**
- Average disparity: **16.9x on CPU, 30.6x on GPU** on PC devices
- Mobile: **15.8x CPU, 7.8x GPU**
- Source: "Anatomizing Deep Learning Inference in Web Browsers" (ACM TOSEM, 2024)
- NOTE: This study measures broader DL inference, not just optimized LLM engines. WebLLM's MLC compilation closes the gap dramatically vs naive browser inference.

**INT8 quantization benefit for Wasm:** SIMD instructions operate on packed 8-bit integers natively. INT8 model on Wasm runs **2-3x faster** than FP32 equivalent.

**Critical dependency:** SharedArrayBuffer requires COOP/COEP headers. Without them, falls back to single-threaded Wasm, which is **3-4x slower** for larger models.

---

## CHAPTER 14: THE SANDBOX (Wasm for LLM Code Execution)

### 14.1 Extism

**What it is:** Open-source universal plug-in system powered by WebAssembly. Product of Dylibso. V1 released January 2024.

**Architecture:** Extism kernel is itself implemented as a Wasm module running inside Wasmtime's Linker. Plugins are Wasm modules that communicate with the kernel.

**Performance (from Dylibso "Back of the Napkin" blog post):**
- Per-call overhead: **4.75-6.7 nanoseconds** per kernel function call
- Consume benchmark (one-way data transfer): 64KB in **49 microseconds** (1.25 GiB/s throughput)
- Echo benchmark (round-trip): 64KB in **78 microseconds** (801 MiB/s)
- Reflect benchmark (full roundtrip host->guest->host->guest->host): 64KB in **224 microseconds**
- Throughput at scale: 6.25 MiB payload in ~22ms (~285 MiB/s)

**Instantiation time:** The blog post does not give a single "instantiation time" number. However, Wasmtime (which Extism uses) achieves **5 microsecond** module instantiation for SpiderMonkey.wasm (from 2ms down to 5us, a 400x improvement). Source: Bytecode Alliance "Wasmtime 1.0 Performance" blog. The techniques: copy-on-write memory initialization, lazy initialization of function imports, pooling allocator.

**Note on the "40 microseconds" claim:** This specific number was not found in any Extism/Dylibso source. The Wasmtime figure is 5 microseconds for instantiation. Extism's overhead per plugin call is in the nanosecond-to-microsecond range depending on payload size. The 40us figure may be an older or aggregate measurement (e.g., instantiation + first call + teardown).

Sources:
- https://dylibso.com/blog/how-does-extism-work/
- https://bytecodealliance.org/articles/wasmtime-10-performance
- https://github.com/extism/extism

---

### 14.2 LLM Code Execution Pattern

**The pattern:** LLM generates code -> compile/wrap to Wasm -> sandbox execute -> return result.

**Who does this:**

1. **NVIDIA (Pyodide approach):** LLM generates Python (e.g., Plotly visualizations) -> wrap in HTML with Pyodide (CPython compiled to Wasm) -> execute in user's browser sandbox. No server-side eval(). Code stays confined to browser sandbox. Source: NVIDIA blog "Sandboxing Agentic AI Workflows with WebAssembly" (Dec 16, 2024).

2. **Shopify Functions:** Merchant/developer code compiled to Wasm -> executed in Wasm sandbox with strict limits. 5ms execution limit. 11 million instruction cap. Uses Wizer for ahead-of-time pre-initialization (eliminates cold start). Lucet runtime (later likely migrated). Module execution: ~100 microseconds. Total with I/O: ~4ms at p99. Scale: 100k modules/minute in load testing. Source: Shopify Engineering blog.

3. **Fermyon Spin:** Wasm components with built-in LLM inference via WASI-NN. Agents packaged as Wasm components, deployed on Akamai edge. OpenAI Agents SDK integration.

4. **Microsoft Wassette:** Wasm Components exposed as MCP tools. AI agents (e.g., GitHub Copilot) autonomously discover and load Wasm Components from OCI registries. Deny-by-default capability system. Cryptographic signing via Notation/Cosign. Source: Microsoft Open Source blog, Aug 2025.

5. **amla-sandbox:** Wasm sandbox with capability enforcement for AI agents. Unforgeable capability tokens. First run compiles Wasm module (~300ms), subsequent loads ~0.5ms. 13MB statically-linked binary. No Docker, no subprocess. Source: amlalabs.com, HN discussion.

6. **Hugging Face smolagents:** Code agents that can use sandboxed environments including Pyodide/Wasm sandbox.

---

### 14.3 Docker vs Firecracker vs Wasm Cold Start

| Technology | Cold Start | Memory Overhead | Notes |
|-----------|-----------|----------------|-------|
| Docker containers | ~50ms (bare), 800ms-1.5s real-world | ~10MB per container | Real-world includes image layers, networking |
| Firecracker microVMs | ~125ms (cold), 30ms (pre-configured), ~150ms (E2B with snapshots) | ~5MB per microVM | Used by AWS Lambda, E2B |
| gVisor | 60-75ms (20-50% overhead vs containers) | ~30MB | Google's container sandbox |
| Wasm (Wasmtime) | **5 microseconds** (module instantiation) | Kilobytes | 400x faster than 2ms starting point |
| Wasm (Spin) | **<0.5ms** (sub-half-millisecond) | Minimal | Fermyon's production claim |
| Traditional VMs | Seconds | Hundreds of MB | Baseline comparison |

**Key comparisons:**
- Wasm vs Docker: ~10,000x faster cold start (5us vs 50ms)
- Wasm vs Firecracker: ~25,000x faster (5us vs 125ms)
- E2B (Firecracker + snapshots): ~150ms, specifically optimized for AI agent workflows, used by 88% of Fortune 100

Sources:
- https://bytecodealliance.org/articles/wasmtime-10-performance (5us Wasmtime)
- https://www.softwareseni.com/firecracker-gvisor-containers-and-webassembly-comparing-isolation-technologies-for-ai-agents/
- https://e2b.dev/ (E2B 150ms)
- https://www.fermyon.com/wasm-functions (Spin <0.5ms)

---

### 14.4 Capability-Based Security for LLM Code

**Why Wasm is perfect for AI-generated code:**

1. **Zero default permissions:** "A WASM module is inert by default with no intrinsic ability to access the file system, network, or any other external resource." Host must explicitly provide capabilities via imports at instantiation. This is EXACTLY what you want for untrusted LLM-generated code.

2. **Deny-by-default:** Components declare capabilities up front. Host enforces at runtime. Not "best effort" -- enforced by the sandbox itself.

3. **No ambient authority:** Unlike a Docker container (which inherits host capabilities unless restricted), Wasm starts with nothing. You add permissions, not remove them.

4. **Attenuation:** When Agent A delegates to Agent B, B can only receive a subset of A's authority. Cryptographically enforced (amla-sandbox). This maps perfectly to multi-agent AI systems.

5. **Memory isolation by design:** Linear memory is bounds-checked. No way to escape to host address space. Buffer overflows crash the module, not the host.

**Practical framing:** Regex sanitization of LLM output is "usually bypassed." VM/Firecracker isolation is "resource and engineering intensive." Wasm is the middle ground -- robust isolation, minimal overhead, no infrastructure changes.

Sources:
- https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/
- https://opensource.microsoft.com/blog/2025/08/06/introducing-wassette-webassembly-based-tools-for-ai-agents/
- https://amlalabs.com/sandbox/

---

### 14.5 NVIDIA Blog Post: "Sandboxing Agentic AI Workflows with WebAssembly"

Published December 16, 2024.

**Problem:** Agentic AI workflows generate Python code (e.g., data visualizations via Plotly). Running this via eval() on application servers is dangerous. Prompt injection and code errors can compromise the server and other users.

**Three approaches compared:**
1. **Regex/restricted runtimes:** "Insufficient" and "usually bypassed"
2. **MicroVM (Firecracker):** "Resource and engineering intensive" -- requires dedicated infrastructure
3. **WebAssembly (Pyodide):** Lightweight middle ground, robust isolation, minimal architecture changes

**Their solution:**
- Use Pyodide (CPython compiled to Wasm) in the user's browser
- LLM generates Python code -> application wraps it in HTML template with Pyodide runtime -> served to user's browser -> browser executes code in sandbox -> visualization renders locally
- Dependencies installed via micropip: `await micropip.install('plotly')`
- Browser sandbox provides OS and user isolation automatically
- Malicious code either fails (missing dependencies) or stays confined

**Key benefits:**
- "Cost-effective by reducing compute requirements" (compute shifts to client)
- No cross-user contamination
- No changes to LLM prompts needed
- Inherits browser security model

Source: https://developer.nvidia.com/blog/sandboxing-agentic-ai-workflows-with-webassembly/

---

## CHAPTER 15: INFERENCE AT THE EDGE (Wasm for Serving AI)

### 15.1 Cloudflare Workers AI

**Architecture:**
- Workers run on V8 isolates (not containers) across 330+ data centers, 180+ edge locations
- V8 isolates: **sub-1ms cold start** (vs Lambda 100-1000ms). ~1/10th memory of a Node.js process. Start in under 5ms.
- Workers handle orchestration (JavaScript/TypeScript/Wasm). GPU inference handled by separate GPU clusters.

**Omni (internal AI platform):**
- Single control plane for running multiple models per GPU
- Scheduler automatically provisions models, spawns instances, routes requests
- **Memory over-commitment:** Currently configured to run 13 models, allocating ~400% GPU memory on a single GPU, saving 4 GPUs worth of hardware
- Mechanism: Injects CUDA stub library that intercepts cudaMalloc, forces unified memory mode (GPU + CPU share address space)
- Overrides cudaMemGetInfo to expose only a subset of memory to each model
- Process-level isolation per model, IPC communication with scheduler

**Infire:** Cloudflare's custom LLM inference engine, built in Rust. Existing engines not efficient for globally distributed deployment.

**Performance:**
- ~40ms inference time for 7B parameter LLMs (per one source, may be p50)
- Swapping a 5GB model back to GPU: ~156ms via PCIe 4.0 (32 GB/sec)
- 4,000% year-over-year increase in inference requests (Q1 2026)

**Model optimization:** Models optimized for edge via ONNX and WebAssembly. WebGPU now available in Workers (blog.cloudflare.com/webgpu-in-workers/).

Sources:
- https://blog.cloudflare.com/how-cloudflare-runs-more-ai-models-on-fewer-gpus/
- https://www.gocodeo.com/post/running-ai-at-the-edge-how-cloudflare-workers-support-serverless-intelligence
- https://blog.cloudflare.com/workers-ai/
- https://blog.cloudflare.com/webgpu-in-workers/

---

### 15.2 Fermyon Spin Cold Start

**Claimed performance:** Cold starts under half a millisecond (<0.5ms). Source: Fermyon marketing materials and blog posts, consistently cited.

**Context vs competitors:**
- AWS Lambda: 100-500ms cold starts
- Azure Functions: 200-500ms
- Spin: <0.5ms -- orders of magnitude faster

**Fermyon Wasm Functions on Akamai:** Positioned as "the fastest edge functions platform on the planet." Built on Akamai's global edge infrastructure.

**SpinKube:** Wasm on Kubernetes. Contributes to CNCF (accepted January 2025). Claim: "50x more applications per node" compared to traditional containers. Source: Fermyon press release, March 2024.

**Independent benchmarks:** No independent third-party benchmarks found that verify the 0.5ms claim specifically. The number comes from Fermyon's own testing. However, the Wasmtime 5-microsecond instantiation number (Bytecode Alliance) is well-documented and would support sub-millisecond end-to-end for a minimal Spin handler.

Sources:
- https://www.fermyon.com/wasm-functions
- https://www.fermyon.com/blog/introducing-spin-v3
- https://www.globenewswire.com/news-release/2024/03/13/2845676/0/en/Fermyon-Delivers-the-First-WebAssembly-Platform-for-Kubernetes-Enabling-50x-More-Applications-Per-Node.html

---

### 15.3 WASI-NN

**What it is:** WASI API for performing ML inference. A Wasm module declares "I need neural network inference" by importing the wasi-nn interface. The host runtime provides the actual backend.

**Core API (4 operations):**
1. `load()` -- load model graph (specifying encoding + execution target)
2. `init_execution_context()` -- create inference session
3. `set_input()` / `compute()` -- provide tensors, run inference
4. `get_output()` -- retrieve results

**Supported graph encodings:** openvino, onnx, tensorflow, pytorch, tensorflowlite

**Execution targets:** CPU, GPU, TPU

**Implementations:**
- Wasmtime: First implementation, OpenVINO backend
- WasmEdge: OpenVINO backend, ONNX via native runtime or Tract (pure Rust)
- Wasm Workers Server 1.5: WASI-NN with OpenVINO on host side
- Fermyon Spin: One of the first WASI-NN implementations, Spin v1.5 added LLM inferencing (infer + generateEmbeddings methods)

**The key abstraction:** A Wasm module doesn't know or care whether inference runs on CPU, GPU, or NPU. It calls the WASI-NN API. The host decides the backend. Same binary, different hardware. This is the portable inference promise.

**Current version:** wasi-nn@0.2.0-rc-2024-10-28

Sources:
- https://github.com/WebAssembly/wasi-nn
- https://dev.to/vaib/revolutionizing-edge-ai-deploying-models-with-webassembly-and-wasi-nn-d0h
- https://developer.fermyon.com/spin/v3/serverless-ai-api-guide

---

### 15.4 Multi-Tenant Isolation

**Density advantage:**
- Wasmer: "half million apps running on a handful of servers"
- Each app isolated in own Wasm instance with separate memory and execution state
- Compiled code pages shared read-only across tenants (unlike containers which replicate binaries)
- Possible to host **thousands of isolated "nanoprocesses" within a single OS process**
- Cloud providers can squeeze **10x to 100x more workloads** onto same hardware vs containers
- Memory overhead per Wasm instance: **kilobytes** (vs containers: ~10MB, VMs: hundreds of MB)
- Less than 5% runtime overhead for async Wasm runtimes

**Comparison table:**

| Runtime | Memory per instance | Isolation level |
|---------|-------------------|----------------|
| Wasm instance | Kilobytes | Process-internal, bounds-checked |
| Docker container | ~10MB | Namespace/cgroup |
| Firecracker microVM | ~5MB | Hypervisor |
| gVisor | ~30MB | Kernel-level |
| Traditional VM | Hundreds of MB | Full hardware |

**Academic support:** Recent research (WAMR integration) reduces memory usage by 11-78% per container compared to existing Wasm runtimes. Source: Jansen et al., VU Amsterdam 2025.

Sources:
- https://wasmer.io/posts/wasm-clouds-the-world-after-containers
- https://www.softwareseni.com/firecracker-gvisor-containers-and-webassembly-comparing-isolation-technologies-for-ai-agents/

---

### 15.5 Two-Tier Architecture

**The pattern that's emerging everywhere:**

**Tier 1: Wasm orchestration (cold start: microseconds)**
- Handles request routing, input validation, prompt construction, response formatting
- Runs on V8 isolates (Cloudflare) or Wasm runtimes (Fermyon, Fastly)
- Scales to thousands of concurrent instances with kilobytes each
- Sub-millisecond cold start

**Tier 2: GPU inference (stays warm)**
- Handles actual model inference (matmuls, attention, FFN)
- GPU processes are long-lived, shared across tenants
- Cold start for GPU is expensive (loading model weights: seconds to minutes)
- Solution: keep GPU processes warm, route from Tier 1

**Examples:**
- **Cloudflare:** Workers (V8 isolate) -> Omni (GPU scheduler) -> Model process
- **Fermyon:** Spin component (Wasm) -> WASI-NN -> GPU backend
- **WebLLM in browser:** ServiceWorker (JS) -> Web Worker (Wasm+WebGPU) -> GPU

**Why this matters:** You get the density/cost benefits of Wasm for the orchestration layer (which handles 99% of requests' lifecycle) while keeping expensive GPU resources for the part that actually needs them. The orchestration layer can scale horizontally at near-zero marginal cost.

---

## CHAPTER 16: THE CONVERGENCE (Universal AI Runtime Thesis)

### 16.1 ONNX Runtime Web

**Three execution providers:**

1. **Wasm (CPU):** Default CPU backend. Uses WebAssembly compilation. Best for very small models or no-GPU environments. All ONNX operators supported.

2. **WebGPU (GPU):** Default GPU backend. Hardware-accelerated. Only a subset of operators supported. Massive speedups for suitable models.

3. **WebNN:** Potential near-native performance. Not enabled by default -- must be manually enabled in browser settings. Routes to platform-native ML frameworks.

**Performance numbers:**
- Segment Anything encoder: WebGPU **19x** faster than Wasm
- Segment Anything decoder: WebGPU **3.8x** faster than Wasm
- Stable Diffusion Turbo: Image generation within 1 second on RTX 4090
- Llama-3.2-1B in Chrome with WebGPU: ~10 tok/s (September 2025)
- 2024 arXiv study: browser GPU inference ~5x slower than native GPU; CPU/Wasm ~15-17x behind native CPU

**WebGPU FP16:** Reduces GPU memory usage and bandwidth, accelerates computation. Source: Microsoft blog.

Sources:
- https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/
- https://onnxruntime.ai/docs/tutorials/web/

---

### 16.2 TensorFlow.js Wasm Backend

**Architecture:** Uses XNNPack (heavily optimized neural network operator library) compiled to Wasm. XNNPack provides the core compute kernels.

**SIMD speedups:**
- SIMD alone: **1.7-4.5x** over plain Wasm
- Multithreading alone: **1.8-2.9x** additional speedup on top of SIMD
- Combined SIMD + multithreading: **up to 10x** over plain Wasm
- These gains are independent (multiplicative)

**Model-specific benchmarks:**
- **BlazeFace** (0.1M params, 20M multiply-adds): Wasm backend 8.2-19.8x faster than plain JS. Comparable to WebGL backend.
- **MobileNet V2** (3.5M params, 300M multiply-adds): Wasm 3x-11.5x faster than plain JS. 5.3-7.7x slower than WebGL.

**INT8 quantization with XNNPack:** **20x speedup** for in-browser inference vs default TFLite quantized kernels. Source: TensorFlow blog "Faster Quantized Inference with XNNPACK."

**Threading caveat:** Chrome 92+ requires Cross-Origin Isolation (COOP/COEP headers) for SharedArrayBuffer. Without it, falls back to SIMD-only (no multithreading), losing the 1.8-2.9x threading benefit.

Sources:
- https://blog.tensorflow.org/2020/09/supercharging-tensorflowjs-webassembly.html
- https://blog.tensorflow.org/2021/09/faster-quantized-inference-with-xnnpack.html
- https://blog.tensorflow.org/2020/03/introducing-webassembly-backend-for-tensorflow-js.html

---

### 16.3 MediaPipe Wasm

**What it does:** Real-time ML inference in browser, compiled to Wasm + WebGPU.

**Face Mesh:** **468 3D face landmarks** in real-time. Uses two DNNs: (1) face detector on full image, (2) landmark regression on detected face region. Attention mechanism concentrates compute on high-variance regions (eyes, lips) while spending less on rigid areas (cheeks).

**Performance:** **30+ fps** on standard webcam in browser. Near real-time on mid-tier devices. Source: Google Research, MediaPipe documentation.

**Hand tracking:** **21 3D joint positions per hand**. High-fidelity tracking from single RGB frame.

**Pose detection:** **33 3D body landmarks** + background segmentation mask. Full body from RGB video.

**Holistic (combined):** Simultaneously tracks 468 face landmarks + 21 hand joints (per hand) + 33 body landmarks. All at real-time rates.

**Architecture:**
- Detection runs once, then lightweight tracking model follows target across frames (amortizes expensive detection)
- GPU pre-processing: ~1.5x pipeline speedup
- Browser deployment: `FilesetResolver.forVisionTasks()` downloads MediaPipe's Wasm runtime
- All processing local -- data never leaves device

**The impressive number:** 468 + 21*2 + 33 = **554 landmarks simultaneously tracked at 30+ fps** in a browser tab.

Sources:
- https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html
- https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
- https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/

---

### 16.4 WebNN API

**What it is:** W3C specification (Candidate Recommendation, updated January 22, 2026) providing graph-based neural network API for the web platform.

**The key difference from WebGPU:** WebGPU treats a graphics API as an inference runtime (writing shaders, managing bindings, re-implementing scheduling). WebNN provides a higher-level abstraction specifically designed for inference. "Think of it as `canvas` for neural networks." Source: Tarek Ziade blog.

**NPU access:** WebNN is currently the **only web API** that provides access to NPU. Routes to GPU, NPU, or CPU without developer intervention.

**Backend mapping:**
- Windows 11 24H2+: DirectML (ONNX Runtime / Windows ML)
- macOS 14.4+ (Apple Silicon): Core ML
- Linux, ChromeOS, Android: TFLite + XNNPACK
- CPU fallback: always available

**Operator coverage:** 95 ops across Core ML, DirectML, TFLite/XNNPACK backends.

**Browser support status (as of March 2026):**
- Chrome 146 Beta: WebNN in Origin Trial
- Chromium initial support: M112 (March 2023), but behind flags
- Edge: planning to launch roughly same time as Chrome
- Firefox: "interest shown" but no public progress
- Safari/WebKit: not mentioned in any compatibility docs
- **NOT stable yet.** Still experimental / Origin Trial phase.

**The W3C timeline:** Comments on updated spec accepted until March 22, 2026. Feedback determines if it advances to full standard. Over 100 significant changes between April 2024 and January 2026, including MLTensor API, abstract device selection, third wave of transformer operators.

Sources:
- https://www.w3.org/TR/webnn/
- https://webnn.io/en/api-reference/browser-compatibility/api
- https://blog.ziade.org/2025/11/21/why-webnn-is-the-future-of-ai-in-browsers/
- https://learn.microsoft.com/en-us/windows/ai/directml/webnn-overview

---

### 16.5 The Thesis: Wasm as Portable Floor, Hardware Accelerators as Ceiling

**The core argument:** Wasm provides a universal, portable execution layer (the "floor") that runs everywhere -- browser, edge, cloud, IoT. Hardware-specific accelerators (GPU via WebGPU, NPU via WebNN, custom silicon via vendor APIs) provide the performance ceiling. Same code, different performance based on available hardware.

**Who articulates this:**

1. **Solomon Hykes** (Docker co-founder), 2019: "If WASM+WASI existed in 2008, we wouldn't have needed to created Docker. That's how important it is. WebAssembly on the server is the future of computing." Source: Twitter/X, March 2019.

2. **Google Chrome team** (Google I/O 2024): "Special purpose compute on the GPU or accelerators can offer performance that is orders of magnitude higher, especially for larger models and on high-end devices." Wasm provides CPU baseline; WebGPU/WebNN provide acceleration. Source: Chrome developer blog.

3. **WASI-NN design:** Module doesn't know or care about hardware. Calls standardized API. Host runtime routes to best available backend (CPU, GPU, TPU, NPU). Same binary, different hardware.

4. **Component Model + WIT + WASI-NN:** The same component can run in browser, on edge, or in cloud with consistent behavior. Language-agnostic modules linked like LEGO blocks. Source: Wasm Radar / CNCF discussions.

5. **Fermyon (Wasm I/O 2025):** "WebAssembly's role as a universal abstraction layer for hardware and GPU/CPU architectures offers advantages to teams trying to address different deployment targets."

**The layered picture:**
```
Application code (any language)
    |
    v
WebAssembly Component (portable binary)
    |
    +---> WASI-NN API (inference)
    |        |
    |        +---> CPU (XNNPack, ONNX, TFLite) -- always available
    |        +---> GPU (WebGPU, CUDA, Metal) -- if present
    |        +---> NPU (WebNN, Core ML, DirectML) -- if present
    |
    +---> WASI filesystem, network, etc. (capability-based)
```

**Why this matters for AI:** You write your AI orchestration logic once, compile to Wasm. It runs in a browser tab, on a Cloudflare edge node, on an IoT device, on a Kubernetes pod. The inference backend automatically uses whatever acceleration is available. The gap between "runs everywhere" and "runs fast everywhere" is being closed by the combination of Wasm portability + hardware-specific acceleration APIs.

Sources:
- https://x.com/solomonstre/status/1111004913222324225 (Solomon Hykes tweet)
- https://developer.chrome.com/blog/io24-webassembly-webgpu-1
- https://www.fermyon.com/blog/ai-workloads-panel-discussion-wasm-io-2024
- https://github.com/WebAssembly/wasi-nn

---

## Cross-Chapter Key Numbers Reference

| Metric | Number | Source |
|--------|--------|--------|
| WebLLM Llama-3.1-8B tok/s (WebGPU, M3 Max) | 41.1 | arXiv 2412.15803 |
| WebLLM % of native (best case) | 80% | arXiv 2412.15803 |
| Wasm-only TinyLlama-1.1B tok/s (CPU) | 2-5 | SitePoint benchmarks |
| WebGPU TinyLlama-1.1B tok/s (RTX GPU) | 25-40 | SitePoint benchmarks |
| whisper.cpp tiny.en real-time factor (Wasm) | 2-3x real-time | whisper.cpp README |
| Wasmtime instantiation | 5 microseconds | Bytecode Alliance |
| Fermyon Spin cold start | <0.5ms | Fermyon |
| Docker cold start | ~50ms | Multiple sources |
| Firecracker cold start | ~125ms | Multiple sources |
| E2B sandbox boot (Firecracker + snapshot) | ~150ms | E2B |
| Shopify Functions execution time (p99) | ~4ms | Shopify Engineering |
| Wasm memory limit (wasm32) | 4GB (2^32) | Wasm spec |
| Chrome per-tab memory limit | ~16GB | Google Research |
| Extism per-call overhead | 4.75-6.7 ns | Dylibso blog |
| ONNX RT Web: WebGPU vs Wasm (SAM encoder) | 19x faster | Microsoft blog |
| TF.js Wasm SIMD+threading speedup | up to 10x | TensorFlow blog |
| MediaPipe face landmarks count | 468 | Google/MediaPipe |
| MediaPipe browser framerate | 30+ fps | Google/MediaPipe |
| Cloudflare GPU memory over-commit | 400% (13 models/GPU) | Cloudflare blog |
| Cloudflare Workers cold start | sub-1ms | Cloudflare |
| Firefox wllama model (SmolLM2-360M) | 369MB | Mozilla blog |
| Wasm tenant density vs containers | 10-100x | Wasmer |
| WASI-NN supported encodings | 5 (openvino, onnx, tf, pytorch, tflite) | WASI-NN spec |
| WebNN operators | 95 | WebNN spec |
| WebNN browser support | Chrome 146 Beta (Origin Trial) | webnn.io |
