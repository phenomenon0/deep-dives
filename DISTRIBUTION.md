# Article Release Ranking & Distribution Plan

Ranked by **content depth**, **topic relevance** (March 2026), and **audience impact**. Interactivity is excluded as a criterion — it can always be added.

---

## Ranking Criteria

| Criterion | Weight | What it measures |
|-----------|--------|------------------|
| Content depth & quality | 40% | Chapter count, technical rigor, narrative arc, does it build toward something real |
| Relevance & timeliness | 35% | How much developers care about this topic right now (March 2026) |
| Audience impact | 25% | Breadth of audience, uniqueness vs existing online content, shareability |

---

## Full Ranking

### Tier 1 — Lead Releases

| Rank | Article | Why |
|------|---------|-----|
| 1 | **Inference Engines Part I** | 21 chapters across 3 parts. Traces a vector through the entire transformer pipeline at instruction level. AI inference is the defining engineering topic of 2026 — every developer who uses LLMs wants to understand what happens underneath. Nothing at this depth exists in long-form article format online. Part I alone covers tokenization, forward pass mechanics, and sampling with enough rigor to stand on its own, while Parts II and III create a natural content pipeline. |
| 2 | **GPU Compute** | 15 chapters from silicon architecture through CUDA optimization to profiling. Covers SMs, warps, memory hierarchy, coalescing, tensor cores, roofline analysis. The GPU compute boom (AI training/inference, rendering) makes this immediately relevant. Rare to find this level of hardware-to-software depth outside of NVIDIA's own documentation. |
| 3 | **Cryptography** | 14 chapters building from hashing through TLS to a complete encrypted messenger (Signal protocol). Covers real-world breaks (Heartbleed, POODLE, Goto Fail) and side-channel attacks. Security is perpetually relevant, and the "build a real thing" arc from first principles to working messenger is compelling. |
| 4 | **Floating Point** | 14 chapters from bit layout through NaN boxing, the fast inverse square root, FP16 vs BF16, to real disasters (Patriot missile, Ariane 5, stock exchanges). The BF16/FP16 chapter connects directly to the AI quantization zeitgeist. "The Numbers That Lie" framing is a strong hook. No comparable deep treatment exists online at this depth. |
| 5 | **Compression** | 13 chapters from Shannon entropy through Huffman, LZ77, DEFLATE, Zstandard to neural network weight compression (GPTQ, AWQ). The ML weight quantization chapter makes this timely. Bridges classical information theory with cutting-edge model optimization — a rare combination. |

### Tier 2 — Strong Follow-ups

| Rank | Article | Why |
|------|---------|-----|
| 6 | **Concurrency** | 14 chapters from cache coherence and MESI protocol through mutexes, atomics, lock-free structures, async/await, to kernel futex internals. Every backend developer needs this. Covers memory orderings and TSan — topics most articles hand-wave through. |
| 7 | **Syscalls & io_uring** | 11 chapters covering the syscall boundary, strace, mmap, epoll, io_uring ring buffer mechanics, sendfile/splice, seccomp, eBPF, vDSO. io_uring is the most important Linux I/O development in a decade. eBPF coverage adds further relevance. Deep kernel-level content that's hard to find elsewhere. |
| 8 | **Network Stack** | 14 chapters tracing a packet from application send() through kernel sk_buff structures to hardware NIC ring buffers. Covers NAPI polling, TSO, GRO, RSS, XDP programs. Complete path from userspace to wire. Systems engineers building high-performance services need this. |
| 9 | **Parsers** | 14 chapters from Chomsky hierarchy through Thompson's construction, recursive descent, Pratt parsing, PEG, parser combinators, to a working JSON parser. Formal language theory grounded in practical implementation. Perennial Hacker News appeal. |
| 10 | **Hash Tables** | 11+ chapters including Swiss tables with SIMD matching, Robin Hood hashing, Bloom filters, consistent hashing, Go maps internals. The Swiss tables explanation alone (Google's abseil implementation) is rare content. Deep algorithmic rigor. |

### Tier 3 — Solid Catalog

| Rank | Article | Why |
|------|---------|-----|
| 11 | **Allocators** | 14 chapters from free lists through buddy/slab/arena to jemalloc/mimalloc/tcmalloc internals, GC algorithms, and memory safety. Production allocator analysis is hard to find. |
| 12 | **WebAssembly** | 10 chapters with hand-written WAT, compiler output analysis, real case studies (Figma, AutoCAD, Photoshop). Practical "build a template engine" arc. Wasm adoption continues to grow. |
| 13 | **Game Servers** | 14 chapters covering tick loops, client prediction, lag compensation, anti-cheat, sharding, matchmaking. Complete real-time multiplayer architecture. Niche but deeply technical. |
| 14 | **Unicode** | 12 chapters from ASCII through grapheme clusters, emoji ZWJ sequences, normalization, to homoglyph security attacks. Every developer hits Unicode bugs. The security angle adds urgency. |
| 15 | **Textures** | 11 chapters with complete Cook-Torrance BRDF implementation, PBR material system, BC1-BC7 compression formats. Deep graphics content for rendering engineers. |
| 16 | **Audio** | Covers physics of sound, Nyquist, synthesis (additive, subtractive, FM/DX7), psychoacoustics, Web Audio API. Good breadth from physics to implementation. |
| 17 | **Shaders** | 12 chapters on rendering pipeline, GLSL, SDFs, ray marching, lighting models, demoscene context, performance cost model. Strong graphics niche content. |
| 18 | **RNG** | 12 chapters from LCGs through Mersenne Twister, CSPRNG, entropy pools, RDRAND, to distribution shaping and Monte Carlo. Covers state recovery attacks — good security angle. |
| 19 | **Inference Engine Part II** | 6 chapters on quantization, KV cache, FlashAttention, continuous batching, speculative decoding. Best released 2-3 weeks after Part I to maintain momentum. |
| 20 | **Inference Engine Part III** | 8 chapters on PagedAttention, tensor parallelism, multi-tenancy, metrics. Completes the series. Release 2-3 weeks after Part II. |

### Tier 4 — Hold or Rework

| Rank | Article | Issue |
|------|---------|-------|
| 21 | **Game Physics** | 12 chapters but narrower audience than other systems topics |
| 22 | **Buffers** | Go-specific; inconsistent styling (brown theme, Inter font vs standard) |
| 23 | **SVG OS Foundation** | Architecture design doc — different format than the teaching articles |
| 24 | **Game Characters** | Sandbox-focused, less textual depth than peers |
| 25 | **Buffer Hacking** | Practical demos but thinner content |
| 26 | **Navigation** | 11 chapters but smallest file; robotics niche is narrow |
| 27 | **Inference Deep Dive (legacy)** | Superseded by 3-part series |

---

## 10 Distribution Channels

### Primary Launch Channels

1. **Hacker News** — Submit as "Show HN" for articles with interactive elements, standard link otherwise. The single highest-value channel for long-form technical content. Timing: Tuesday-Thursday, 8-10 AM US Eastern. These articles are exactly what reaches the front page — deep, original, well-crafted technical writing.

2. **Reddit r/programming** — 6M+ subscribers. Long-form technical content performs well. Cross-post to topic-specific subreddits: r/MachineLearning and r/LocalLLaMA (inference), r/crypto (cryptography), r/gamedev (game servers), r/compsci (parsers, floating point).

3. **Lobste.rs** — Smaller, curated community of senior engineers. Less noise than HN, higher signal. Technical deep-dives are the exact content this community values. Requires invitation to post.

### Amplification Channels

4. **X/Twitter** — Thread format: 3-4 key insights from the article with the strongest hook as the first tweet. Tag domain experts (for inference: Karpathy, Hotz, etc.). Link to full article in final tweet.

5. **Bluesky** — Same thread strategy as X. Developer-heavy early adopter demographic. Technical content gets higher engagement here than on X currently.

6. **LinkedIn** — Short professional framing post (200 words) linking to the article. Algorithm favors technical content from individual accounts. Reaches engineering managers and senior engineers who share within their organizations.

### Long-tail Channels

7. **Dev.to / Hashnode** — Publish a condensed excerpt (first 2-3 chapters) as a teaser with a "read the full version" link. Built-in discovery, newsletter distribution, and strong SEO.

8. **Newsletter cross-promotion** — Pitch to relevant newsletter authors. For inference: "Latent Space" (swyx), "The Batch" (Andrew Ng). For general CS: "Pointer.io", "TLDR", "ByteByteGo" (Alex Xu). A single newsletter mention can drive 10K+ targeted views.

9. **Discord & Slack communities** — Share in #resources channels. For inference: LocalLlama Discord, MLOps Community Slack, Latent Space Discord. For crypto: Cryptography Engineering channels. High engagement, generates word-of-mouth.

10. **Personal site with RSS** — Host at a permanent URL on your own domain. Add Open Graph meta tags and a compelling preview image for rich link previews. Submit RSS to aggregators. This is the canonical URL all other channels link to — builds SEO authority and ensures you own the audience long-term.
