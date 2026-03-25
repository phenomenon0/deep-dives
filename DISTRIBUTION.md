# Article Readiness Ranking & Distribution Plan

## Recommendation: Release Inference Engine Part I First

**File:** `inference-engine-deep-dive.html`
**Title:** *Inference Engines — Part I: The Machine*

### Why this one leads:
- Most timely topic in tech (March 2026) — everyone wants to understand how LLMs actually run
- Broadest audience: ML engineers, backend devs, hobbyists running local models, AI-curious generalists
- One of only 5 articles with real D3.js interactive visualizations
- Series hook — Part I naturally drives readers back for Parts II and III
- Answers a universal question: "What actually happens when an LLM generates a token?"

---

## Full Readiness Ranking

### Tier 1 — Ship Now

| Rank | Article | File | Lines | Interactives |
|------|---------|------|-------|-------------|
| 1 | Inference Engine Part I: The Machine | `inference-engine-deep-dive.html` | 2597 | D3 + 3 SVGs |
| 2 | Textures | `textures-deep-dive.html` | 6586 | 38 SVGs |
| 3 | Unicode — How Poop Actually Works | `unicode-deep-dive.html` | 3028 | 12 SVGs |
| 4 | Cryptography | `crypto-deep-dive.html` | 2829 | CSS diagrams |
| 5 | GPU Compute — How Silicon Thinks in Parallel | `gpu-deep-dive.html` | 2416 | None |

### Tier 2 — Strong Follow-ups

| Rank | Article | File | Lines | Interactives |
|------|---------|------|-------|-------------|
| 6 | Audio | `audio-deep-dive.html` | 3194 | 5 SVGs |
| 7 | Compression | `compression-deep-dive.html` | 3019 | CSS only |
| 8 | Concurrency | `concurrency-deep-dive.html` | 2778 | Minimal |
| 9 | Floating Point | `floatingpoint-deep-dive.html` | 2884 | CSS only |
| 10 | Shaders | `shaders-deep-dive.html` | 2441 | 4 SVGs |

### Tier 3 — Solid Catalog Pieces

| Rank | Article | File | Lines |
|------|---------|------|-------|
| 11 | Inference Engine Part II | `inference-engine-deep-dive-2.html` | 2527 |
| 12 | Parsers | `parsers-deep-dive.html` | 2614 |
| 13 | Game Servers | `gameserver-deep-dive.html` | 2825 |
| 14 | RNG | `rng-deep-dive.html` | 2370 |
| 15 | Hash Tables | `hashtables-deep-dive.html` | 2504 |
| 16 | Allocators | `allocators-deep-dive.html` | 2551 |
| 17 | Syscalls & io_uring | `syscalls-deep-dive.html` | 2135 |
| 18 | SVG OS Foundation | `svg-os-foundation.html` | 2300 |
| 19 | Network Stack | `network-deep-dive.html` | 2073 |
| 20 | Game Characters | `gamecharacter-deep-dive.html` | 2099 |
| 21 | Inference Engine Part III | `inference-engine-deep-dive-3.html` | 1319 |

### Tier 4 — Hold or Rework Before Publishing

| Rank | Article | File | Lines | Issue |
|------|---------|------|-------|-------|
| 22 | Buffer Hacking | `buffer-hacking.html` | 1512 | Small, niche, no interactives |
| 23 | WebAssembly | `webassembly-deep-dive.html` | 1426 | 16 chapters in 1426 lines feels thin |
| 24 | Game Physics | `gamephysics-deep-dive.html` | 1384 | Small, no interactives for a visual topic |
| 25 | Buffers | `buffers-deep-dive.html` | 1618 | Wrong theme (brown `#1a1815` vs standard `#0a0a0f`), uses Inter font instead of IBM Plex Mono — visually inconsistent |
| 26 | Navigation | `navigation-deep-dive.html` | 678 | Far too small compared to other articles |
| 27 | Inference Deep Dive (legacy) | `inference-deep-dive.html` | 1104 | Superseded by 3-part series |
| 28-29 | part2/part3 duplicates | `part2-making-it-fast.html`, `part3-serving-at-scale.html` | — | Duplicates of inference-engine-deep-dive-2/3 |

---

## 10 Places to Post (for Inference Engine Part I)

1. **Hacker News** — Submit as "Show HN". #1 venue for interactive technical deep-dives.
2. **Reddit r/programming** — High-traffic subreddit that rewards long-form technical content.
3. **Reddit r/MachineLearning** — Perfect topic match. Audience craves depth over hype.
4. **Reddit r/LocalLLaMA** — Community obsessed with inference internals. Will share widely.
5. **X/Twitter** — Thread with 3-4 key insights + screenshots of D3 visualizations, link to full article.
6. **LinkedIn** — Short post with hook: "Ever wondered what happens when an LLM generates a token?"
7. **Bluesky** — Same thread format as X. Tech early-adopter demographic is ideal.
8. **Lobste.rs** — Curated HN alternative with high-signal technical audience.
9. **Dev.to** — Cross-post an excerpt with "read the full interactive version" link back to your site.
10. **Discord communities** (MLOps Community, Latent Space, EleutherAI) — Share in #resources channels.
