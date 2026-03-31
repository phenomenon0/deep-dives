# GGUF & AWQ File Formats Deep Dive — Research Notes

Compiled 2026-03-31. Binary-level format anatomy, architecture-specific layouts, and cross-model comparison.

---

## 1. Why File Formats Matter

A model is not just a bag of weights. Every inference runtime needs **quantization parameters**, **architectural hyperparameters** (layer counts, head dimensions, RoPE scaling factors), a **tokenizer vocabulary**, and enough metadata to reconstruct the computation graph without out-of-band documentation. How that information is packed on disk determines what tooling can load it, how fast it loads, and whether it survives format evolution.

Two ecosystems diverged from different priorities. **GGUF** (llama.cpp ecosystem) optimized for a single self-contained file that works on CPU, is memory-mappable, and carries everything needed for inference — tokenizer included. **SafeTensors + JSON sidecars** (HuggingFace / vLLM / AWQ ecosystem) optimized for GPU workflows, splitting weights from metadata across multiple files and relying on a shared schema rather than a binary container spec.

Understanding the binary layout has concrete payoffs: you can debug conversion failures instead of guessing at them, verify quantization integrity by inspecting block headers directly, build custom loaders or format translators, and optimize loading pipelines by knowing exactly which bytes are padding versus data. This guide covers the byte-level anatomy of both formats, then shows how **LLaMA**, **Qwen**, and **GPT-family** models differ structurally on disk.

---

## 2. Evolution: GGML → GGJT → GGUF

### GGML (late 2022)

**GGML** was created by **Georgi Gerganov** alongside the `ggml` tensor library as a minimal binary format for quantized inference on consumer hardware. Its design reflected the prototype stage of the project.

- Hyperparameters stored as a **flat list of untyped values** — position-dependent, no field labels. A 32-bit int at offset 0 was `n_vocab`; the next was `n_embd`; and so on, purely by convention.
- Each model architecture (LLaMA, GPT-2, Falcon, etc.) required **bespoke loading code** that knew which value lived at which position. Adding a new architecture meant writing a new loader from scratch.
- The **tokenizer was stored in separate files**, not embedded — consumers had to source `tokenizer.model` or `vocab.json` independently.
- Any change to the hyperparameter list **broke all existing model files** with no recovery path.
- **No versioning field** — impossible to detect format mismatches at load time; silent corruption was the failure mode.

### GGMF

**GGMF** was a minimal patch on GGML: a version field was prepended to the header. Otherwise structurally identical. Only one version (`v1`) was ever released, and the underlying position-dependence of hyperparameters was left untouched. It served as a stopgap.

### GGJT (v1–v3)

**GGJT** addressed the most pressing performance problem: memory mapping.

- Added **tensor alignment** — each tensor padded to a 32-byte boundary so it can be `mmap`'d directly without copying data into a separate buffer. This made cold-start times and RAM overhead meaningfully better on large models.
- Versions v1, v2, and v3 are **structurally identical** at the container level but carry incompatible quantization encodings. New quantization types (`Q4_K`, `Q5_K`, `Q6_K` — the K-quants) required GGJT v3 because older runtimes could not interpret the new block layouts.
- **No extensible metadata** — adding a new hyperparameter (e.g., a new RoPE scaling parameter) still required updating every loader and bumping the positional schema.
- Used primarily by `llama.cpp` for LLaMA-family models; the raw `ggml` examples continued using the original GGML format.

### GGUF (August 21, 2023)

**GGUF** was a ground-up redesign that solved the extensibility problem structurally.

- **Key-value metadata with typed values** — every field has a string key, an explicit type tag, and a value. No position dependence; readers skip unknown keys gracefully.
- **Architecture stored as metadata** via `general.architecture` (e.g., `"llama"`, `"qwen2"`, `"gpt2"`). The file is self-describing; no out-of-band knowledge of the architecture is required.
- **Tokenizer fully embedded** — vocabulary tokens, scores, BPE merge rules, and special token IDs (`bos_token_id`, `eos_token_id`, chat template strings) all live in the KV section.
- A **single file contains everything** needed to load and run inference: architecture, weights, quant parameters, tokenizer.
- **Backwards-compatible extensibility** — new metadata keys are silently ignored by older readers. Format version bumps are not required for additive changes.
- Supports **50+ model architectures** as of 2025 (LLaMA, Mistral, Qwen, Phi, Gemma, Falcon, GPT-2, BERT, and many others).
- Replaced all predecessor formats entirely; `ggml`, `ggmf`, and `ggjt` are no longer accepted by current `llama.cpp` builds.

```
Sep 2022    Georgi Gerganov starts ggml tensor library
Mar 2023    llama.cpp released, uses GGML/GGJT formats
Mar 2023    K-quants introduced (Q4_K, Q5_K, Q6_K) — requires GGJT v3
Aug 2023    GGUF v3 released — extensible KV metadata, single-file models
2024-25     GGUF becomes de facto standard for local inference (50+ architectures)
```
