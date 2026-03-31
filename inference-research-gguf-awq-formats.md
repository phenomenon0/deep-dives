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

---

## 3. GGUF Binary Layout

This section is a byte-level walkthrough of the GGUF format. Understanding the layout precisely is essential for writing parsers, debugging corrupt files, or auditing model metadata without loading the full weights.

### Header (fixed, 24 bytes)

The file begins with a fixed 24-byte header. All multi-byte fields are **little-endian** unless the magic number indicates otherwise.

```
Offset  Field                Size       Type         Value/Notes
──────────────────────────────────────────────────────────────────
0x00    Magic number         4 bytes    uint32       0x47475546 (ASCII "GGUF")
0x04    Version              4 bytes    uint32       3 (current spec)
0x08    Tensor count         8 bytes    uint64       Number of tensors in file
0x10    Metadata KV count    8 bytes    uint64       Number of key-value pairs
```

The **magic number** `0x47475546` decodes to the ASCII bytes `G`, `G`, `U`, `F` in memory order. Big-endian GGUF files exist — identifiable by magic `0x46554747` — but are uncommon in practice and not produced by mainstream tooling. Parsers should check the magic first and reject unrecognized values immediately rather than attempting to read further.

### Metadata KV Section

Immediately after the header, `metadata_kv_count` key-value pairs are encoded sequentially. Each pair has the structure:

```
gguf_string_t  key         // uint64 length + UTF-8 bytes (no null terminator)
uint32_t       value_type  // one of 13 type IDs
<value>                    // type-dependent encoding
```

**STRING encoding** (`gguf_string_t`): a `uint64` length prefix followed by exactly that many UTF-8 bytes. There is no null terminator. Key strings follow a `lower_snake_case` dotted hierarchy (e.g., `llama.attention.head_count`, `tokenizer.ggml.model`), capped at 65535 bytes in practice. Values of type `STRING` use the same encoding.

**ARRAY encoding**: a `uint32` element type tag, a `uint64` element count, then `count` elements of that type packed sequentially. Arrays are **homogeneous** — the element type tag applies uniformly to all elements. Nested arrays are not permitted.

The complete type system:

| ID | Type | Encoding |
|----|------|----------|
| 0 | `UINT8` | 1 byte unsigned |
| 1 | `INT8` | 1 byte signed |
| 2 | `UINT16` | 2 bytes LE |
| 3 | `INT16` | 2 bytes LE |
| 4 | `UINT32` | 4 bytes LE |
| 5 | `INT32` | 4 bytes LE |
| 6 | `FLOAT32` | 4 bytes IEEE 754 |
| 7 | `BOOL` | 1 byte (`0` = false, `1` = true) |
| 8 | `STRING` | `uint64` length + UTF-8 bytes |
| 9 | `ARRAY` | `uint32` element_type + `uint64` count + count × element |
| 10 | `UINT64` | 8 bytes LE |
| 11 | `INT64` | 8 bytes LE |
| 12 | `FLOAT64` | 8 bytes IEEE 754 |

KV pairs must be parsed strictly in sequence. There is no index or seek table — skipping a pair requires knowing its exact byte length, which depends on decoding the type and, for strings and arrays, reading their length fields.

### Tensor Info Section

After all metadata KV pairs, `tensor_count` tensor descriptors follow contiguously. Each descriptor encodes:

```
For each tensor (tensor_count times):
    gguf_string_t  name          // e.g. "blk.0.attn_q.weight"
    uint32_t       n_dimensions  // typically 1 or 2
    uint64_t       dimensions[n_dimensions]  // shape, reversed vs PyTorch convention
    uint32_t       type          // ggml_type enum (see §5)
    uint64_t       offset        // byte offset relative to start of tensor data region
```

**Dimension ordering** is a persistent source of confusion. GGUF stores dimensions in the reverse order of PyTorch's row-major convention. A weight that PyTorch represents as shape `[out_features, in_features]` = `[4096, 4096]` is stored with `dimensions = [4096, 4096]` — numerically identical in the square case, but axes are semantically swapped. For non-square tensors (e.g., `[4096, 1024]`), the GGUF `dimensions` array reads `[1024, 4096]`. Keep this in mind when reconstructing shapes for export or comparison.

The `offset` field is relative to the **start of the tensor data region** (after the alignment padding), not to the start of the file. Each offset is guaranteed to be a multiple of the active alignment value.

### Tensor Data Section

After the last tensor info entry, the file is padded with `0x00` bytes to the next multiple of the **alignment** value. Alignment defaults to **32 bytes** and can be overridden via the `general.alignment` metadata key. The tensor data region then begins at that aligned boundary and extends to the end of the file.

Each tensor's raw bytes start at `tensor_data_start + tensor.offset`. Every offset is itself a multiple of alignment, so each tensor can be `mmap`'d directly and accessed with correct cache-line alignment — no copy required. This is the primary motivation for the padding scheme.

Complete file layout:

```
┌─────────────────────────────────┐
│  Header (24 bytes)              │  magic + version + counts
├─────────────────────────────────┤
│  Metadata KV pairs              │  variable-length typed entries
│  (metadata_kv_count entries)    │
├─────────────────────────────────┤
│  Tensor Info array              │  name + dims + type + offset
│  (tensor_count entries)         │
├──── padding to alignment ───────┤
│  Tensor Data                    │  contiguous quantized weights
│  (each tensor at aligned        │
│   offset within this region)    │
└─────────────────────────────────┘
```

A minimal parser needs only four capabilities: read the 24-byte header, decode `gguf_string_t` values, dispatch on the 13 value type IDs, and apply the alignment padding formula `ceil(pos / alignment) × alignment` to locate the tensor data region. Everything else in the format builds on these primitives.

---

## 4. GGUF Metadata Keys

GGUF metadata is organized into namespaced key groups. Keys use dot-notation with a consistent prefix that signals which subsystem owns the value. Below is the complete taxonomy with real values drawn from production models.

### General Metadata (`general.*`)

These keys apply to every GGUF file regardless of architecture.

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `general.architecture` | string | `"llama"` | Architecture identifier — determines which arch-specific keys apply |
| `general.name` | string | `"Llama-3.1-8B-Instruct"` | Human-readable model name |
| `general.file_type` | uint32 | `15` | Quantization type enum (15 = Q4_K_M) |
| `general.quantization_version` | uint32 | `2` | Quant format version |
| `general.alignment` | uint32 | `32` | Tensor data alignment in bytes |
| `general.author` | string | `"Meta"` | Model author |
| `general.license` | string | `"llama3.1"` | License identifier |
| `general.source.huggingface.repository` | string | `"meta-llama/Llama-3.1-8B-Instruct"` | HF repo |

`general.architecture` is load-bearing: every arch-specific key below is prefixed with its value. If this key is missing or mismatched, the runtime cannot locate the correct structural parameters.

### Architecture-Specific Keys (`{arch}.*`)

`{arch}` is replaced by the value of `general.architecture` — so for LLaMA models it becomes `llama.*`, for Qwen2 it becomes `qwen2.*`, and so on. These keys define the model's structural dimensions.

| Key | Type | LLaMA 3.1 8B | Qwen2 7B | GPT-NeoX 20B |
|-----|------|--------------|----------|--------------|
| `{arch}.context_length` | uint32 | `131072` | `32768` | `2048` |
| `{arch}.embedding_length` | uint32 | `4096` | `3584` | `6144` |
| `{arch}.block_count` | uint32 | `32` | `28` | `44` |
| `{arch}.feed_forward_length` | uint32 | `14336` | `18944` | `24576` |
| `{arch}.attention.head_count` | uint32 | `32` | `28` | `64` |
| `{arch}.attention.head_count_kv` | uint32 | `8` | `4` | `64` |
| `{arch}.attention.layer_norm_rms_epsilon` | float32 | `1e-5` | `1e-6` | — |
| `{arch}.attention.layer_norm_epsilon` | float32 | — | — | `1e-5` |
| `{arch}.rope.freq_base` | float32 | `500000.0` | `1000000.0` | `10000.0` |
| `{arch}.rope.dimension_count` | uint32 | `128` | `128` | `24` |

**Key distinctions:**

- **Grouped Query Attention (GQA):** When `head_count_kv` < `head_count`, the model uses GQA. LLaMA 3.1 8B has 8 KV heads against 32 query heads — a 4:1 ratio that reduces KV cache size by 4×. When both values are equal (as in GPT-NeoX 20B), the model uses standard **Multi-Head Attention (MHA)**.

- **Normalization type:** `layer_norm_rms_epsilon` is used by architectures with **RMSNorm** (LLaMA, Qwen2, Mistral). `layer_norm_epsilon` is used by architectures with **LayerNorm** (GPT-NeoX, Falcon). A runtime can infer the normalization type from which key is present.

- **RoPE base frequency:** `rope.freq_base` varies significantly with intended context length. Older models targeting 2K–4K context use the original `10000.0`. LLaMA 3.1 bumps this to `500000.0` to support 128K context; Qwen2 uses `1000000.0` for similar reasons. Higher base frequencies slow the rate at which positional embeddings rotate, extending effective range.

### RoPE Scaling Keys

When a model has been fine-tuned for a longer context than its base `rope.freq_base` alone provides, these keys describe the scaling strategy applied.

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.rope.scaling.type` | string | Scaling strategy: `"linear"`, `"yarn"`, `"ntk-aware"` |
| `{arch}.rope.scaling.factor` | float32 | Multiplier applied to context extension (e.g., `8.0` for 8× extension) |
| `{arch}.rope.scaling.original_context_length` | uint32 | Pre-scaling base context length |

**YaRN** is the most common value for `scaling.type` in long-context LLaMA variants. Linear scaling is simpler but degrades perplexity at range. If `scaling.type` is absent, the runtime applies no RoPE correction beyond `freq_base`.

### Tokenizer Metadata (`tokenizer.*`)

These keys fully specify the tokenizer so inference can run without external vocabulary files.

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `tokenizer.ggml.model` | string | `"llama"` / `"gpt2"` | Tokenizer family — `"llama"` = SentencePiece, `"gpt2"` = BPE |
| `tokenizer.ggml.pre` | string | `"llama-bpe"` / `"qwen2"` | Pre-tokenization regex variant |
| `tokenizer.ggml.tokens` | string[] | `["<unk>", "<s>", "</s>", ...]` | Full vocabulary as an ordered string array |
| `tokenizer.ggml.scores` | float32[] | `[-1000.0, -1000.0, ...]` | Per-token log-probability scores (SentencePiece only) |
| `tokenizer.ggml.token_type` | uint32[] | `[3, 3, 3, 1, 1, ...]` | Token type: `1`=normal, `2`=unknown, `3`=control, `4`=user_defined, `5`=unused, `6`=byte |
| `tokenizer.ggml.merges` | string[] | `["Ġ t", "Ġ a", ...]` | BPE merge rules (BPE tokenizers only) |
| `tokenizer.ggml.bos_token_id` | uint32 | `128000` | Beginning-of-sequence token ID |
| `tokenizer.ggml.eos_token_id` | uint32 | `128001` | End-of-sequence token ID |
| `tokenizer.ggml.unknown_token_id` | uint32 | `0` | Unknown/OOV token ID |
| `tokenizer.ggml.padding_token_id` | uint32 | `128004` | Padding token ID |
| `tokenizer.ggml.add_bos_token` | bool | `true` | Auto-prepend BOS during tokenization |
| `tokenizer.ggml.add_eos_token` | bool | `false` | Auto-append EOS during tokenization |
| `tokenizer.chat_template` | string | *(Jinja2 template)* | Chat formatting template (applied by the runtime before tokenization) |

**Key distinctions by tokenizer family:**

- **LLaMA 2 / LLaMA 3 (SentencePiece):** `tokenizer.ggml.model = "llama"`. Has `scores` array (SentencePiece log-probs used during decoding). No `merges` key. LLaMA 3 additionally sets `tokenizer.ggml.pre = "llama-bpe"` to select the correct pre-tokenization regex.

- **Qwen2 (BPE):** `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "qwen2"`. Has `merges` array. No `scores` key. Despite the `"gpt2"` label, the merge rules and vocabulary are Qwen2-specific — the label refers to the BPE algorithm, not the GPT-2 vocabulary.

- **GPT-2 / GPT-NeoX (BPE):** Also `tokenizer.ggml.model = "gpt2"`, distinguished by vocabulary content and `pre` value rather than the model key itself.

The `tokenizer.ggml.pre` key was added after the original GGUF spec to handle divergent pre-tokenization behavior between models that share the same base algorithm. Runtimes that ignore it will produce incorrect token boundaries for affected models.

---

## 5. GGUF Tensor Types & Quantization Block Structures

### Type Enum Table

The `ggml_type` enum assigns a numeric ID to every tensor storage format. Block size is the number of weights per block, and bits/weight is the effective storage cost including all metadata (scales, offsets, codebook indices).

| ID | Name | Block Size | Bytes/Block | Bits/Weight | Notes |
|----|------|------------|-------------|-------------|-------|
| 0 | `F32` | 1 | 4 | 32.00 | Full precision |
| 1 | `F16` | 1 | 2 | 16.00 | Half precision |
| 2 | `Q4_0` | 32 | 18 | 4.50 | Legacy symmetric 4-bit |
| 3 | `Q4_1` | 32 | 20 | 5.00 | Legacy asymmetric 4-bit |
| 6 | `Q5_0` | 32 | 22 | 5.50 | Legacy symmetric 5-bit |
| 7 | `Q5_1` | 32 | 24 | 6.00 | Legacy asymmetric 5-bit |
| 8 | `Q8_0` | 32 | 34 | 8.50 | 8-bit, used for activation quant |
| 9 | `Q8_1` | 32 | 36 | 9.00 | 8-bit with offset |
| 10 | `Q2_K` | 256 | 84 | 2.625 | K-quant 2-bit |
| 11 | `Q3_K` | 256 | 110 | 3.4375 | K-quant 3-bit |
| 12 | `Q4_K` | 256 | 144 | 4.50 | K-quant 4-bit |
| 13 | `Q5_K` | 256 | 176 | 5.50 | K-quant 5-bit |
| 14 | `Q6_K` | 256 | 210 | 6.5625 | K-quant 6-bit |
| 15 | `Q8_K` | 256 | 292 | 9.125 | K-quant 8-bit |
| 16 | `IQ2_XXS` | 256 | 66 | 2.0625 | Information-theoretic 2-bit |
| 17 | `IQ2_XS` | 256 | 74 | 2.3125 | Slightly higher quality 2-bit |
| 18 | `IQ3_XXS` | 256 | 98 | 3.0625 | Information-theoretic 3-bit |
| 19 | `IQ1_S` | 256 | 50 | 1.5625 | Extreme 1-bit |
| 20 | `IQ4_NL` | 32 | 18 | 4.50 | Non-linear 4-bit |
| 21 | `IQ3_S` | 256 | 110 | 3.4375 | Information-theoretic 3-bit |
| 22 | `IQ2_S` | 256 | 82 | 2.5625 | 2-bit variant |
| 23 | `IQ4_XS` | 256 | 136 | 4.25 | 4-bit extended |
| 24–27 | `I8`/`I16`/`I32`/`I64` | 1 | 1/2/4/8 | 8–64 | Integer storage types, not for model weights |
| 28 | `F64` | 1 | 8 | 64.00 | Double precision |
| 29 | `IQ1_M` | 256 | 56 | 1.75 | 1-bit medium |
| 30 | `BF16` | 1 | 2 | 16.00 | Brain float 16 |
| 34–35 | `TQ1_0`/`TQ2_0` | — | — | ~1.69/2.06 | Ternary quants |

### Block Anatomy: Legacy Types

**Q4_0** — The simplest quantization block. One scale, symmetric around zero:

```c
struct block_q4_0 {           // 18 bytes total, 32 weights
    fp16_t  d;                // 2 bytes  — scale factor
    uint8_t qs[16];           // 16 bytes — 32 × 4-bit weights, 2 per byte
};
// Dequant: weight[i] = d × (nibble[i] - 8)
// Nibble range [0, 15] mapped to [-8, +7]. No offset term.
```

**Q4_1** — Adds a **minimum value** (`m`) for asymmetric distributions:

```c
struct block_q4_1 {           // 20 bytes total, 32 weights
    fp16_t  d;                // 2 bytes  — scale factor
    fp16_t  m;                // 2 bytes  — minimum (offset)
    uint8_t qs[16];           // 16 bytes — 32 × 4-bit weights
};
// Dequant: weight[i] = d × nibble[i] + m
// Better for distributions not centered at zero; costs 2 extra bytes/block.
```

**Q8_0** — Used internally for on-the-fly activation quantization in fused matmul kernels, not for model storage:

```c
struct block_q8_0 {           // 34 bytes total, 32 weights
    fp16_t d;                 // 2 bytes  — scale factor
    int8_t qs[32];            // 32 bytes — 32 × 8-bit weights
};
// Dequant: weight[i] = d × qs[i]
// Quantized activations pair with Q4_K/Q5_K weights in vecdot kernels.
```

### Block Anatomy: K-Quants (Super-Block Architecture)

K-quants use a **two-level hierarchy**: a 256-weight **super-block** containing 8 sub-blocks of 32 weights each. Per-sub-block scales are themselves quantized against super-block-level parameters — this **double quantization** cuts metadata overhead while preserving per-group adaptation. Codebooks are K-means optimized at quantization time.

**Q4_K** — The most widely deployed K-quant:

```c
struct block_q4_K {           // 144 bytes total, 256 weights
    fp16_t  d;                // 2 bytes   — super-block scale
    fp16_t  dmin;             // 2 bytes   — super-block minimum
    uint8_t scales[12];       // 12 bytes  — packed sub-block scales & mins
    uint8_t qs[128];          // 128 bytes — 256 × 4-bit weights
};
// Two-level dequant:
// 1. Recover sub-block scale/min from `scales` using d and dmin
// 2. weight[i] = sub_scale × nibble[i] + sub_min
```

**Q6_K** — Higher precision; 6-bit weights packed across two arrays:

```c
struct block_q6_K {           // 210 bytes total, 256 weights
    uint8_t ql[128];          // 128 bytes — lower 4 bits of each 6-bit weight
    uint8_t qh[64];           // 64 bytes  — upper 2 bits of each 6-bit weight
    int8_t  scales[16];       // 16 bytes  — INT8 sub-block scales
    fp16_t  d;                // 2 bytes   — super-block scale
};
// 6 bits split across ql and qh for byte-boundary packing efficiency.
// Sub-block scales are INT8, dequantized by multiplying by d.
```

**Q2_K** packs 2-bit weights with 4-bit scales and 4-bit mins into 84 bytes — viable only for very large models where memory pressure outweighs quality loss.

### Block Anatomy: IQ Types (Information-Theoretic)

**IQ types** replace uniform quantization grids with **learned codebooks** — reconstruction values are not evenly spaced but chosen to minimize expected squared error over a representative weight corpus. This yields meaningfully better quality at the same bit width, at the cost of more complex dequantization logic (table lookups vs. multiply-add).

- **`IQ2_XXS`** (2.06 bpw): 256-entry shared codebook. Two bits index 4 entries per lookup; a grid of higher-level indices handles the super-block. Practical floor for 70B+ models on 24 GB consumer GPUs.
- **`IQ1_S`** / **`IQ1_M`** (1.56–1.75 bpw): Single-bit-class extreme compression. Quality degrades sharply below 7B parameters; useful only for massive models where the alternative is CPU offload.
- **`IQ4_NL`** (4.5 bpw): 16 non-linearly spaced reconstruction values optimized for typical LLM weight histograms. Matches `Q4_K` byte-for-byte but improves perplexity by ~0.1–0.2 on standard benchmarks. The block layout is identical to `Q4_0` — only the dequantization lookup table differs.

### Quantization File Type Suffixes

GGUF filenames use suffixes that encode **mixed-precision strategies** — not every tensor gets the same quant type. Embedding layers and attention output projections are more sensitive to quantization error and receive higher-precision treatment.

| Suffix | Attention / Output | FFN | Embeddings | Effective bpw |
|--------|--------------------|-----|------------|---------------|
| `Q4_K_S` | `Q4_K` | `Q4_K` | `Q6_K` | ~4.4 |
| `Q4_K_M` | `Q4_K` | `Q6_K` (sensitive layers) | `Q6_K` | ~4.8 |
| `Q5_K_S` | `Q5_K` | `Q5_K` | `Q6_K` | ~5.4 |
| `Q5_K_M` | `Q5_K` | `Q6_K` (sensitive layers) | `Q6_K` | ~5.7 |
| `Q8_0` | `Q8_0` | `Q8_0` | `Q8_0` | ~8.5 |

The `_S` (small), `_M` (medium), and `_L` (large) variants control **how many tensors** receive the higher-precision upgrade. `_M` is the standard recommendation: it targets FFN down-projections and attention output matrices — the layers most sensitive to quantization noise — while keeping the rest at the base quant level. The per-bpw cost difference between `_S` and `_M` is typically 0.3–0.5 bits, with a perplexity improvement of 0.05–0.15 on Llama-class models.

For most deployment scenarios: `Q4_K_M` at ~4.8 bpw offers the best quality-per-gigabyte tradeoff; `Q5_K_M` is preferred when VRAM budget allows and inference quality is the priority.
