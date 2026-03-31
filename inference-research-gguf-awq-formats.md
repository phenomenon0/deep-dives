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

---

## 6. SafeTensors Format

The **SafeTensors** format is a simple, memory-mappable container for tensor data developed by HuggingFace. It is the default weight format for standard HF model repos, AWQ, and GPTQ checkpoints.

### Binary Layout

The file has three contiguous regions:

```
Offset    Field                Size         Notes
──────────────────────────────────────────────────────────────
0x00      Header length        8 bytes      uint64 LE — size N of JSON header
0x08      JSON header          N bytes      UTF-8 JSON, must start with '{'
0x08+N    Tensor data buffer   remaining    Contiguous binary, LE, row-major ('C' order)
```

### JSON Header Structure

```json
{
  "tensor_name": {
    "dtype": "F16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]
  },
  "__metadata__": {
    "format": "pt"
  }
}
```

`data_offsets` is `[BEGIN, END]` relative to the start of the **data buffer**, not the file. Tensor byte size = `END - BEGIN`. The `__metadata__` key is the only reserved key; all other keys are tensor names.

### Dtype Support

| Dtype | Width | Notes |
|-------|-------|-------|
| `F16` | 2 bytes | Half-precision float |
| `BF16` | 2 bytes | Brain float |
| `F32` | 4 bytes | Single-precision float |
| `F64` | 8 bytes | Double-precision float |
| `I8` / `U8` | 1 byte | Signed / unsigned int |
| `I16` / `I32` / `I64` | 2–8 bytes | Signed integers |
| `BOOL` | 1 byte | Boolean |

### Constraints and Safety Properties

- **Max header size**: 100 MB — hard cap to prevent DoS via malformed headers.
- **No overlapping regions**: tensor `data_offsets` ranges must not intersect.
- **No duplicate keys** in the JSON header.
- **Empty tensors** (shape containing a `0` dimension) are valid.
- Header may contain trailing `0x20` (space) padding bytes — parsers must tolerate this. No null terminators.
- All numeric data is **little-endian**, **row-major** order throughout.

### Alignment Gap vs. GGUF

SafeTensors provides **no alignment guarantees** — tensors begin at whatever byte offset follows the previous tensor. This contrasts with GGUF's 32-byte-aligned tensor data. The consequence: **CUDA GDS (GPU Direct Storage)** requires 4 KB-aligned reads; SafeTensors files cannot be used with GDS without an intermediate copy or padding layer.

### Multi-File and Config Split

SafeTensors deliberately stores **only tensors** — no architecture metadata, no tokenizer. Everything else lives in sidecar files:

| File | Contents |
|------|----------|
| `config.json` | Architecture params, `quantization_config` |
| `tokenizer.json` | Vocabulary, merge rules |
| `tokenizer_config.json` | Special tokens, chat template |
| `model.safetensors.index.json` | Shard map: tensor name → shard filename |

Large models are split across shards named `model-00001-of-00003.safetensors`. The index JSON maps each tensor name to its shard file, enabling **random-access loading** of individual tensors without reading the full checkpoint.

---

## 7. AWQ Model Format

**AWQ (Activation-aware Weight Quantization)** is a 4-bit post-training quantization scheme. On disk it is a standard HuggingFace directory — SafeTensors weights plus JSON config files — with quantization metadata embedded in `config.json`.

### Directory Layout

```
model-awq/
├── config.json                 # Architecture + quantization_config
├── model.safetensors           # Quantized weights (or sharded)
├── tokenizer.json              # Vocabulary and tokenizer config
├── tokenizer_config.json       # Special tokens and chat template
├── generation_config.json      # Default generation parameters
└── special_tokens_map.json     # Special token mappings
```

### `quantization_config` Block (inside `config.json`)

```json
{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 128,
  "zero_point": true,
  "version": "gemm"
}
```

| Field | Meaning |
|-------|---------|
| `bits` | Always `4` for AWQ |
| `group_size` | Input channels sharing one scale/zero — typically `128` |
| `zero_point` | `true` = asymmetric quant (scale + zero); `false` = symmetric (rare) |
| `version` | Kernel variant: `"gemm"`, `"gemv"`, or `"marlin"` |

### Three Tensors Per Quantized Linear Layer

Each quantized linear layer replaces its `weight` with three tensors. Using `model.layers.N.mlp.down_proj` as an example:

**`qweight`** (INT32) — packed 4-bit integer weights, 8 nibbles per int32.
- Shape: `[in_features, out_features / 8]`
- Packing uses `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` interleave for CUDA unpack efficiency.
- Bit extraction: `(qweight >> shift) & 0xF` with shifts `[0, 4, 8, 12, 16, 20, 24, 28]`.

**`qzeros`** (INT32) — packed 4-bit zero points, one per group.
- Shape: `[in_features / group_size, out_features / 8]`
- Identical packing scheme as `qweight`.

**`scales`** (FP16) — per-group scale factors, **not packed**.
- Shape: `[in_features / group_size, out_features]`
- Stored as full FP16 values; one scale per `(group, output_channel)` pair.

### Shape Example: LLaMA-2-7B `down_proj`

Original weight: `[11008, 4096]` FP16 = **86 MB**

| Tensor | Shape | Dtype | Size |
|--------|-------|-------|------|
| `qweight` | `[11008, 512]` | INT32 | 22.5 MB |
| `qzeros` | `[86, 512]` | INT32 | 176 KB |
| `scales` | `[86, 4096]` | FP16 | 704 KB |
| **Total** | | | **~23.4 MB** |

**3.7× compression** on this layer (86 MB → 23.4 MB).

### Dequantization Pipeline

```
1. Unpack:   int4_weight = (qweight >> shift) & 0xF
2. Unpack:   int4_zero   = (qzeros  >> shift) & 0xF
3. Dequant:  weight_fp16 = (int4_weight - int4_zero) × scale
```

Scale and zero point are applied per **group** — every `group_size` consecutive input channels share the same `(scale, zero)` pair. Smaller `group_size` improves accuracy at the cost of higher `scales`/`qzeros` overhead.

### Layers That Remain in FP16

| Layer | Reason |
|-------|--------|
| `model.embed_tokens.weight` | Token embedding — quality-critical, shared with `lm_head` |
| `lm_head.weight` | Output logit projection (often weight-tied to embeddings) |
| `model.layers.*.input_layernorm.weight` | RMSNorm scale — tiny, quantizing saves negligible space |
| `model.layers.*.post_attention_layernorm.weight` | Same |
| `model.norm.weight` | Final RMSNorm |

Layer norms are single vectors (e.g., 4096 values × 2 bytes = 8 KB per layer). Quantizing them yields no meaningful size reduction while introducing disproportionate quality degradation.

### Kernel Variants

| `version` | Use Case | Weight Layout | Performance |
|-----------|----------|---------------|-------------|
| `gemm` | Batched prefill / prompt processing | Standard AWQ packing | Efficient for batch > 1 |
| `gemv` | Single-token decode | Same packing, different kernel path | Optimized for batch = 1 |
| `marlin` | Both | Restructured tile layout for Marlin kernel | 2–4× faster than `gemm` on A100/H100 |

**Marlin AWQ** physically rearranges the packed weight data in the SafeTensors file into a tile layout that maximizes tensor core utilization — the quantized values are identical, but their memory layout is restructured. A model saved with `"version": "marlin"` is not directly interchangeable with `"gemm"` without re-packing the `qweight` and `qzeros` tensors.

---

## 8. LLaMA — What It Looks Like on Disk

LLaMA 3.1 8B is the reference architecture for modern decoder-only transformers. Everything below is measured from actual files.

### In GGUF

Key metadata from the header:

```
general.architecture           = "llama"
llama.block_count              = 32
llama.attention.head_count     = 32
llama.attention.head_count_kv  = 8        ← GQA: 4:1 ratio
llama.embedding_length         = 4096
llama.feed_forward_length      = 14336
llama.rope.freq_base           = 500000.0
tokenizer.ggml.model           = "llama"  ← SentencePiece BPE
```

Tensor manifest — one full layer block plus globals:

```
token_embd.weight                    [128256, 4096]   vocab × embed
blk.0.attn_norm.weight               [4096]           RMSNorm
blk.0.attn_q.weight                  [4096, 4096]     Q projection
blk.0.attn_k.weight                  [1024, 4096]     K projection (GQA: 8 heads × 128 dim)
blk.0.attn_v.weight                  [1024, 4096]     V projection (GQA)
blk.0.attn_output.weight             [4096, 4096]     output projection
blk.0.ffn_norm.weight                [4096]           RMSNorm
blk.0.ffn_gate.weight                [14336, 4096]    SwiGLU gate
blk.0.ffn_up.weight                  [14336, 4096]    SwiGLU up
blk.0.ffn_down.weight                [4096, 14336]    SwiGLU down
output_norm.weight                   [4096]           final RMSNorm
output.weight                        [128256, 4096]   LM head
```

Total: 2 global embed/output + 2 global norms + 10 tensors × 32 layers = **324 tensors**

Key structural features:

- **GQA visible in tensor shapes**: K and V projections are `[1024, 4096]` not `[4096, 4096]` — 8 KV heads × 128 dim = 1024. The asymmetry is directly measurable.
- **SwiGLU FFN**: Three projections (`ffn_gate`, `ffn_up`, `ffn_down`) instead of two. The gate projection enables the gated activation; its absence identifies non-LLaMA architectures immediately.
- **Separate output head**: `output.weight` is not tied to `token_embd.weight` — two distinct tensors with identical shape `[128256, 4096]`.

### In AWQ / SafeTensors

Same model, AWQ-quantized. Each linear weight becomes three tensors:

```
model.embed_tokens.weight                              [128256, 4096]   F16  not quantized
model.layers.0.input_layernorm.weight                  [4096]           F16
model.layers.0.self_attn.q_proj.qweight                [4096, 512]      I32  packed INT4
model.layers.0.self_attn.q_proj.qzeros                 [32, 512]        I32
model.layers.0.self_attn.q_proj.scales                 [32, 4096]       F16
model.layers.0.self_attn.k_proj.qweight                [4096, 128]      I32  smaller (GQA)
model.layers.0.self_attn.k_proj.qzeros                 [32, 128]        I32
model.layers.0.self_attn.k_proj.scales                 [32, 1024]       F16
model.layers.0.self_attn.v_proj.qweight                [4096, 128]      I32
model.layers.0.self_attn.v_proj.qzeros                 [32, 128]        I32
model.layers.0.self_attn.v_proj.scales                 [32, 1024]       F16
model.layers.0.self_attn.o_proj.qweight                [4096, 512]      I32
model.layers.0.self_attn.o_proj.qzeros                 [32, 512]        I32
model.layers.0.self_attn.o_proj.scales                 [32, 4096]       F16
model.layers.0.mlp.gate_proj.qweight                   [4096, 1792]     I32
model.layers.0.mlp.gate_proj.qzeros                    [32, 1792]       I32
model.layers.0.mlp.gate_proj.scales                    [32, 14336]      F16
model.layers.0.mlp.up_proj.qweight                     [4096, 1792]     I32
model.layers.0.mlp.up_proj.qzeros                      [32, 1792]       I32
model.layers.0.mlp.up_proj.scales                      [32, 14336]      F16
model.layers.0.mlp.down_proj.qweight                   [14336, 512]     I32
model.layers.0.mlp.down_proj.qzeros                    [32, 512]        I32
model.layers.0.mlp.down_proj.scales                    [32, 4096]       F16
model.layers.0.post_attention_layernorm.weight          [4096]           F16
model.norm.weight                                      [4096]           F16
lm_head.weight                                         [128256, 4096]   F16  not quantized
```

What changes in AWQ:

- **Naming convention**: HuggingFace-style `model.layers.N.self_attn.q_proj` replaces GGUF's `blk.N.attn_q`
- **Triple tensor expansion**: Every quantized weight spawns `qweight` (packed INT4), `qzeros` (zero points), and `scales` (per-group FP16)
- **Unquantized boundaries**: Embeddings, layer norms, and `lm_head` stay FP16 throughout
- **Tensor count**: 32 layers × (4 attn projections × 3 + 3 FFN projections × 3 + 2 norms) + 3 global = **707 tensors** vs 324 in GGUF

---

## 9. Qwen2 — How It Differs

### In GGUF

```
general.architecture           = "qwen2"
qwen2.block_count              = 28
qwen2.attention.head_count     = 28
qwen2.attention.head_count_kv  = 4          ← More aggressive GQA: 7:1 ratio
qwen2.embedding_length         = 3584
qwen2.feed_forward_length      = 18944
qwen2.rope.freq_base           = 1000000.0  ← 2× higher than LLaMA 3.1
tokenizer.ggml.model           = "gpt2"     ← BPE, not SentencePiece
tokenizer.ggml.pre             = "qwen2"
```

Tensor manifest for one layer:

```
blk.0.attn_norm.weight               [3584]
blk.0.attn_q.weight                  [3584, 3584]
blk.0.attn_q.bias                    [3584]          Qwen2-specific attention bias
blk.0.attn_k.weight                  [512, 3584]     4 KV heads × 128 dim
blk.0.attn_k.bias                    [512]
blk.0.attn_v.weight                  [512, 3584]
blk.0.attn_v.bias                    [512]
blk.0.attn_output.weight             [3584, 3584]
blk.0.ffn_norm.weight                [3584]
blk.0.ffn_gate.weight                [18944, 3584]
blk.0.ffn_up.weight                  [18944, 3584]
blk.0.ffn_down.weight                [3584, 18944]
```

Differences from LLaMA 3.1:

| Property | LLaMA 3.1 8B | Qwen2 7B |
|---|---|---|
| Tokenizer type | SentencePiece BPE | GPT-2 BPE (`merges` array) |
| Vocabulary size | 128,256 | 152,064 |
| GQA ratio | 4:1 (32Q / 8KV) | 7:1 (28Q / 4KV) |
| KV projection shape | `[1024, 4096]` | `[512, 3584]` |
| FFN ratio | 3.5:1 | 5.3:1 |
| RoPE base | 500,000 | 1,000,000 |
| Attention bias | None | Q/K/V all have `.bias` |
| Output weight | Independent | May be tied to `token_embd` |

The **attention bias tensors** (`blk.0.attn_q.bias`, etc.) are the fastest way to distinguish Qwen2 from LLaMA in a tensor manifest — LLaMA has no such tensors at all.

### In AWQ / SafeTensors

Follows the same three-tensor quantization pattern as LLaMA, with these differences:

- **Bias tensors unquantized**: `model.layers.0.self_attn.q_proj.bias` appears as a standalone FP16 tensor alongside the `qweight`/`qzeros`/`scales` triple
- **Narrower KV projections**: `k_proj.qweight` is `[3584, 64]` (packed) vs LLaMA's `[4096, 128]`
- **Larger embedding matrix**: `model.embed_tokens.weight` is `[152064, 3584]` — about 200MB at FP16
- **Possible absent `lm_head`**: When output weights are tied, `lm_head.weight` is omitted; the loader resolves the reference back to `embed_tokens.weight` at runtime

---

## 10. GPT-NeoX / GPT-J / GPT-2 — Fused Attention and Two-Layer FFN

These architectures share GPT-2 lineage and differ from LLaMA/Qwen2 in two structural ways visible directly in tensor shapes: **fused QKV** and **two-projection FFN**.

### GPT-NeoX in GGUF

```
general.architecture             = "gptneox"
gptneox.block_count              = 44
gptneox.attention.head_count     = 64
gptneox.attention.head_count_kv  = 64        ← Full MHA, no GQA
gptneox.embedding_length         = 6144
gptneox.feed_forward_length      = 24576
tokenizer.ggml.model             = "gpt2"
```

Tensor manifest for one layer:

```
blk.0.attn_norm.weight               [6144]
blk.0.attn_norm.bias                 [6144]          LayerNorm bias (absent in LLaMA)
blk.0.attn_qkv.weight                [18432, 6144]   FUSED Q+K+V: 3 × 6144 = 18432
blk.0.attn_qkv.bias                  [18432]
blk.0.attn_output.weight             [6144, 6144]
blk.0.attn_output.bias               [6144]
blk.0.ffn_norm.weight                [6144]
blk.0.ffn_norm.bias                  [6144]
blk.0.ffn_up.weight                  [24576, 6144]   up only — no gate projection
blk.0.ffn_up.bias                    [24576]
blk.0.ffn_down.weight                [6144, 24576]
blk.0.ffn_down.bias                  [6144]
```

### GPT-2 in GGUF

```
general.architecture          = "gpt2"
gpt2.block_count              = 12
gpt2.attention.head_count     = 12
gpt2.embedding_length         = 768
gpt2.feed_forward_length      = 3072
tokenizer.ggml.model          = "gpt2"
```

```
token_embd.weight                    [50257, 768]
position_embd.weight                 [1024, 768]    LEARNED positional embeddings — no RoPE
blk.0.attn_norm.weight               [768]
blk.0.attn_norm.bias                 [768]
blk.0.attn_qkv.weight                [2304, 768]    fused QKV: 3 × 768
blk.0.attn_qkv.bias                  [2304]
blk.0.attn_output.weight             [768, 768]
blk.0.attn_output.bias               [768]
blk.0.ffn_norm.weight                [768]
blk.0.ffn_norm.bias                  [768]
blk.0.ffn_up.weight                  [3072, 768]
blk.0.ffn_up.bias                    [3072]
blk.0.ffn_down.weight                [768, 3072]
blk.0.ffn_down.bias                  [768]
output.weight                        [50257, 768]   tied to token_embd in original
```

### Structural Comparison Across Lineages

| Feature | LLaMA 3.1 | Qwen2 | GPT-NeoX | GPT-2 |
|---|---|---|---|---|
| Attention tensor layout | Separate Q, K, V | Separate Q, K, V | Fused `attn_qkv` | Fused `attn_qkv` |
| GQA | Yes (4:1) | Yes (7:1) | No (MHA) | No (MHA) |
| FFN projections | gate + up + down | gate + up + down | up + down | up + down |
| Activation | SwiGLU | SwiGLU | GELU | GELU |
| Normalization | RMSNorm (weight only) | RMSNorm (weight only) | LayerNorm (weight + bias) | LayerNorm (weight + bias) |
| Attention bias | None | Q/K/V | All projections | All projections |
| Positional encoding | RoPE | RoPE | RoPE | Learned absolute |
| `position_embd` tensor | Absent | Absent | Absent | Present |

The **fused QKV** is the defining on-disk signature of the GPT-2 lineage. Where LLaMA shows three separate tensors `attn_q`, `attn_k`, `attn_v`, GPT-NeoX and GPT-2 show a single `attn_qkv.weight` whose first dimension is exactly 3× the embedding size. Any tool parsing GGUF files can identify the architecture family from this shape alone without reading the `general.architecture` key.

### In AWQ / SafeTensors (GPT-NeoX)

The fused QKV carries through into quantized form:

```
gpt_neox.layers.0.input_layernorm.weight                           [6144]         F16
gpt_neox.layers.0.input_layernorm.bias                             [6144]         F16
gpt_neox.layers.0.attention.query_key_value.qweight                [6144, 2304]   I32
gpt_neox.layers.0.attention.query_key_value.qzeros                 [48, 2304]     I32
gpt_neox.layers.0.attention.query_key_value.scales                 [48, 18432]    F16
gpt_neox.layers.0.attention.dense.qweight                          [6144, 768]    I32
gpt_neox.layers.0.attention.dense.qzeros                           [48, 768]      I32
gpt_neox.layers.0.attention.dense.scales                           [48, 6144]     F16
gpt_neox.layers.0.mlp.dense_h_to_4h.qweight                       [6144, 3072]   I32
gpt_neox.layers.0.mlp.dense_h_to_4h.qzeros                        [48, 3072]     I32
gpt_neox.layers.0.mlp.dense_h_to_4h.scales                        [48, 24576]    F16
gpt_neox.layers.0.mlp.dense_4h_to_h.qweight                       [24576, 768]   I32
gpt_neox.layers.0.mlp.dense_4h_to_h.qzeros                        [48, 768]      I32
gpt_neox.layers.0.mlp.dense_4h_to_h.scales                        [48, 6144]     F16
```

The naming convention diverges sharply from LLaMA's AWQ layout: `gpt_neox.layers.N.attention.query_key_value` vs `model.layers.N.self_attn.q_proj`. The GGUF converter normalizes both to `blk.N.attn_qkv`, erasing the HuggingFace namespace differences. When writing loaders that consume SafeTensors directly, the namespace prefix (`model.`, `gpt_neox.`, `transformer.`) must be detected and handled per-architecture — it is not standardized across model families.

---

## 11. Full Architecture Comparison

The table below consolidates key structural parameters across four representative model families, covering both GGUF metadata fields and AWQ packaging characteristics.

| Feature | LLaMA 3.1 8B | Qwen2 7B | GPT-NeoX 20B | GPT-2 124M |
|---------|-------------|----------|--------------|------------|
| GGUF `general.architecture` | `llama` | `qwen2` | `gptneox` | `gpt2` |
| Parameters | 8B | 7B | 20B | 124M |
| Layers | 32 | 28 | 44 | 12 |
| Hidden dim | 4096 | 3584 | 6144 | 768 |
| FFN dim | 14336 | 18944 | 24576 | 3072 |
| Attention heads (Q) | 32 | 28 | 64 | 12 |
| KV heads | 8 (GQA) | 4 (GQA) | 64 (MHA) | 12 (MHA) |
| Head dim | 128 | 128 | 96 | 64 |
| Vocab size | 128,256 | 152,064 | 50,432 | 50,257 |
| Tokenizer | SentencePiece | BPE | BPE | BPE |
| Position encoding | RoPE | RoPE | RoPE | Learned |
| Norm type | RMSNorm | RMSNorm | LayerNorm | LayerNorm |
| FFN type | SwiGLU (3 proj) | SwiGLU (3 proj) | GELU (2 proj) | GELU (2 proj) |
| Attention bias | No | Yes (QKV) | Yes (all) | Yes (all) |
| Tied embeddings | No | Sometimes | No | Yes |
| GGUF tensors (total) | ~324 | ~370 | ~534 | ~148 |
| GGUF Q4_K_M size | ~4.9 GB | ~4.4 GB | ~11.4 GB | ~80 MB |
| AWQ tensors (total) | ~707 | ~800+ | ~1100+ | N/A |
| AWQ model size | ~4.2 GB | ~3.7 GB | ~10.1 GB | N/A |

**Tensor count differences** are driven by architectural bias coverage and quantization packaging overhead:

- **Qwen2** carries 3 extra bias vectors per layer (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`), pushing its GGUF tensor count ~46 above LLaMA's despite having fewer layers.
- **GPT-NeoX** has bias on every projection — both attention and FFN — inflating tensor count to ~534 across 44 layers.
- **AWQ roughly triples the GGUF tensor count** because each quantized weight matrix spawns three tensors: the packed `qweight`, the scale `scales`, and the zero-point `qzeros`. Unquantized tensors (norms, embeddings) remain as-is, so the multiplier is sub-3× in practice.
- **GPT-2 124M is not a practical AWQ target**: the quantization metadata overhead (`scales`, `qzeros` per group) is significant relative to the tiny weight matrices, and the absolute memory savings are measured in tens of MB rather than GB.

### File Size Comparison for 7B-Class Models

| Format | LLaMA 3.1 8B | Qwen2 7B |
|--------|-------------|----------|
| FP16 SafeTensors | ~16 GB | ~14 GB |
| AWQ INT4 | ~4.2 GB | ~3.7 GB |
| GGUF Q4_K_M | ~4.9 GB | ~4.4 GB |
| GGUF Q8_0 | ~8.5 GB | ~7.5 GB |
| GGUF Q2_K | ~2.8 GB | ~2.5 GB |

**GGUF Q4_K_M runs ~700 MB larger than AWQ INT4** for equivalent models for two compounding reasons. First, the **mixed-precision strategy** in Q4_K_M promotes certain sensitive tensors — attention norms, the output projection, and roughly 20% of FFN layers — to `Q6_K`, which is 6 bits rather than 4. Second, GGUF **embeds the full tokenizer** (vocabulary, merge rules, special token metadata) directly in the file header, adding 1–4 MB depending on vocab size. AWQ files carry no tokenizer; the tokenizer lives separately in `tokenizer.json` alongside the model shards.

---

## 12. GGUF Tensor Name Mapping

The GGUF converter (`convert_hf_to_gguf.py` in llama.cpp) normalizes all architecture-specific HuggingFace tensor names into a **standardized flat namespace**. The authoritative pattern registry is `gguf-py/gguf/tensor_mapping.py`, which maps 80+ source name variants per logical tensor to handle HF-format checkpoints, original Meta/EleutherAI releases, and community fine-tunes. `{bid}` denotes the block (layer) index, zero-based.

### Standard GGUF Tensor Names

| GGUF Name | Role |
|-----------|------|
| `token_embd.weight` | Token embedding matrix |
| `position_embd.weight` | Positional embedding (GPT-2 only) |
| `blk.{bid}.attn_norm.weight` | Pre-attention layer norm |
| `blk.{bid}.attn_q.weight` | Query projection |
| `blk.{bid}.attn_k.weight` | Key projection |
| `blk.{bid}.attn_v.weight` | Value projection |
| `blk.{bid}.attn_qkv.weight` | Fused QKV (GPT-NeoX, GPT-2) |
| `blk.{bid}.attn_output.weight` | Attention output projection |
| `blk.{bid}.ffn_norm.weight` | Pre-FFN layer norm |
| `blk.{bid}.ffn_gate.weight` | SwiGLU gate (LLaMA, Qwen2, Mistral) |
| `blk.{bid}.ffn_up.weight` | FFN up/expansion projection |
| `blk.{bid}.ffn_down.weight` | FFN down/contraction projection |
| `output_norm.weight` | Final layer norm |
| `output.weight` | LM head / output projection |

Architectures that **fuse QKV** into a single checkpoint tensor (GPT-NeoX, GPT-2) map to `attn_qkv`; the converter splits it into `attn_q` / `attn_k` / `attn_v` only when the source stores them separately. Architectures with **tied embeddings** (GPT-2) omit `output.weight` entirely — at inference time llama.cpp reuses `token_embd.weight` transposed.

### Source Name → GGUF Name Mapping

**Embeddings & Output:**

| GGUF Name | LLaMA (HF) | Qwen2 (HF) | GPT-NeoX | GPT-2 | Falcon |
|-----------|-----------|------------|----------|-------|--------|
| `token_embd` | `model.embed_tokens` | `model.embed_tokens` | `gpt_neox.embed_in` | `transformer.wte` | `transformer.word_embeddings` |
| `position_embd` | — | — | — | `transformer.wpe` | — |
| `output` | `lm_head` | `lm_head` | `embed_out` | `lm_head` *(tied)* | `lm_head` |
| `output_norm` | `model.norm` | `model.norm` | `gpt_neox.final_layer_norm` | `transformer.ln_f` | `transformer.ln_f` |

**Attention:**

| GGUF Name | LLaMA (HF) | Qwen2 (HF) | GPT-NeoX | GPT-2 / GPT-J |
|-----------|-----------|------------|----------|---------------|
| `blk.{bid}.attn_norm` | `model.layers.{bid}.input_layernorm` | `model.layers.{bid}.input_layernorm` | `gpt_neox.layers.{bid}.input_layernorm` | `transformer.h.{bid}.ln_1` |
| `blk.{bid}.attn_q` | `.self_attn.q_proj` | `.self_attn.q_proj` | — *(fused)* | — *(fused)* |
| `blk.{bid}.attn_k` | `.self_attn.k_proj` | `.self_attn.k_proj` | — *(fused)* | — *(fused)* |
| `blk.{bid}.attn_v` | `.self_attn.v_proj` | `.self_attn.v_proj` | — *(fused)* | — *(fused)* |
| `blk.{bid}.attn_qkv` | — | — | `.attention.query_key_value` | `.attn.c_attn` |
| `blk.{bid}.attn_output` | `.self_attn.o_proj` | `.self_attn.o_proj` | `.attention.dense` | `.attn.c_proj` |

**FFN:**

| GGUF Name | LLaMA (HF) | Qwen (original) | GPT-NeoX | GPT-2 |
|-----------|-----------|-----------------|----------|-------|
| `blk.{bid}.ffn_norm` | `.post_attention_layernorm` | `.post_attention_layernorm` | `.post_attention_layernorm` | `transformer.h.{bid}.ln_2` |
| `blk.{bid}.ffn_gate` | `.mlp.gate_proj` | `.mlp.w2` | — *(no gate)* | — *(no gate)* |
| `blk.{bid}.ffn_up` | `.mlp.up_proj` | `.mlp.w1` | `.mlp.dense_h_to_4h` | `.mlp.c_fc` |
| `blk.{bid}.ffn_down` | `.mlp.down_proj` | `.mlp.c_proj` | `.mlp.dense_4h_to_h` | `.mlp.c_proj` |

**Watch out for Qwen vs Qwen2 naming divergence.** The original Qwen checkpoints use a `transformer.h.{bid}` path inherited from GPT-style naming; Qwen2 switched to the LLaMA-style `model.layers.{bid}` convention. `tensor_mapping.py` handles both patterns but they resolve to different branch paths in the converter — a common source of confusion when loading Qwen v1 weights with a Qwen2 architecture class.

**Original (non-HF) Meta checkpoints** use a further different scheme: `tok_embeddings.weight` instead of `model.embed_tokens`, `layers.{bid}.attention.wq` instead of `model.layers.{bid}.self_attn.q_proj`, and `norm.weight` instead of `model.norm`. `tensor_mapping.py` covers all three variants (Meta original, HF-converted, and GGUF canonical) to make the converter robust against whichever checkpoint format is provided.

---

## 13. Practical: Inspecting Files

### Reading GGUF Metadata

Use the `gguf` Python package (`pip install gguf`) to inspect metadata and tensor layout without loading weights into memory:

```python
from gguf import GGUFReader

reader = GGUFReader("model.gguf")

# Print all metadata
for key in reader.fields:
    field = reader.fields[key]
    if len(field.parts) > 0:
        print(f"{key} = {field.parts[-1].tolist() if hasattr(field.parts[-1], 'tolist') else field.parts[-1]}")

# Key metadata
arch = reader.fields['general.architecture'].parts[-1].tobytes().decode()
print(f"Architecture: {arch}")
print(f"Layers: {reader.fields[f'{arch}.block_count'].parts[-1][0]}")

# List all tensors
for tensor in reader.tensors:
    print(f"{tensor.name}: shape={tensor.shape}, type={tensor.tensor_type.name}")
```

Example output for LLaMA-3.1-8B-Instruct Q4_K_M:

```
general.architecture = llama
general.name = Llama-3.1-8B-Instruct
llama.block_count = 32
llama.attention.head_count = 32
llama.attention.head_count_kv = 8
llama.context_length = 131072
llama.embedding_length = 4096
llama.feed_forward_length = 14336
...
token_embd.weight: shape=(128256, 4096), type=Q6_K
blk.0.attn_q.weight: shape=(4096, 4096), type=Q4_K
blk.0.attn_k.weight: shape=(1024, 4096), type=Q4_K
blk.0.attn_v.weight: shape=(1024, 4096), type=Q4_K
blk.0.ffn_gate.weight: shape=(14336, 4096), type=Q4_K
...
```

### Reading AWQ / SafeTensors

AWQ metadata is split across two files: `model.safetensors` (or shards) for weights, and `config.json` for quantization parameters.

```python
from safetensors import safe_open
import json

# Inspect tensors without loading full weights
with safe_open("model.safetensors", framework="pt") as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        print(f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

# Or parse only the raw JSON header — zero weight I/O
with open("model.safetensors", "rb") as f:
    header_size = int.from_bytes(f.read(8), "little")
    header = json.loads(f.read(header_size))
    for name, info in header.items():
        if name != "__metadata__":
            print(f"{name}: dtype={info['dtype']}, shape={info['shape']}")

# Check quantization config
with open("config.json") as f:
    config = json.load(f)
    qconfig = config.get("quantization_config", {})
    print(f"Method: {qconfig.get('quant_method')}")   # "awq"
    print(f"Bits: {qconfig.get('bits')}")             # 4
    print(f"Group size: {qconfig.get('group_size')}") # 128
    print(f"Version: {qconfig.get('version')}")       # "gemm" or "marlin"
```

The raw header parse is useful for large sharded models where you want to audit layout without triggering any tensor allocation.

### Quick Diagnostics

`llama.cpp` ships CLI tools that require no Python:

- **`llama-gguf-dump model.gguf`** — prints the complete metadata key-value store and full tensor list with types and shapes
- **`llama-gguf-hash model.gguf`** — emits per-tensor and whole-file SHA256 hashes for integrity checks
- **`huggingface-cli scan-cache`** — lists locally cached HF model files with sizes and revisions

---

## 14. Common Gotchas

**GGUF dimension reversal.** GGUF stores tensor dimensions in the reverse order of PyTorch. A weight with PyTorch shape `[4096, 1024]` has GGUF dimensions `[1024, 4096]`. Any tool that compares shapes across formats must transpose before comparing, or it will report a false mismatch on every non-square tensor.

**Alignment math.** Tensor data offsets in GGUF must be exact multiples of `general.alignment` (default `32`). When constructing or patching GGUF files manually, use `offset = ((pos + alignment - 1) // alignment) * alignment` to compute the next valid boundary. Off-by-one alignment errors produce models that load silently but have garbage weights, or segfault on mmap access.

**Tied embeddings handling.** Some GGUF converters duplicate `token_embd.weight` as `output.weight`, doubling file size for that tensor. Others omit `output.weight` and rely on the loader to infer weight tying. In AWQ/SafeTensors, tied models may omit `lm_head.weight` entirely or include it as a zero-byte reference. Loaders must handle both patterns explicitly.

**AWQ nibble packing order.** AWQ uses `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` for nibble interleaving within each INT32 word. Unpacking with naive sequential order produces garbled weights that pass all shape checks but generate nonsense output. GPTQ uses a different interleaving — the two formats are not byte-compatible despite both being INT4 in SafeTensors.

**SafeTensors alignment and GPU Direct Storage.** SafeTensors provides no alignment guarantee on tensor data offsets. If the JSON header byte count is odd, tensor data begins at an odd byte offset. CUDA GPU Direct Storage (GDS) requires aligned reads; direct loading will fail with cryptic CUDA errors or silently read incorrect data.

**Tokenizer type mismatch on conversion.** SentencePiece tokenizers (LLaMA 2) embed vocabulary as `scores`; BPE tokenizers (LLaMA 3, Qwen2) embed vocabulary as `merges`. Swapping them produces shifted token IDs that appear functional but degrade output quality, particularly on code and non-English text. After any GGUF conversion, verify both `tokenizer.ggml.model` and `tokenizer.ggml.pre` match the source model.

**Missing `tokenizer.ggml.pre`.** Older GGUF files often lack the `tokenizer.ggml.pre` key. Runtimes that fall back to a generic pre-tokenizer will produce incorrect token boundaries for architectures requiring specific pre-tokenization regexes — notably LLaMA 3 (`"llama-bpe"`) and Qwen2 (`"qwen2"`). The failure mode is subtle: plausible but degraded output, especially on code and multilingual input.

**GPTQ vs AWQ kernel mismatch.** Both GPTQ and AWQ store INT4 weights in SafeTensors, but GPTQ uses different tensor names (`qweight`, `qzeros`, `scales`, `g_idx`) and a different packing layout. Loading an AWQ checkpoint with a GPTQ kernel (or vice versa) produces wrong results with no runtime error. Always check `quantization_config.quant_method` in `config.json` before selecting a kernel.

**Marlin layout incompatibility.** AWQ models with `"version": "marlin"` in `quantization_config` have physically rearranged weight data optimized for Marlin GEMM. Loading them with a standard GEMM kernel produces garbage output. The `version` field must match the inference kernel — `"gemm"` and `"marlin"` checkpoints are not interchangeable at the file level.

**GGUF version.** Version 3 is the only version produced by current tooling. Versions 1 and 2 existed briefly during early development. A file reporting `version != 3` is either from mid-2023 or corrupt; no current converter or loader targets those versions.
