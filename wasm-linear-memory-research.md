# WebAssembly Linear Memory — Research Notes

Research compiled 2026-03-20 for deep-dive chapter.

---

## 1. Linear Memory Model

### The core abstraction

WebAssembly linear memory is a single contiguous array of raw bytes. The spec defines it as:

> "A memory instance is a record containing a memory type and a sequence of bytes."
> — WebAssembly 3.0 Spec, Runtime Structure (webassembly.github.io/spec/core/exec/runtime.html)

Every pointer in compiled C/Rust/Go code = an index (i32 or i64) into this flat byte array. There is no concept of separate heap/stack/data segments at the Wasm level — that's all a toolchain concern layered on top.

### Page size

The spec defines page size as a constant:

> "The WebAssembly page size is defined to be the constant 65536 — abbreviated 64 Ki."
> — WebAssembly Spec, Execution: Runtime (webassembly.github.io/spec/core/exec/runtime.html)

V8 confirms in source:
```cpp
constexpr size_t kWasmPageSize = 0x10000;  // 65,536 bytes
constexpr uint32_t kWasmPageSizeLog2 = 16;
```
— Source: `v8/src/wasm/wasm-constants.h` (chromium.googlesource.com)

Memory size must always be a multiple of the page size.

### Memory type definition

A memory type consists of three components (WebAssembly 3.0):
- **Address type**: `i32` or `i64` (determines index width)
- **Limits**: minimum (required) and optional maximum, both in pages
- **Page unit**: sizes expressed in page increments

Example in WAT:
```wat
(memory 1 10)  ;; initial = 1 page (64 KB), max = 10 pages (640 KB)
```

### Load/store instructions

All memory access is through typed load/store instructions. Key facts:

| Instruction | Opcode | Description |
|---|---|---|
| `i32.load` | `0x28` | Load 4 bytes as i32 |
| `i64.load` | `0x29` | Load 8 bytes as i64 |
| `f32.load` | `0x2a` | Load 4 bytes as f32 |
| `f64.load` | `0x2b` | Load 8 bytes as f64 |
| `i32.load8_s` | `0x2c` | Load 1 byte, sign-extend to i32 |
| `i32.load8_u` | `0x2d` | Load 1 byte, zero-extend to i32 |
| `i32.load16_s` | `0x2e` | Load 2 bytes, sign-extend to i32 |
| `i32.load16_u` | `0x2f` | Load 2 bytes, zero-extend to i32 |
| `i32.store` | `0x36` | Store i32 (4 bytes) |
| `i32.store8` | `0x3a` | Store low byte of i32 |
| `i32.store16` | `0x3b` | Store low 2 bytes of i32 |

— Source: MDN WebAssembly Reference (developer.mozilla.org)

Key properties:
- **Byte order**: always little-endian, regardless of host platform
- **Alignment hints**: the instruction encodes a natural alignment hint, but misaligned access does NOT trap — it's just a hint for optimization. The spec says alignment is a hint, not a requirement.
- **Effective address**: `base_address (on stack) + static offset (in instruction)`. The effective address is computed as `i + offset` where `i` is the i32/i64 value on the stack.
- **Out-of-bounds**: traps. If `effective_address + size > memory.length`, the engine raises a trap.
- **Narrower stores wrap**: `i32.store8` takes the low byte of the i32 value.

---

## 2. Guard Page Trick

### The problem

Every `i32.load` and `i32.store` needs bounds checking. Naive approach:

```
if (base + offset >= memory_size) trap();
value = memory[base + offset];
```

This is a conditional branch on every single memory access. For memory-intensive code, this is catastrophic.

### The virtual memory solution

All major engines use virtual memory guard pages to eliminate bounds checks at zero instruction cost in the hot path.

**How it works:**
1. Engine reserves a large contiguous virtual address region (much larger than the actual Wasm memory)
2. The actual Wasm memory pages are mapped as readable/writable
3. Beyond the valid memory, guard pages are mapped as `PROT_NONE` (no access) or simply left unmapped
4. Any out-of-bounds access hits the guard region, triggers a hardware fault (SIGSEGV on Unix, EXCEPTION_ACCESS_VIOLATION on Windows)
5. The engine's signal/exception handler catches the fault, checks if it came from Wasm JIT code, and converts it to a Wasm trap

**Result**: Zero additional instructions in the compiled code for bounds checking. The CPU's MMU does it for free.

### Engine-specific implementations

#### V8 (Chrome)

V8 reserves an **8 GB** virtual memory region for 32-bit Wasm memory:

```cpp
constexpr size_t kFullGuardSize32 = uint64_t{8} * GB;
```
— Source: `v8/src/objects/backing-store.cc` (chromium.googlesource.com)

Guard regions are enabled when `trap_handler::IsTrapHandlerEnabled()` returns true. The reserved memory is allocated with `kNoAccess` permissions, then selectively committed as needed.

V8's trap handler:
- **Linux/macOS**: Catches `SIGSEGV` signals. Validates: (a) signal was kernel-generated (not user-sent), (b) faulted address belongs to Wasm memory. Supports x64, ARM64, LOONG64, RISCV64.
- **Windows**: Uses Vectored Exception Handling. Catches `EXCEPTION_ACCESS_VIOLATION`. Sets `Rip`/`R10` (x64) or `Pc`/`X16` (ARM64) to redirect to a landing pad.

Source: `v8/src/trap-handler/handler-inside-posix.cc` and `handler-inside-win.cc` (chromium.googlesource.com)

The V8 code comments emphasize the security sensitivity: "Signal handlers are notoriously difficult to get right, and getting it wrong can lead to security vulnerabilities."

#### SpiderMonkey (Firefox)

SpiderMonkey defines these constants in `WasmMemory.h` (searchfox.org/mozilla-central):

| Constant | Value | Purpose |
|---|---|---|
| `HugeOffsetGuardLimit` | `1 << 25` = **32 MiB** | Catches folded base+offset accesses |
| `HugeUnalignedGuardPage` | 64 KiB (1 page) | Catches slop from unaligned accesses |
| `GuardSize` (non-huge) | 64 KiB (1 page) | Smaller guard for constrained mode |
| `NullPtrGuardSize` | 4096 bytes | Catches near-NULL pointer access |

Total reservation for "huge memory" mode:
```
HugeMappedSize = HugeIndexRange (4 GiB) + HugeOffsetGuardLimit (32 MiB) + HugeUnalignedGuardPage (64 KiB)
             ≈ 4 GiB + 32 MiB + 64 KiB
```

The 32 MiB guard limit comes from real-world analysis. SpiderMonkey analyzed large Wasm module corpuses and found that **20 MiB was the maximum offset immediate** encountered. They rounded up to 32 MiB (next power of two). This is also referenced by Wasmtime as the basis for their guard size.

From `WasmMemory.cpp`:
> "All memories in a process use the same strategy, selected at process startup" — because compiled machine code embeds the strategy, preventing per-memory variation.
> "The signal handler is the final source of truth" for catching out-of-bounds accesses.

#### Wasmtime (Bytecode Alliance)

Wasmtime's defaults from `tunables.rs` (github.com/bytecodealliance/wasmtime):

**64-bit hosts:**
| Setting | Default | Purpose |
|---|---|---|
| `memory_reservation` | **4 GiB** (2^32) | Total virtual memory per Wasm memory |
| `memory_guard_size` | **32 MiB** (32 × 2^20) | Guard region after accessible memory |
| `memory_reservation_for_growth` | **2 GiB** (2 × 2^30) | Pre-reserved for memory.grow |
| `signals_based_traps` | enabled | Use signal handlers, not explicit checks |

**32-bit hosts:**
| Setting | Default | Purpose |
|---|---|---|
| `memory_reservation` | **10 MiB** (10 × 2^20) | Scaled down for limited address space |
| `memory_guard_size` | **64 KiB** (0x1_0000) | Minimal guard |
| `memory_reservation_for_growth` | **1 MiB** (2^20) | Minimal pre-reservation |

Rationale from Wasmtime source: "For 32-bit we scale way down to 10MB of reserved memory. This impacts performance severely but allows us to have more than a few instances running around."

The contributing architecture doc describes the overall strategy: "2GiB unmapped before linear memory, 4GiB for linear memory itself, and 2GiB unmapped afterwards" — totaling 8 GiB reservation. The before-guard is defense-in-depth; the after-guard enables bounds check elimination for WebAssembly's 33-bit effective addressing (i32 base + i32 offset = up to 33 bits).

### Why the trick works for Wasm specifically

The 32-bit address space is the key. A Wasm i32 index can be at most 2^32 - 1 = 4,294,967,295. If you map 4 GiB of virtual memory (+ guard pages), then ANY possible i32 index either hits valid memory or hits the guard region. No explicit check needed. The CPU's MMU handles it via the page table.

This is why engines reserve 4+ GiB: it covers the entire i32 address space. The extra beyond 4 GiB covers the offset component of `base + offset`.

### Performance impact

No published benchmarks isolate guard-page overhead vs. explicit-check overhead specifically. However:

- The Jangda et al. USENIX ATC 2019 paper ("Not So Fast") found Wasm runs 1.45x-1.55x slower than native on SPEC CPU. They attribute the gap to: increased register pressure (2x more loads/stores), more branches (including stack overflow checks per function call and indirect call type checks), and larger code size — NOT to bounds checking overhead. This is because guard pages make bounds checking free.

- The paper notes that "WebAssembly requires several dynamic safety checks" including "stack overflow checks per function call" and "function table indexing checks" which add conditional branches. These are distinct from memory bounds checks (which guard pages handle).

- Wasmtime notes that explicit bounds checking on 32-bit hosts "impacts performance severely."

### What about 32-bit hosts?

On 32-bit hosts, you can't reserve 4+ GiB of virtual address space. So engines fall back to **explicit bounds checks** or much smaller guard regions. This is why 32-bit performance is worse.

---

## 3. memory.grow Semantics

### Instruction behavior

From the WebAssembly spec (webassembly.github.io/spec/core/exec/modules.html) and MDN:

```
memory.grow $delta  ;; $delta = number of pages to add
```

- **On success**: returns the **previous size of memory in pages** (i32). NOT the new size.
- **On failure**: returns **-1** (i32, which is 0xFFFFFFFF unsigned).
- **New memory is zero-initialized**: The spec explicitly states:

> "Let meminst' be the memory instance { type (at [i' .. j?] page), bytes b* 0x00^(n · 64 Ki) }"

The new bytes appended are `0x00^(n · 64 Ki)` — n pages worth of zero bytes.
— Source: WebAssembly Spec, Execution: Modules, `growmem` (webassembly.github.io)

- **Failure conditions**: Grows fails (returns -1) if:
  - `current_pages + delta > maximum_pages` (declared maximum exceeded)
  - Implementation-defined limits exceeded

### JavaScript API behavior

`WebAssembly.Memory.prototype.grow(delta)`:
- Returns previous size in pages (same as the instruction)
- Throws `RangeError` if growth would exceed maximum
- **Critical**: After grow(), the previous `ArrayBuffer` reference is **detached** (byteLength becomes 0). You must re-read `memory.buffer` to get the new buffer.
- **Exception for SharedArrayBuffer**: When `shared: true`, the original SharedArrayBuffer is NOT detached. Its `byteLength` doesn't update, but a new larger SharedArrayBuffer is accessible via `memory.buffer`. All references share the same underlying memory.

— Source: MDN WebAssembly.Memory.grow() (developer.mozilla.org)

### Does NOT relocate existing memory

This is a key property. `memory.grow` extends the linear memory **in place**. The base address stays stable. Existing content is preserved. This is possible because:
1. Engines pre-reserve a large virtual address range (4+ GiB on 64-bit)
2. Growth just commits more pages within that already-reserved range
3. No data copying or pointer fixup needed

This is fundamentally different from `realloc()` in C, which may relocate.

---

## 4. Why 64 KB Pages

### Official rationale

From the WebAssembly design rationale document (github.com/WebAssembly/design/blob/main/Rationale.md):

> "To allow efficient engines to employ virtual-memory based techniques for bounds checking, memory sizes are required to be page-aligned. For portability across a range of CPU architectures and operating systems, WebAssembly defines a fixed page size."

> "64KiB represents the **least common multiple** of many platforms and CPUs."

### Platform page sizes

| Platform | Page Sizes |
|---|---|
| x86/x86-64 | **4 KB** (standard), 2 MB / 1 GB (huge pages) |
| ARM (AArch32) | **4 KB** |
| ARM64 (AArch64) | **4 KB**, **16 KB**, or **64 KB** (configurable at kernel build time) |
| Windows VirtualAlloc | 4 KB page size, but **64 KB allocation granularity** |

— ARM64 page sizes: Linux kernel docs, `arch/arm64/booting.html` (docs.kernel.org). The kernel header encodes page size as: 0=unspecified, 1=4K, 2=16K, 3=64K.

### Windows allocation granularity — the binding constraint

Windows `VirtualAlloc` has a **64 KB allocation granularity** — the starting address of any reservation is rounded down to the nearest 64 KB boundary. This is distinct from the 4 KB page size.

The historical reason for Windows' 64 KB granularity: it was designed for the Alpha AXP RISC processor. RISC processors load 32-bit immediates as two 16-bit halves. With 64 KB alignment, DLL relocation only needs to fix up the upper 16 bits, avoiding complex carry handling between halves. Without this, every address computation would need an extra instruction — a ~50% penalty on address-heavy code.

— Source: Raymond Chen, "The Old New Thing" (devblogs.microsoft.com/oldnewthing/20031008-00/?p=42223)

### So why 64 KB specifically?

64 KB is the **lowest common denominator** (or more precisely, the LCM):
- It's a multiple of x86's 4 KB pages (64 KB = 16 × 4 KB)
- It matches ARM64's largest base page option (64 KB)
- It matches Windows VirtualAlloc allocation granularity (64 KB)
- It's a power of two (2^16)

### Future: Custom Page Sizes proposal

The fixed 64 KB page was too coarse for embedded/constrained environments. The **Custom Page Sizes** proposal (Phase 3, March 2026) allows:
- **1 byte** pages (minimum)
- **65,536 byte** pages (the traditional default)

Future expansion may allow any power of two between 1 and 65,536.

Motivation: "If a Wasm module only requires a small amount of additional working memory, it doesn't need to reserve a full 64 KiB." Embedded systems with < 64 KiB total RAM couldn't use Wasm at all with the fixed page size.

— Source: github.com/WebAssembly/custom-page-sizes/blob/main/proposals/custom-page-sizes/Overview.md

---

## 5. Memory Layout for Compiled C/C++

### WebAssembly C ABI (BasicCABI)

WebAssembly uses a **Harvard architecture**: code and data occupy separate spaces. The Wasm call stack (operand stack, locals, return addresses) is invisible in linear memory — it's managed by the engine. Only address-taken local variables need explicit stack frames in linear memory.

— Source: github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md

Data model: **ILP32** for wasm32 (int, long, pointer all 32-bit). Future wasm64 uses LP64.

### Emscripten / LLVM wasm-ld memory layout

The memory layout (from low to high addresses):

```
Address 0          ← NULL (never mapped, catches null pointer dereference)
...
GLOBAL_BASE (1024) ← Static data starts here
...
__data_end         ← End of static data
...
                   ← Stack (grows DOWNWARD from __stack_pointer)
...
__heap_base        ← Heap starts here (grows UPWARD via sbrk/memory.grow)
...
                   ← Top of memory (grows via memory.grow)
```

Key facts:
- **GLOBAL_BASE** defaults to **1024** (0x400). "Useful for optimizing load/store offsets." The first 1024 bytes are unused/unmapped to catch null pointer access.
- **Static data**: Lives between `GLOBAL_BASE` and `__data_end`. Compiler-managed globals, string literals, etc.
- **Stack**: Comes after static data, grows **downward** (toward lower addresses). This is the "shadow stack" in linear memory — used for address-taken locals and variable-length arrays. NOT the Wasm operand stack.
- **Heap**: Starts at `__heap_base`, grows **upward** toward higher memory.

— Source: Surma, "C to WebAssembly" (surma.dev/things/c-to-webassembly/) — confirmed by LLVM/wasm-ld behavior.

### Stack in linear memory

The BasicCABI defines:
- **SP (Stack Pointer)**: Points to bottom of frame, maintains **16-byte alignment**
- **Red zone**: 128-byte safety buffer below SP. Leaf functions needing < 128 bytes can skip updating SP entirely.
- Stack grows **downward** (high to low addresses)

Frame layout (high to low):
```
BP       | Previous frame boundary
...      | Alignment padding
FP + s   | Static-size objects
SP + d   | Dynamic-size objects
SP - 128 | Red zone (128 bytes, leaf functions only)
```

— Source: github.com/WebAssembly/tool-conventions/blob/main/BasicCABI.md

### Emscripten default settings

From Emscripten settings reference (emscripten.org/docs/tools_reference/settings_reference.html):

| Setting | Default | Notes |
|---|---|---|
| `STACK_SIZE` | **65,536 bytes (64 KB)** | Cannot be enlarged after init |
| `INITIAL_HEAP` | **16,777,216 bytes (16 MB)** | Initial heap memory |
| `INITIAL_MEMORY` | **-1** (auto-calculated) | Computed from INITIAL_HEAP + STACK_SIZE + static data |
| `MAXIMUM_MEMORY` | **2,147,483,648 bytes (2 GB)** | Max when ALLOW_MEMORY_GROWTH is on |
| `ALLOW_MEMORY_GROWTH` | **false** | Must be explicitly enabled |
| `GLOBAL_BASE` | **1024** | Start of static data in linear memory |

So the default Emscripten output starts with roughly: 1024 (null guard) + static data + 64 KB stack + 16 MB heap ≈ ~16-17 MB initial memory.

The USENIX ATC 2019 paper used `-s TOTAL_MEMORY=1073741824` (1 GB) for SPEC CPU benchmarks with `-s ALLOW_MEMORY_GROWTH=1`.

---

## 6. Security Implications

### Fundamental isolation guarantee

A Wasm module **cannot access memory outside its own linear memory** — period. This is enforced at two levels:

1. **Instruction-level**: Load/store instructions can only address the module's own linear memory. There is no instruction to read arbitrary process memory, access the DOM, or touch other modules' memory.

2. **Bounds checking**: Every memory access is bounds-checked (either via guard pages or explicit checks). Out-of-bounds access traps rather than accessing adjacent memory.

> "Whenever there's a load or a store in WebAssembly, the engine does an array bounds check to make sure that the address is inside the WebAssembly instance's memory."
> — Lin Clark, Mozilla Hacks (hacks.mozilla.org/2017/07/memory-in-webassembly-and-why-its-safer-than-you-think/)

### Buffer overflows stay sandboxed

A buffer overflow in a C program compiled to Wasm can still corrupt data — but ONLY within that module's own linear memory. It cannot:
- Escape the linear memory sandbox
- Access the JavaScript heap
- Read/write browser internals
- Touch other Wasm modules' memory
- Access the file system or network

This is a fundamental improvement over native code, where a buffer overflow can corrupt the entire process address space.

### Per-module memory isolation

Each Wasm **Instance** gets its own isolated memory:

> "A Module paired with all the state it uses at runtime including a Memory, Table, and set of imported values."
> — MDN WebAssembly Concepts (developer.mozilla.org)

Multiple modules do NOT share a common address space (unlike threads in a native process). Modules can only share memory if they explicitly import/export the same `WebAssembly.Memory` object.

### Shared memory for threads

When `shared: true` is set on a `WebAssembly.Memory`, the buffer is a `SharedArrayBuffer` instead of an `ArrayBuffer`. This enables:
- Multiple Wasm instances (in different Web Workers) to share the same linear memory
- Atomic operations for synchronization

```javascript
const sharedMemory = new WebAssembly.Memory({
  initial: 10,
  maximum: 100,
  shared: true,  // Returns SharedArrayBuffer
});
```

SharedArrayBuffer requires COOP/COEP headers (Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy) for security.

— Source: MDN WebAssembly.Memory (developer.mozilla.org)

### The 4 GB limit and Memory64

With i32 addressing, a Wasm module can address at most **2^32 bytes = 4 GiB** of linear memory. In practice, V8 limits 32-bit Wasm to:
- **4 GiB** (65,536 pages) on 64-bit hosts
- **2 GiB - 64 KiB** (32,767 pages) on 32-bit hosts

— Source: `v8/src/wasm/wasm-limits.h` (chromium.googlesource.com)

The **Memory64 proposal** extends addressing to i64:
- Limits use `u64` instead of `u32`
- Memory arguments support 64-bit offsets
- V8 limits 64-bit Wasm to **16 GiB** (262,144 pages) on 64-bit hosts
- The spec allows up to **128 TiB** (page counts remain 32-bit values supporting up to this amount)

Memory64 status (as of March 2026):
- **Not listed in active proposals** — the repository was archived January 15, 2025, indicating it has been merged into the core spec
- Implementation complete in: V8/Chrome, Firefox, wabt, binaryen, emscripten
- Table64 extension still in progress for V8/Chrome and Firefox

— Source: github.com/WebAssembly/memory64/blob/main/proposals/memory64/Overview.md

---

## 7. Concrete Numbers

### Default/maximum memory sizes

| Metric | Value | Source |
|---|---|---|
| Wasm page size | **65,536 bytes (64 KB)** | WebAssembly Spec |
| Max pages (32-bit, spec) | **65,536 pages = 4 GiB** | WebAssembly Spec |
| Max pages (32-bit, V8 on 64-bit host) | **65,536 pages = 4 GiB** | `wasm-limits.h` |
| Max pages (32-bit, V8 on 32-bit host) | **32,767 pages = 2 GiB - 64 KiB** | `wasm-limits.h` |
| Max pages (64-bit, spec) | **262,144 pages = 16 GiB** | `wasm-limits.h` |
| Emscripten default STACK_SIZE | **65,536 bytes (64 KB)** | Emscripten settings reference |
| Emscripten default INITIAL_HEAP | **16,777,216 bytes (16 MB)** | Emscripten settings reference |
| Emscripten default MAXIMUM_MEMORY | **2,147,483,648 bytes (2 GB)** | Emscripten settings reference |
| Emscripten default GLOBAL_BASE | **1024 bytes** | Emscripten settings reference |

### Virtual memory reservations by engine

| Engine | Reservation (64-bit host) | Guard Size | Source |
|---|---|---|---|
| V8 | **8 GiB** total | Included in 8 GiB | `backing-store.cc` |
| SpiderMonkey | **~4 GiB + 32 MiB + 64 KiB** | 32 MiB offset guard + 64 KiB unaligned guard | `WasmMemory.h` |
| Wasmtime | **4 GiB** reservation + **32 MiB** guard | 32 MiB | `tunables.rs` |

### Chrome heap limit context

From the USENIX ATC 2019 paper:
> "JavaScript contexts (like the main context and each web worker context) have a fixed limit on their heap sizes, which is currently approximately **2.2 GB** in Google Chrome."

This limit applies to the ArrayBuffer backing Wasm memory.

### SPEC CPU benchmark memory usage

The USENIX paper compiled SPEC CPU with `-s TOTAL_MEMORY=1073741824` (**1 GB**) and `ALLOW_MEMORY_GROWTH=1`. Two benchmarks (`638.imagick_s` and `657.xz_s`) were excluded because they require **more than 4 GB** of RAM — exceeding Wasm's 32-bit address space.

### ffmpeg.wasm

The core library download is **~31 MB** (the `.wasm` binary). Runtime memory allocation not publicly documented.

— Source: ffmpegwasm.netlify.app/docs/getting-started/usage

### Performance overhead (Wasm vs native)

From Jangda et al., "Not So Fast" (USENIX ATC 2019):

| Metric | Chrome | Firefox |
|---|---|---|
| **Mean slowdown** (SPEC CPU, geomean) | **1.55x** | **1.45x** |
| **Median slowdown** | **1.53x** | **1.54x** |
| **Peak slowdown** | **2.5x** | **2.08x** |
| Loads retired (vs native) | **2.02x** more | **1.92x** more |
| Stores retired (vs native) | **2.30x** more | **2.16x** more |
| Branch instructions retired | **1.75x** more | **1.65x** more |
| Instructions retired | **1.80x** more | **1.75x** more |
| CPU cycles | **1.54x** more | **1.38x** more |
| L1 I-cache misses | **2.83x** more | **2.04x** more |

Root causes identified (NOT bounds checking — that's handled by guard pages):
1. **Increased register pressure**: Chrome reserves `r13` for GC roots, `r10` as scratch. Firefox reserves `r15` for heap pointer, `r11` as scratch. None available for Wasm code.
2. **Stack overflow checks**: Both engines add a comparison + conditional branch at the start of every function call.
3. **Indirect call type checks**: Every `call_indirect` verifies the function signature matches.
4. **Poor register allocation**: Both use linear scan allocators vs. Clang's graph-coloring allocator.
5. **Increased code size**: 1.80x more instructions → 2.83x more L1 I-cache misses.

WebAssembly is **1.54x faster** than asm.js (mean speedup) in Chrome, and **1.39x faster** in Firefox.

---

## Key Diagram Ideas

1. **Linear memory as a flat array**: Show address 0 at left, growing right, with typed overlays showing how i32.load at offset 100 reads 4 bytes [100..103].

2. **Guard page layout**: Show the virtual address space: [null guard | valid Wasm memory | GUARD PAGES (unmapped)]. An arrow showing OOB access hitting the guard → SIGSEGV → trap handler → Wasm trap.

3. **Emscripten memory layout**: Vertical strip showing: NULL zone (0-1023) | GLOBAL_BASE (1024) | static data | stack (growing down) | heap_base | heap (growing up) | memory.grow extends here.

4. **memory.grow semantics**: Before/after diagram showing pages, return value, zero-initialization of new pages.

5. **V8's 8 GiB reservation**: Show 4 GiB address space + 4 GiB guard fully covering any possible i32 index.
