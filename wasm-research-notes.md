# WebAssembly Deep Dive — Research Notes

Compiled 2026-03-20. Structured for direct use in 16-chapter HTML content.

## Progress

| Ch | Title | Research | Written | Committed |
|----|-------|----------|---------|-----------|
| 01 | The Escape from JavaScript | done | done | 471c614 |
| 02 | The Stack Machine | done | done | 3ae73b9 |
| 03 | The Binary Format | done | done | bd34f22 |
| 04 | Linear Memory | done | done | 10e345c |
| 05 | Types and Instructions | done | done | 9ad56ca |
| 06 | The Toolchain | done | done | c6dc0e8 |
| 07 | Handwriting Wasm | done | done | abea750 |
| 08 | The JavaScript Bridge | done | done | abea750 |
| 09 | How Browsers Actually Run It | done | done | f2f3454 |
| 10 | SIMD: The Speed Multiplier | done | done | f2f3454 |
| 11 | WASI & The Component Model | done | done | 632af65 |
| 12 | Wasm in the Wild | done | done | 632af65 |
| 13 | Inference in the Tab | done | done | 8af83ea |
| 14 | The Sandbox | done | done | 8af83ea |
| 15 | Inference at the Edge | done | done | 8af83ea |
| 16 | The Convergence | done | done | 8af83ea |

## Key Sources Used

- WebAssembly spec: webassembly.github.io/spec/core/
- Figma blog: "WebAssembly cut Figma's load time by 3x" (June 2017)
- Mozilla Hacks: asm.js announcement, streaming compilation, Firefox 58
- V8 blog: Liftoff compiler, Emscripten upstream LLVM backend
- WebLLM paper: arXiv 2412.15803 (browser inference benchmarks)
- Extism / Dylibso: plugin instantiation benchmarks
- Lin Clark's Code Cartoons: WebAssembly performance visualization
- "Not So Fast" (USENIX ATC 2019): Wasm vs native benchmarks
- "Bringing the Web up to Speed with WebAssembly" (PLDI 2017): founding paper

---
