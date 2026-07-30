# Deep-Dives House Style

The average technical blog post takes 4 minutes to read and teaches you what a function does. A deep-dive takes 45 minutes and changes how you think about the system underneath it. That difference is not length — it is structure, voice, and the decision to never hand-wave.

This document is the canonical reference for writing new `deep-dives` essays. It is also a worked example of the voice it describes.

---

## What Makes These Articles Different

A reader finishes a deep-dive and can draw the system on a whiteboard. Not from memory — from understanding. They know where the bottleneck lives, why the obvious design fails, and what production systems do about it.

Three properties make that happen:

**Every abstraction is unpacked.** If you say "the kernel copies the buffer," you show the syscall, the byte count, the latency cost. If you name a data structure, you show its fields. Readers trust these articles because they can follow the chain all the way down to hardware.

**The structure teaches.** Each section earns the next one. You don't explain NAT traversal because it's "next in the outline" — you explain it because the tunnel from the previous chapter fails the moment you try it on a real network. Consequence drives sequence.

**Visuals are arguments.** A diagram is not an illustration of what the prose already said. It is a teaching artifact that carries information the prose cannot — byte layouts, state machines, protocol timelines, interactive inspectors. The prose frames the question; the visual answers it.

### Canonical References

Study these before writing a new piece. **The inference engine article is the primary reference** for voice, tab structure, setpiece design, and interactive teaching.

- `inference-engine-deep-dive.html` — **the gold standard**: semantic tabs (Intuition/Watch It/Trivia), the Sampling Explorer setpiece, D3 pipeline animation, act breaks, probability-colored text, pipeline breadcrumbs. Start here.
- `buffer-hacking.html` — interactive canvas demos, the diner-sign framebuffer setpiece, functional teaching animations
- `gguf-deep-dive.html` — the most evolved component library: interactive metadata trees, type badges, Gwern-style popups, inline SVG
- `efficiency-papers-1a-making-it-fit.html` — series styling, interactive visualizations with range sliders, timeline components
- `network-deep-dive.html` — the baseline: 14-chapter pacing, byte layouts, multi-language code tabs

---

## Voice

### Open with consequence, not definition

The first sentence of a section should hit. Lead with the number, the cost, the failure mode — not the Wikipedia definition.

> ❌ "The KV cache is a data structure that stores key-value pairs for attention layers."
>
> ✅ "Every token you generate means reading the entire KV cache. At Qwen 72B with 32K context, that is **10 GB of reads — for one word**."

> ❌ "A buffer is a region of memory used for temporary storage."
>
> ✅ "Your keyboard types at 100 WPM. Your CPU runs at 5 GHz. Without a buffer between them, every keystroke would either freeze the CPU waiting or get lost."

The consequence makes the reader care. The definition can come after.

### Metaphors must do work

One metaphor per major section. It must collapse a complex idea into a single image the reader carries forward. If the metaphor doesn't change how someone thinks about the system, cut it.

> "You know those old signs at diners — the ones where individual light bulbs spell out 'OPEN' and the owner can reach behind the sign and unscrew a bulb to turn off one letter? The framebuffer is that sign. Every pixel on your screen is a bulb. The framebuffer is the wiring board behind it."

That is not decoration. It rewires the reader's model of what a screen is — from "a display device" to "a memory-mapped array of colored lights." Every subsequent explanation of pixel manipulation is simpler because the diner sign is doing the work.

Bad metaphor: "Encryption is like a lock." (Everyone already thinks this. It teaches nothing.)
Good metaphor: "Quantization is **triage** — the edges of the network get more precision because errors there propagate through every layer." (Reframes quantization from "making things smaller" to "deciding what matters.")

### Short sentences are percussion

Alternate compression and expansion. Open with a short payoff line. Follow with a longer explanatory paragraph. Then punch again.

> "Just you, memory, and photons."
>
> "Get this one wrong and the runtime cannot find anything."
>
> "A nonce is not a suggestion. Reuse it once and the entire keystream is recoverable."

These are the sentences readers remember. Place them at section pivots — the moment where the concept lands.

### Close by opening

Chapters do not end with summaries. Summaries tell the reader what they already read. Instead, each closer is a gate to the next question:

> "Every one of those boxes labeled BUFFER is a place where you can intervene. Where you can do something *wild*."

> "This is the central insight of every optimization in this article: **trade compute for memory access**. The GPU has compute to spare. What it lacks is bandwidth."

The reader is pulled forward, not sent home.

### "You" for agency

The reader is the actor, not a spectator. "You" creates ownership. "We" creates distance.

> ✅ "If you could write directly to that array, you'd be painting the screen with nothing but memory writes."
>
> ❌ "We can see that writing directly to the array would allow screen painting."

Use "we" only for editorial choices: "we start with GGUF because..." Use third person for systems and historical actors: "Gerganov had the model running on a MacBook."

### Frame every visual

One sentence before each diagram or interactive element, promising what it will prove:

> "Click any key below to see what the runtime does with it:"
>
> "The visualization below shows what that looks like as context grows."
>
> "You press a key. It feels instant. It isn't. Between your finger and the character appearing on screen, the keystroke passes through *five separate buffers*:"

Never drop a diagram mid-flow without setup. The framing sentence is the contract: "here is what you are about to see and why it matters."

### Transitions by consequence

Not "next" or "then." The end of one section creates the problem the next section solves:

> "You have seen the memory wall. Now look at where most of that memory goes."

> "Our tunnel works beautifully when both peers can see each other. But in the real world, almost every device hides behind a NAT."

The reader crosses from one section to the next because they *need* to, not because the table of contents says so.

### No hand-waving

If you name a layer, table its properties. If you say "the keystroke passes through several buffers," show every buffer — its size, its speed, its lifetime, its purpose. The reader came here because other articles said "and then the kernel handles it." We show the kernel handling it.

> "Three hundred typed key-value pairs, and the very first one is the most important."

Then show which key, what type, what the runtime does with it.

### Teach by unpacking

Every major section follows this skeleton:

1. **Concept** — one sentence that names the thing and its consequence
2. **Intuition** — a metaphor or mental model that reframes the reader's thinking
3. **Mechanism** — how the gears work (code, diagrams, byte layouts)
4. **Proof** — working code or interactive demo that the reader can verify
5. **Play** — an interactive element, a "try this" prompt, or an implication that branches the idea outward

Not every section needs all five. But concept → mechanism → proof is the minimum.

### The Setpiece Doctrine

Every article needs one major interactive that **is** the article. Not a supporting illustration. The thing a reader screenshots, shares, and remembers six months later.

A setpiece is not a chart with sliders. It is a **self-contained teaching instrument** where the reader's interaction produces genuine surprise. The reader should walk away thinking "I finally understand this" because they *played* with it, not because they read about it.

**Canonical setpieces:**

| Article | Setpiece | Why it works |
|---|---|---|
| `inference-engine-deep-dive.html` | **Sampling Explorer** | Drag temperature/top-p/min-p sliders, watch the same sentence mutate word by word. Probability colors every token. The reader *feels* what temperature does instead of reading a formula. |
| `buffer-hacking.html` | **Diner-sign framebuffer** | Individual light bulbs glow and dim as memory bytes change. The metaphor is not described; it is rendered. The reader sees memory-mapped I/O by watching bulbs. |
| `gguf-deep-dive.html` | **Metadata inspector tree** | Click any key, see its type, offset, and raw bytes. The format's complexity is navigable, not listed. |

**Setpiece requirements:**

1. **Interactivity is the explanation.** Removing the setpiece should leave a hole in the reader's understanding, not just a missing widget.
2. **One concept, many angles.** The Sampling Explorer teaches one idea (sampling shapes output) but lets you approach it from three directions (temperature, top-p, min-p).
3. **Color encodes meaning.** Probability-colored text (green >50%, cyan 20-50%, yellow 5-20%, magenta <5%) makes the invisible visible. Strikethrough for pruned tokens.
4. **Position after build-up.** The setpiece appears after the reader understands the components, not before. The Sampling Explorer comes after the sampling pipeline is explained. The diner sign comes after the framebuffer concept is introduced.
5. **Self-contained.** Works without reading the surrounding prose. Someone landing on just that section should still learn something.

**Finding your setpiece:** Ask "what is the one thing in this topic that only makes sense when you interact with it?" That is the setpiece. If everything in the topic can be explained in prose and static diagrams, the topic might not be deep-dive material.

---

## Visual System

### The Dark Terminal Magazine

The aesthetic is a dark terminal-native magazine. Near-black backgrounds. Neon green as the primary signal color. Serif body text for readability. Monospace for everything structural — headings, labels, code, navigation.

```
--bg: #0a0a0f           near-black background
--bg-card: #12121a      card/panel background
--bg-code: #0e0e16      code block background
--text: #c8c8d4         body text, muted gray
--text-dim: #6a6a7e     dimmed labels
--green: #00ff88         primary — headings, main signal, "this is the point"
--cyan: #00ddff          secondary — subheadings, links, "look here next"
--magenta: #ff00aa       accent — keywords, labels, "pay attention"
--orange: #ff8800        support — warnings, wild techniques
--red: #ff3344           danger — failure modes, critical warnings
--yellow: #ffdd00        numbers, highlights
--blue: #4488ff          function names, source addresses
--purple: #aa44ff        annotations, preprocessor
```

Green means "this is the main signal." Other colors carry semantic meaning: red is danger, orange is "this is wild but it works," cyan is "here is what to look at next." Colors are vocabulary, not decoration.

### Typography

- **Body**: Source Serif 4 — readable serif, 18px, line-height 1.7
- **Headings, nav, labels, code**: JetBrains Mono
- **Generated/scored text**: Source Serif 4 at 1.15rem, line-height 2.0. This is the "textbot" voice: the same body font but slightly larger, with extra line-height to make room for colored underlines and probability annotations. Used inside setpieces like the Sampling Explorer where model output is displayed with per-token metadata.
- **Structural labels**: JetBrains Mono, 10-11px, uppercase, letter-spacing 0.08-0.15em. Pipeline breadcrumbs, act break numbers, diagram labels, tab labels all share this treatment.
- **Layout**: 820px max-width reading column, fixed nav with blur backdrop, generous vertical rhythm

**Color as typography:** In generated-text displays, word color IS information. A green word means >50% confidence. A magenta word means the model was guessing. Strikethrough dim text means pruned by top-p/min-p. This is not decoration; removing the color removes information. Document the color scale in a legend (`.viz-controls` row) beneath any probability-colored text block.

### Motion & Teaching Animations

The hero title glitches; the subtitle cursor blinks. Those are ambient. Everything else has a purpose.

**Teaching animations are encouraged.** The diner sign in buffer-hacking has light bulbs that glow and dim to show how a framebuffer maps memory to pixels. That animation *is* the explanation. A paragraph saying "each byte controls a pixel" is weaker than watching a bulb light up when a byte changes. Look for these opportunities in every article:

- A packet flowing through layers (highlighting each header as it gets added)
- A handshake sequence where messages animate between initiator and responder
- A NAT mapping table that updates as packets cross the boundary
- A state machine where the current state pulses and transitions animate on hover
- A counter incrementing inside a cipher to show nonce evolution
- Ring buffers filling and draining to show producer/consumer dynamics

The rule is not "no animation." The rule is **every animation teaches**. If you removed it and the reader lost understanding, it earned its place. If you removed it and nothing changed, it was decoration.

No parallax. No scroll-triggered effects. No ambient motion in body prose. Teaching animations live inside `.diagram`, `.js-demo`, or `.viz-container` elements, and they either respond to the reader's interaction or loop to demonstrate a continuous process.

**Implementation tools:** For D3.js interactive visualizations, use the `/technical-explainer` skill (concept-map-first approach, architectural building blocks) and the `/claude-d3js-skill` skill (d3.js best practices, SVG-based data visualization). These skills inform the design and implementation of setpieces and teaching animations. Load d3.v7 from CDN; wrap each animation in an IIFE; pair with `.viz-controls` for user interaction.

---

## Component Catalog

### Foundation (every article)

These are the floor. Every deep-dive has them:

- **Hero** with glitch title and blink cursor (`.hero`, `.glitch`, `.blink`)
- **Fixed nav** with short anchor labels (2–3 words per chapter)
- **Drop-cap** on section-opening paragraphs (`.drop-cap`)
- **Diagrams** — card-styled containers for ASCII art, SVG, or pre blocks (`.diagram`, `.diagram-label`)
- **Power callouts** — left-bordered insight boxes (`.power`). Variants: `.danger` (red), `.hack` (magenta), `.wild` (orange), `.zen` (cyan), `.purple`
- **Comparison blocks** — two-column grids for design tradeoffs (`.vs`, `.vs-card`)
- **Tables** — monospace headers, row hover, data-dense
- **Code blocks** — syntax-highlighted with top-right language label (`.label`)
- **Memory cells** — colored byte visualization (`.mem-row`, `.mc` with `.g`/`.c`/`.m`/`.o`/`.r`/`.y`/`.b`/`.d`/`.e`/`.w`/`.p` variants)
- **Syntax classes** — `.cm` (comment), `.kw` (keyword), `.ty` (type), `.st` (string), `.nm` (number), `.fn` (function), `.pp` (preprocessor)
- **Details/summary** — collapsible secondary content
- **Scanline overlay** — the subtle CRT effect across the whole page

### Evolved Components

These emerged from specific articles and are now standard tools. Use them when the content demands it — not as decoration.

**Tabbed panels** — CSS-only tab switching using radio inputs (`.tabbed`, `.tab-bar`, `.tab-panel`). Up to 4 tabs per group via `.t1`–`.t4` classes and `.p1`–`.p4` panels. No JavaScript needed. *(From: buffer-hacking, efficiency-papers, inference-engine)*

Tabs are not just for multi-language code. They are a **teaching rhythm**. The inference engine article establishes a semantic tab vocabulary:

| Tab name | Purpose | When to use |
|---|---|---|
| **Intuition** | Metaphor, mental model, "think of it as..." | Every concept that needs a reframe before the mechanism |
| **Watch It** / **Play It** | Interactive D3/Canvas animation the reader triggers | Any process that unfolds over time (generation loops, packet flows, state machines) |
| **Math** / **Code** | The formal mechanism, pseudocode, or real code | After intuition, never first |
| **Trivia** | Historical context, fun facts, surprising numbers | Reward for the curious; never load-bearing content |

**The "Intuition-first" rule:** When a tabbed panel has an Intuition tab, it must be the default (checked). The reader sees the mental model before the math. Code and math tabs are opt-in for readers who want depth.

**Tab naming is terse:** 1-2 words, title case, no articles. "Intuition" not "The Intuition Behind It." "Watch It Generate" not "Click Here to Watch the Generation Process."

**Metadata trees** — split-pane interactive inspector. A scrollable list on the left, click-to-reveal detail panel on the right. For any structured data where inspect-on-click beats a wall of JSON: config keys, peer tables, format fields. Requires ~40 lines of JS. *(From: gguf-deep-dive)*

**Expandable cards** — chevron-animated disclosure beyond `<details>`. Richer headers with color indicators, badges, and summary content. For lists of items with substantial hidden detail: tensor layers, protocol messages, peer connections. *(From: gguf-deep-dive)*

**Type badges** — small inline data-type indicators (`.tbadge`). 9px monospace, colored background per type. For metadata displays, tree views, or anywhere data types need visual differentiation. *(From: gguf-deep-dive)*

**Gwern-style popups** — hover-activated floating panels with title bar and scrollable body. Define a term once, reference it everywhere with `data-popup` attributes. Readers get a refresher without scrolling back. Requires ~50 lines of JS for positioning. *(From: gguf-deep-dive)*

**Interactive demos** — live JS containers with canvas, controls, range sliders, and dynamic readouts. For when the concept is best understood by playing with it. The reader adjusts a slider and sees the consequence immediately. *(From: buffer-hacking)*

**Timeline components** — chronological layout with monospace year labels and left-border visual. For tracing the evolution of a technique or protocol. *(From: efficiency-papers)*

**Scenario/decision cards** — grid of recommendation cards with colored left borders. For "which option should I choose?" contexts: format selection, architecture decisions, use-case mapping. *(From: gguf-deep-dive, mesh-vpn-deep-dive)*

**File layout bars** — horizontal bars showing file structure with names, descriptions, sizes. Hover reveals additional detail. *(From: gguf-deep-dive)*

**Act breaks** — cinematic structural dividers (`.act-break`) for articles with distinct movements. Centered, bordered top and bottom, with `.act-number` (dim monospace label), `.act-title` (green glow heading), and `.act-subtitle` (italic dim prose). Use sparingly: 2-3 per article maximum, at genuine shifts in the article's argument. Not section dividers; act breaks mark the moment the article changes *what question it is answering*. *(From: inference-engine)*

**Pipeline breadcrumbs** — inline colored stage indicators at the top of a chapter. Monospace, 10px, each stage colored by its semantic role (green for embed, yellow for norm, magenta for attention, orange for FFN, cyan for projection). Shows the reader the whole map before zooming in. Use when an article has a multi-stage pipeline and individual chapters zoom into one stage. *(From: inference-engine)*

**Probability-colored text** — inline text where each word's color encodes a continuous value (probability, confidence, importance). The Sampling Explorer uses: green >50%, cyan 20-50%, yellow 5-20%, magenta <5%, with strikethrough for pruned candidates. The text sits in Source Serif (body font) at 1.15rem with generous line-height (2.0), inside a `viz-container`. The colored underline (2px solid, 44% opacity of the word color) provides a second visual channel. Use whenever generated or scored text needs to show per-token metadata. *(From: inference-engine)*

**D3.js interactive animations** — SVG-based teaching animations driven by d3.v7. Include `<script src="https://d3js.org/d3.v7.min.js"></script>` in `<head>`. Wrap each animation in an IIFE to avoid namespace collisions. Pair with `.viz-controls` for buttons (green-dim background, monospace 11px) and `.viz-container` for the frame. Use for anything with state transitions, flowing data, or time-sequenced processes. The inference engine's token generation animation pulses through pipeline stages; the sampling explorer updates sentence words in real-time. *(From: inference-engine, efficiency-papers)*

**Inline SVG diagrams** — responsive SVG with `viewBox`, color-coded by semantic meaning, text labels inside the SVG. Preferred over ASCII art for complex diagrams: protocol comparisons, architecture overviews, state machines. Wrap in `.diagram` container.

---

## Structure

### Default Section Progression

1. Why this thing exists (the consequence that forced its invention)
2. The mental model (a diagram or metaphor the reader carries)
3. How it actually works (mechanism with code and byte layouts)
4. Where it gets hard (the bottleneck, the failure mode, the tradeoff)
5. What production systems do about it (real implementations, real numbers)
6. What this changes (zoom out to the broader system or industry)

Not every section follows all six steps. But the arc from "why" to "what it changes" should be present across the full article.

### Pacing

8–14 chapters is typical. Each chapter: 1,000–3,000 words plus diagrams and code. The article should have at least two "checkpoints" — moments where the reader has something working (a tunnel, a mesh, a parser) before the next layer of complexity adds to it.

---

## Deliverable Format

Ship as a single HTML file with embedded CSS and embedded JavaScript. Internal anchor navigation. No external dependencies except Google Fonts. At least three visual teaching aids per major section. The ending widens the lens — it does not summarize.

### Starter Checklist

Before drafting, lock:

- **Title** — keyword-first: "GGUF, Decoded" not "Understanding GGUF"
- **One-sentence thesis**
- **Section spine** — 8-14 chapters with working titles
- **The setpiece** — what is the one major interactive that IS this article? Name it, describe what the reader does, and what they learn by doing it. If you cannot answer this, the article is not ready to draft.
- **Tab plan** — which sections get tabbed panels? Default to Intuition | Math/Code for mechanism sections, Watch It/Play It | Code | Trivia for process sections
- **Required visuals** — at least four, at least two interactive (the setpiece counts as one)
- **Core tradeoff** — the tension the essay resolves
- **Evolved components** — which interactive elements (metadata tree, tabs, popups, D3 animations, probability text) this topic needs

---

## Research-to-Essay Workflow

1. Gather rough notes.
2. Find the one governing idea. Everything else orbits this.
3. Build the section spine. Each section title should be a noun or short phrase, not a sentence.
4. For each section, decide: diagram, code, table, interactive, or some combination.
5. Draft the hero title and subtitle *after* the structure is clear. The title is a promise; you need to know what you're promising.
6. Write transitions so each section earns the next one. If removing a section doesn't break the chain, the section is optional.
7. Remove every fact that doesn't change the reader's model.
8. Build the interactive components. These often reshape the sections they serve — a metadata tree might replace a page of JSON; a tabbed panel might replace a comparison that was struggling in prose.
9. Final pass: does every section follow concept → mechanism → proof? Does every diagram have a framing sentence? Does every chapter close by opening?
