---
name: media-release
description: Craft platform-specific submission packages for technical content — code, benchmarks, articles, tools, research, repos, or projects. Research-driven and content-agnostic. Shared research phase, then generates for HN, Reddit (per-subreddit), or both. Invoke with target platform and content.
---

# Media Release Skill

Research-driven launch packages for technical content across platforms. One research pass, then platform-specific generation — HN, Reddit (per-subreddit), or both.

## Mental Model

Every platform is a different ranking system with different trust signals. But the research is the same: what's the landscape, what gap does this content fill, and what are the hard claims you can make.

**Structure:**
1. **Shared phases** (1-2): Research + Content Analysis — run once, reuse everywhere
2. **Platform modules** (3+): Generate for the platforms the user wants — HN, Reddit, or both

Ask the user which platforms to target. If they don't specify, ask.

---

## PHASE 1: Landscape Research (Shared)

Before analyzing the content, research the competitive landscape. Use web search across all target platforms.

**For HN:**
- Search for top 5-10 prior HN submissions on the same or related topics
- Note which titles worked (high points), which flopped, what angles were used
- Find critic patterns in comments on related posts

**For Reddit (per target subreddit):**
- Search for "r/{sub}" + topic — find the top 5 posts from the last month
- Check sub rules, self-promotion policy, flair requirements
- Note format preferences (link vs text), tone, what gets upvoted in comments
- Check if self-promotion is banned or ratio-gated (dealbreaker — check first)

**For both:**
- **Identify the conversation gap** — what hasn't been said across all platforms? What do existing posts miss?
- **Check recency** — if similar content hit any target platform in the last 2-4 weeks, flag it
- **Note vocabulary per platform** — HN and Reddit subs often use different terms for the same thing

**Output:**

```
### Landscape Research

**Cross-platform gap:** [one sentence — the thing nobody has covered well]
**Recency risk:** clear / caution / wait

**HN landscape:**
- Top 3 prior posts: [title, points, date, key comment themes]
- HN vocabulary: [key terms]
- Critic patterns: [common objections]

**Reddit landscape (per sub):**
| Subreddit | Top related post | Self-promo policy | Format | Flair | Tone | Risk |
|-----------|-----------------|-------------------|--------|-------|------|------|
| r/... | "title" (Xpts) | ... | ... | ... | ... | ... |
```

---

## PHASE 2: Content Analysis (Shared)

Read the target content. This could be anything technical:

| Content type | What to read | Where the signal lives |
|-------------|-------------|----------------------|
| **Article / blog** | HTML file, URL, or markdown | Headings, narrative, embedded data |
| **Code / repo** | README, key source files, benchmarks dir, tests | Architecture decisions, perf numbers, design trade-offs |
| **Benchmark results** | Raw data, methodology, comparison tables | The numbers + methodology choices |
| **Library / tool** | README, API surface, examples, changelog | Problem it solves, how it differs, perf characteristics |
| **Research / paper** | Abstract, results, methodology | Novel claims, surprising findings, reproducibility |

**Extract:**

- **Title and description** — what does this thing call itself?
- **Structure** — sections, components, architecture, module layout
- **Hard data points** — benchmarks, measurements, percentages, timings, comparisons, profiling results, flame graphs, memory/throughput numbers
- **Differentiators** — what makes this different from existing resources? (Informed by Phase 1)
- **Runnable artifacts** — demos, playgrounds, CLI tools, Docker images, live endpoints, interactive visualizations, benchmark scripts

**Determine:**

- **Core insight** — one sentence: the single most non-obvious thing this content reveals or enables
- **Quotable facts** — 3-5 specific numbers, surprising findings, or counter-intuitive results
- **Platform-specific angles** — which slice of this content matters to which audience:
  - HN angle: [what the HN crowd cares about]
  - Per-subreddit angle: [what each sub's audience cares about — these will differ]

**Tier classification:**

| Tier | Criteria | Action |
|------|----------|--------|
| **S** | Fills the gap from Phase 1 + has hard numbers or original work + can't get this elsewhere | Proceed |
| **B** | Gap is unclear, no unique data, or similar posts performed well recently | Warn user. Suggest 2-3 angle-sharpening strategies. Do NOT proceed without confirmation. |

---

## PLATFORM MODULE: Hacker News

Generate when the user targets HN.

### HN-3. Title Engineering

Generate **5 title candidates** using proven HN patterns:

| Pattern | Template | When to use |
|---------|----------|-------------|
| **A — Builder narrative** | "I built X to solve Y — here's what actually mattered" | Created something, learned non-obvious lessons |
| **B — Counterintuitive hook** | "X is slower/harder/weirder than you think — here's why" | Data contradicts common assumptions |
| **C — Show HN** | "Show HN: X — [one-line value prop]" | Live demo URL + source repo both exist |
| **D — Technical dissection** | "X internals: how Y actually works under the hood" | Deep content on a system people use but don't understand |
| **E — Comparison** | "X vs Y: what the benchmarks don't tell you" | Original comparative analysis with data |

**Rank each 1-5 on:**
1. Specificity — vague = death
2. Curiosity gap — without clickbait
3. Depth signal — went deep, not wide
4. Length — under 80 characters
5. Differentiation — signals something Phase 1 prior posts didn't cover

**Reject any title that:** uses "Ultimate guide" / exclamation marks / superlatives without data / reads like a product launch / overlaps with recent HN front-page post.

### HN-4. Opening Hook

First 3 sentences must accomplish:
1. State a real, specific problem or question
2. Deliver a surprising insight or number
3. Signal credibility — what you built, measured, or dissected

**Banned:** "I've been working on...", "I recently...", "As a developer...", "I'm excited to share...", any sentence without a concrete claim.

Draft 2 variants:
- **Variant A (technical-first):** Lead with the data point
- **Variant B (narrative-first):** Lead with the problem

Each MUST contain at least one hard number from Phase 2.

### HN-5. Post Body

Strict skeleton — skip sections that don't apply, but maintain order:

1. **Problem** (2-3 sentences)
2. **Why existing solutions/resources fail** (2-3 sentences, use Phase 1 research)
3. **What this covers / does** (bullet list — concrete, not vague)
4. **What surprised me** (2-3 sentences — the intellectual bait)
5. **Hard numbers** (if applicable — bullet list or mini-table)
6. **Try it** (if applicable — how to run, demo, or interact)
7. **Tradeoffs and limitations** (1-2 sentences)
8. **Link** — clean URL, no tracking parameters

**Constraints:** 150-250 words. No exclamation marks. No hype. Engineer-to-engineer tone. Concrete > abstract.

### HN-6. Comment Prep

**First author comment** (post immediately): extra context not in body, invite specific discussion. Under 150 words.

**5 anticipated questions** (research-driven from Phase 1 critics):
- Pre-draft each response: engineer tone, metrics, specifics, under 100 words
- Acknowledge valid criticism directly, correct wrong criticism with data not attitude

### HN-7. Pre-Launch Checklist

```
- [ ] URL loads and is not paywalled
- [ ] Title under 80 characters
- [ ] Body 150-250 words
- [ ] Hard number in opening hook
- [ ] No exclamation marks anywhere
- [ ] No unsupported superlatives
- [ ] Runnable artifacts / demos called out
- [ ] First author comment drafted
- [ ] 5 pre-drafted responses ready
- [ ] Tradeoffs acknowledged
- [ ] Title doesn't overlap recent HN front-page post
- [ ] Landscape gap is clear
```

**Timing:** Tue-Thu, 8-10 AM US Pacific. Acceptable: Mon/Fri same window. Avoid: weekends, holidays, major tech news days.

### HN-8. Failure Protocol

If < 10 points after 2 hours:
1. Don't delete. Wait 2-3 weeks minimum
2. Different title from the ranked list
3. Sharpen the hook — harder number, more specific claim
4. Try different pattern (D → B, etc.)
5. Maximum 3 attempts per content

---

## PLATFORM MODULE: Reddit

Generate when the user targets Reddit. Produces a **separate, native post per subreddit**.

### Reddit-3. Subreddit Selection

From the Phase 1 research, filter to viable subs:

**Drop a sub if:**
- Self-promotion banned and no comment history there
- Content doesn't fit the sub's actual topic
- Very similar post in last 2 weeks
- Sub is dead (< 5 posts/day)

**Recommend posting order** — most forgiving sub first, to build a thread you can reference.

### Reddit-4. Per-Subreddit Post Crafting

**For EACH sub**, produce a complete, tailored post. No text reuse across subs.

#### Title (per sub culture)

| Sub culture | Title style | Example |
|-------------|-------------|--------|
| **Academic** (r/MachineLearning, r/compilers) | Precise, formal, methodology | "Profiling FFN vs attention memory bandwidth on consumer GPUs — results and methodology" |
| **Builder** (r/selfhosted, r/LocalLLaMA, r/homelab) | Practical, results-first | "Got my 3090 to 100 tok/s by fixing FFN bandwidth — here's the config" |
| **Engineering** (r/programming, r/systems, r/cpp) | Technical, insight-driven | "Memory bandwidth, not KV cache, is the real bottleneck in LLM inference" |
| **Niche** (r/CUDA, r/golang, r/rust) | Deep, assumes domain knowledge | "Avoiding warp divergence in mixed-precision FFN kernels" |

**Anti-patterns (all Reddit):** clickbait, "I made a thing" with no specifics, ALL CAPS, emoji on technical subs.

#### Body (per sub culture)

| Sub type | Lead with | Tone | Length |
|----------|-----------|------|--------|
| **Academic** | Methodology and scope, limitations early | Formal but not stiff | 200-400 words |
| **Builder** | The result, then setup, then journey/mistakes | Casual first-person | 200-500 words |
| **Engineering** | Technical insight, support with data | Technical but accessible | 100-200 words (link post + comment context) |
| **Niche** | Deep fast, code snippets inline, specific question | Assumes expertise | 100-300 words |

#### Flair & Tags

Research and specify exact flair. Many subs auto-remove without it.
- r/MachineLearning: `[R]` research, `[P]` project, `[D]` discussion
- r/LocalLLaMA: `Discussion`, `Tutorial`, `News`
- Others: check sub sidebar in Phase 1

### Reddit-5. Comment Strategy (Per Sub)

**First comment** (post immediately):
- Builder subs: share a mistake or dead end
- Academic subs: limitations or future work
- Niche subs: genuine question to the experts

**3 anticipated questions per sub** — based on Phase 1 comment patterns. Draft responses matching the sub's exact tone.

### Reddit-6. Cross-Post Strategy

- **Stagger 4-24 hours** between subs — simultaneous posting triggers spam filters
- **Start with the most forgiving sub**
- **Adapt later posts** based on earlier reception — if sub A raised a good point, address it preemptively in sub B
- **Never use Reddit's crosspost feature** for self-promotion — write each post natively

### Reddit-7. Per-Sub Checklist

For each target sub:
```
- [ ] Sub rules checked — compliant
- [ ] Self-promotion policy verified
- [ ] Correct flair selected
- [ ] Title matches sub culture and tone
- [ ] Body length appropriate for this sub
- [ ] No hype language
- [ ] Hard numbers included (if relevant)
- [ ] First comment drafted
- [ ] 3 anticipated responses ready
- [ ] Posting time optimal for this sub
```

**Timing:** US tech subs: 8-11 AM EST weekdays. Builder subs: evenings/weekends. Niche subs: check posting patterns. Global subs: morning EST (US+EU overlap).

### Reddit-8. Failure Protocol

**Removed post:** read reason, fix violation, wait 24+ hours, message mods if unclear.
**No traction (0-5 after 4 hours):** check for shadow-removal, wait 1-2 weeks, different title+time. Max 2 attempts per sub.
**Controversial (50%+ downvotes):** read the comments — they're telling you something. Engage genuinely. The thread can save a controversial post.

---

## Wrap Up

Present the complete **Launch Package**. Structure depends on which platforms were selected:

### If HN only:

```
## Launch Package: [Content Title]

### Research (shared)
[Phase 1 output]

### Content Analysis
[Phase 2 output]

### HN Package
[HN-3 through HN-8 output]
```

### If Reddit only:

```
## Launch Package: [Content Title]

### Research (shared)
[Phase 1 output]

### Content Analysis
[Phase 2 output]

### Reddit Package
[Subreddit assessment table]
[Per-sub sections with full posts]
[Cross-post schedule]
```

### If both:

```
## Launch Package: [Content Title]

### Research (shared)
[Phase 1 output — covers both platforms]

### Content Analysis
[Phase 2 output — with platform-specific angles]

### HN Package
[Full HN output]

### Reddit Package
[Full Reddit output — per sub]

### Cross-Platform Schedule
| Order | Platform | Target | Timing | Notes |
|-------|----------|--------|--------|-------|
| 1 | HN | — | [time] | Post first, gauge reaction |
| 2 | Reddit | r/... | [time] | Adapt based on HN reception |
| 3 | Reddit | r/... | [time] | Adapt based on #1-2 |
```

**IMPORTANT:** Every section must be complete, ready to paste, no placeholders, no "TBD". Each platform/subreddit post must be independently native to its community.
