---
name: hn-media-release
description: Craft a complete Hacker News submission package for any technical content — articles, tools, libraries, research, repos, or projects. Research-driven and content-agnostic. Produces title options, post body, pre-drafted comments, timing, and a launch checklist.
---

# HN Media Release Skill

Produce a copy-pasteable HN launch package for technical content. Treat every post like an experiment: research (landscape), hypothesis (title), payload (body), measurement (traction), iteration (failure protocol).

## Mental Model

HN rewards **earned insight under time pressure**. Not polish, not branding, not completeness. The ranking system optimizes for fast, credible curiosity. Everything below serves one goal: **trigger high-trust early velocity from technically credible users within ~30 minutes of posting.**

## Workflow

Make a todo list for all phases and mark each complete as you finish it.

### 1. Landscape Research

Before analyzing the content, research the competitive landscape on HN. Use web search to:

- **Search HN for the topic** — find the top 5-10 prior submissions on the same or closely related subjects. Note which titles worked (high points), which flopped, and what angles were used.
- **Identify the conversation gap** — what hasn't been said? What do the existing top posts miss? The content must fill a gap, not repeat a conversation.
- **Check recency** — if a similar post hit the front page in the last 2-4 weeks, the window may be closed. Flag this to the user.
- **Note the vocabulary** — what terms does the HN community use for this topic? Match their language, not marketing language.
- **Find the critics** — what objections and skepticism appeared in comments on related posts? These become your Phase 6 prep material.

**Output a brief research summary:**
- Top 3 prior HN posts on this topic (title, points, date, key comment themes)
- The gap this content can fill
- Recency risk: clear / caution / wait
- Community vocabulary notes

### 2. Content Analysis

Read the target content (file, URL, repo, or whatever the user provides). Extract:

- **Title and description** — what does this thing call itself?
- **Structure** — sections, chapters, components, architecture
- **Hard data points** — benchmarks, measurements, percentages, timings, comparisons to known tools, original experiments
- **Differentiators** — what makes this different from existing resources on the same topic? (Informed by Phase 1 research)
- **Interactive or visual elements** — demos, visualizations, playgrounds, live examples (e.g., D3.js charts, Three.js scenes, interactive diagrams, CLI demos, playground links)

Then determine:

- **Core insight** — one sentence capturing the single most non-obvious thing this content reveals or enables
- **Quotable facts** — list 3-5 specific numbers, surprising findings, or counter-intuitive results

**Tier classification (informed by Phase 1 landscape):**

| Tier | Criteria | Action |
|------|----------|--------|
| **S** | Fills the gap identified in Phase 1 + has hard numbers or original work + offers something you can't get elsewhere | Proceed to Phase 3 |
| **B** | Solid content but the gap is unclear, no unique data, or similar posts already performed well recently | Warn user. Suggest 2-3 specific angle-sharpening strategies based on what Phase 1 revealed is missing from the conversation |

**IMPORTANT:** If Tier B, do NOT proceed without user confirmation. Present the landscape research and let the user decide whether to sharpen or proceed anyway.

### 3. Title Engineering

Generate **5 title candidates** using these proven HN patterns:

| Pattern | Template | When to use |
|---------|----------|-------------|
| **A — Builder narrative** | "I built X to solve Y — here's what actually mattered" | You created something and learned non-obvious lessons |
| **B — Counterintuitive hook** | "X is slower/harder/weirder than you think — here's why" | Your data contradicts common assumptions |
| **C — Show HN** | "Show HN: X — [one-line value prop]" | Live demo URL + source repo both exist |
| **D — Technical dissection** | "X internals: how Y actually works under the hood" | Deep explanatory content on a system people use but don't understand |
| **E — Comparison** | "X vs Y: what the benchmarks don't tell you" | Original comparative analysis with data |

**Ranking criteria (score each 1-5):**

1. **Specificity** — vague = death. "How GPUs work" loses to "Why your GPU hits 70% utilization on FFN projections"
2. **Curiosity gap** — creates a question without clickbait
3. **Depth signal** — implies the author went deep, not wide
4. **Length** — must be under 80 characters
5. **Differentiation** — does this title signal something the Phase 1 prior posts didn't cover?

**Anti-patterns — reject any title that:**
- Uses "Ultimate guide" / "Everything you need to know" / "Complete guide"
- Contains exclamation marks
- Uses superlatives without data ("blazing fast", "revolutionary", "game-changing")
- Reads like a startup launch or product announcement
- Is generic enough to be a Wikipedia article title
- Overlaps too closely with a recent high-performing HN post title from Phase 1

Present the ranked list with a recommended pick and one-sentence reasoning for the top choice.

### 4. Opening Hook

The first 3 sentences decide whether anyone reads further. They must accomplish:

1. **Sentence 1:** State a real, specific problem or question
2. **Sentence 2:** Deliver a surprising insight or number
3. **Sentence 3:** Signal credibility — what you built, measured, or dissected

**Banned openers:**
- "I've been working on..."
- "I recently..."
- "As a developer..."
- "I'm excited to share..."
- Any sentence without a concrete claim

**Draft 2 variants:**

- **Variant A (technical-first):** Lead with the data point or technical finding
- **Variant B (narrative-first):** Lead with the problem you were trying to solve

Each variant MUST contain at least one hard number or specific technical claim from Phase 2.

**Example of a good hook (for an inference article):**
> "Most discussions focus on KV cache, but in our profiling FFN projections dominated bandwidth (~70% utilization) while attention was ~16%. Fixing that doubled throughput. Here's the breakdown of what actually moved the needle."

**Example of a good hook (for a developer tool):**
> "Our test suite took 47 minutes. The bottleneck wasn't the tests — it was the process spawning. Replacing fork() with a persistent worker pool cut it to 3 minutes. Here's the architecture."

### 5. Post Body

Follow this skeleton strictly. Every section earns its place or gets cut. Not all sections apply to every content type — skip what doesn't fit.

1. **Problem** (2-3 sentences) — What question does this answer? Why should an engineer care right now?
2. **Why existing resources fail** (2-3 sentences) — What's missing from what's already out there? (Use Phase 1 research to be specific)
3. **What this covers / does** (bullet list) — Concrete sections/features/topics, not vague promises
4. **What surprised me** (2-3 sentences) — The non-obvious finding or design decision. This is the intellectual bait
5. **Hard numbers** (if applicable) — Benchmarks, measurements, comparisons. Bullet list or mini-table
6. **Interactive / demo elements** (if applicable) — Call out anything the reader can play with: "You can drag X to see Y change", "Try it in the playground at Z"
7. **Tradeoffs and limitations** (1-2 sentences) — What this does NOT cover or handle. Intellectual honesty signals
8. **Link** — The URL, clean and bare. No tracking parameters

**Constraints:**
- Total length: **150-250 words**. Longer = skipped
- No exclamation marks anywhere
- No "I'm excited to share"
- Use "I" sparingly — focus on the content, not the author
- Concrete > abstract at every decision point
- Write like an engineer talking to peers at a whiteboard, not a marketer announcing a product

### 6. Comment Prep

#### 6a. First Author Comment

Draft a comment to post immediately after submission. This seeds the thread and frames the discussion.

- Add context NOT in the post body — motivation, build process, what's next
- Invite specific discussion: "I'm curious whether others have seen X in production" or "Would love to hear how this compares to your experience with Y"
- Keep under 150 words

#### 6b. Anticipated Questions

Using the critic patterns from Phase 1 research AND the content analysis, identify the **5 most likely questions or attacks** from HN commenters. Common patterns:

1. "How is this different from [well-known resource]?" (use Phase 1 prior posts to predict which ones)
2. "Your methodology is flawed because..."
3. "This doesn't cover [adjacent topic]"
4. "Why didn't you use [alternative approach]?"
5. (One domain-specific objection based on the content's actual claims)

Pre-draft a response for each:

- **Engineer tone:** metrics, specifics, links to sources
- **Acknowledge valid criticism directly** — never deflect
- **If the criticism is wrong**, correct with data not attitude
- Keep each response **under 100 words**
- Every reply adds credibility. Bad reply: "Thanks!" Good reply: "Yeah — we saw that too. In our case, QKV was ~54% bandwidth utilization, but FFN gate/up hit ~72%, which surprised us."

### 7. Pre-Launch Checklist

Output a checklist with pass/fail for each item:

```
- [ ] URL loads correctly and is not paywalled
- [ ] Title is under 80 characters
- [ ] Post body is 150-250 words
- [ ] At least one hard number in the opening hook
- [ ] No exclamation marks anywhere in title, body, or comments
- [ ] No superlatives without supporting data
- [ ] Demo/interactive elements called out (if content has them)
- [ ] First author comment drafted
- [ ] 5 pre-drafted responses ready
- [ ] Tradeoffs/limitations acknowledged in body
- [ ] Title does not overlap with recent HN front-page post (from Phase 1)
- [ ] Landscape research completed — gap is clear
```

**Timing recommendation:**
- **Optimal:** Tuesday-Thursday, 8:00-10:00 AM US Pacific
- **Acceptable:** Monday or Friday, same window
- **Avoid:** Weekends, US holidays, days with major tech news (Apple events, big acquisitions, etc.)

**Seed strategy:**
- Identify 2-3 communities or individuals who would genuinely find this valuable
- Subreddits, Discord servers, Twitter/X accounts — for organic sharing, not vote manipulation
- Frame as "I just published this, curious if it's clear" — genuinely seeking feedback

### 8. Failure Protocol

Most posts fail. Even good ones. This is normal.

**If < 10 points after 2 hours:**

1. Do NOT delete the post
2. Wait **minimum 2-3 weeks** before resubmitting
3. Pick a different title from the Phase 3 ranked list
4. Sharpen the opening hook — lead with a harder number or more specific claim
5. Consider reframing: if original was Pattern D (dissection), try Pattern B (counterintuitive)
6. **Maximum 3 submission attempts** for the same content

**Between attempts:**
- Share on 1-2 other platforms (relevant subreddit, Twitter/X) to test which angle gets engagement
- Use the response to calibrate the next HN title
- Re-run Phase 1 research to check if the landscape has shifted

## Wrap Up

Present the complete **Launch Package** using this exact structure:

```
## HN Launch Package: [Content Title]

### Landscape Research
- Top prior HN posts:
  1. "[title]" — X points, [date] — [key theme]
  2. "[title]" — X points, [date] — [key theme]
  3. "[title]" — X points, [date] — [key theme]
- Gap this content fills: [one sentence]
- Recency risk: clear / caution / wait
- Community vocabulary: [key terms]

### Content Analysis
- Core insight: [one sentence]
- Tier: S / B
- Quotable facts:
  1. ...
  2. ...
  3. ...

### Title Options (ranked)
1. **[Recommended]** ...  (score: X/5)
2. ... (score: X/5)
3. ... (score: X/5)
4. ... (score: X/5)
5. ... (score: X/5)

### Opening Hook
**Variant A (technical-first):**
[3 sentences]

**Variant B (narrative-first):**
[3 sentences]

### Post Body (ready to paste)
[Full text, 150-250 words]

### First Author Comment
[Ready to paste]

### Pre-drafted Responses
**Q1: [anticipated question]**
A: [response]

**Q2: [anticipated question]**
A: [response]

**Q3: [anticipated question]**
A: [response]

**Q4: [anticipated question]**
A: [response]

**Q5: [anticipated question]**
A: [response]

### Pre-Launch Checklist
- [x] / [ ] ...

### Timing
- Recommended window: [specific date range and time]
- Seed targets: [2-3 communities]

### Failure Protocol
- Backup title: [from ranked list]
- Reframe strategy: [which pattern to try next]
```

**IMPORTANT:** The launch package must be complete and ready to execute. Every field filled, every response drafted, every checkbox evaluated. No placeholders, no "TBD".
