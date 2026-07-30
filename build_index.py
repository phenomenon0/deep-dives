#!/usr/bin/env python3
"""Generate index.html + sitemap.xml for deep-dives, and backfill missing OG/meta tags.

Run from the deep-dives directory:  python3 build_index.py
Add --no-meta to skip rewriting article <head> blocks.
"""

import html as H
import re
import sys
from datetime import date
from pathlib import Path

SITE = "https://femiadeniran.com"
BASE = "/deep-dives"
HERE = Path(__file__).parent

# --- curation -------------------------------------------------------------
# Superseded drafts and duplicates. Files stay on disk; they just don't ship.
EXCLUDE = {
    "inference-engine-deep-dive-blog.html",  # dupe of inference-engine-deep-dive.html, lives on /blog
    "part2-making-it-fast.html",  # superseded by inference-engine-deep-dive-2.html
    "part3-serving-at-scale.html",  # superseded by inference-engine-deep-dive-3.html
    "inference-deep-dive.html",  # early short version, superseded by the 3-part series
    "index.html",
}

# Ordered collections. Anything unlisted lands in "Everything Else".
COLLECTIONS = [
    (
        "The Efficiency Papers",
        "Nine essays on why large models are too expensive to run, and the research that keeps making them cheaper.",
        [
            "efficiency-papers-1-attention-tax.html",
            "efficiency-papers-1a-making-it-fit.html",
            "efficiency-papers-1b-making-it-serve.html",
            "efficiency-papers-2-building-it-right.html",
            "efficiency-papers-2-fewer-bits.html",
            "efficiency-papers-3-serving-the-swarm.html",
            "efficiency-papers-4-breaking-the-chain.html",
            "efficiency-papers-5-sparse-by-design.html",
            "efficiency-papers-6-beyond-attention.html",
        ],
    ),
    (
        "Inference Engines",
        "How a prompt becomes tokens, why the second token is cheaper than the first, and what breaks at ten thousand concurrent users.",
        [
            "inference-engine-deep-dive.html",
            "inference-engine-deep-dive-2.html",
            "inference-engine-deep-dive-3.html",
        ],
    ),
    (
        "Models & Machine Learning",
        "What is actually inside a model file, and what the training loop is really optimizing.",
        [
            "gguf-deep-dive.html",
            "deepseek-r1-deep-dive.html",
            "rl-deep-dive.html",
            "gnm-lab-deep-dive.html",
            "video-intelligence-deep-dive.html",
        ],
    ),
    (
        "Memory & the Machine",
        "Allocation, caching, buffering, and the numbers that lie to you.",
        [
            "allocators-deep-dive.html",
            "caches-deep-dive.html",
            "buffers-deep-dive.html",
            "buffer-hacking.html",
            "buffer-hacking-2.html",
            "floatingpoint-deep-dive.html",
            "hashtables-deep-dive.html",
            "concurrency-deep-dive.html",
            "syscalls-deep-dive.html",
        ],
    ),
    (
        "Graphics & Games",
        "Silicon that thinks in parallel, and the pipelines that turn math into pixels.",
        [
            "gpu-deep-dive.html",
            "shaders-deep-dive.html",
            "textures-deep-dive.html",
            "gamephysics-deep-dive.html",
            "gamecharacter-deep-dive.html",
            "gameserver-deep-dive.html",
            "svg-os-foundation.html",
        ],
    ),
    (
        "Networks, Protocols & Crypto",
        "What happens after send(), how to build a mesh VPN from nothing, and how to encrypt what travels over it.",
        [
            "network-deep-dive.html",
            "mesh-vpn-deep-dive.html",
            "crypto-deep-dive.html",
        ],
    ),
    (
        "Data, Text & Signal",
        "Turning bytes into meaning: parsing, compression, encoding, randomness, sound.",
        [
            "parsers-deep-dive.html",
            "compression-deep-dive.html",
            "unicode-deep-dive.html",
            "rng-deep-dive.html",
            "audio-deep-dive.html",
            "webassembly-deep-dive.html",
        ],
    ),
    (
        "Systems I Built",
        "Deep-dives into my own machinery, written the same way as the rest.",
        [
            "control-deck-deep-dive.html",
            "navigation-deep-dive.html",
        ],
    ),
]


def clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = H.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip("_").strip()


def extract(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    head = raw[: raw.find("</head>") + 7] if "</head>" in raw else raw[:6000]

    m = re.search(r"<title>(.*?)</title>", head, re.S)
    title = clean(m.group(1)) if m else path.stem

    desc = None
    m = re.search(r'<meta[^>]+name="description"[^>]*content="([^"]*)"', head)
    if m:
        desc = clean(m.group(1))
    if not desc:
        m = re.search(
            r'class="[^"]*(?:subtitle|hero-sub|tagline)[^"]*"[^>]*>(.*?)</', raw, re.S
        )
        if m:
            desc = clean(m.group(1))
    if not desc:
        m = re.search(r"<p[^>]*>(.{80,}?)</p>", raw, re.S)
        desc = clean(m.group(1)) if m else ""

    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    words = len(clean(body).split())

    return {
        "file": path.name,
        "title": title,
        "desc": desc,
        "words": words,
        "minutes": max(1, round(words / 220)),
        "raw": raw,
    }


def split_title(title: str) -> tuple[str, str]:
    """Split 'Topic — Subtitle' into (topic, subtitle) for card display."""
    for sep in (" — ", " – ", ": ", " - "):
        if sep in title:
            a, b = title.split(sep, 1)
            return a.strip(), b.strip()
    return title, ""


# --- meta backfill --------------------------------------------------------


def backfill_meta(art: dict) -> bool:
    """Inject description/OG/Twitter tags when absent. Returns True if written."""
    raw = art["raw"]
    if "og:title" in raw:
        return False
    if "</title>" not in raw:
        return False

    url = f"{SITE}{BASE}/{art['file']}"
    t = H.escape(art["title"], quote=True)
    d = H.escape(art["desc"][:300], quote=True)

    block = [""]
    if not re.search(r'<meta[^>]+name="description"', raw):
        block.append(f'  <meta name="description" content="{d}">')
    block += [
        f'  <link rel="canonical" href="{url}">',
        f'  <meta property="og:title" content="{t}">',
        f'  <meta property="og:description" content="{d}">',
        f'  <meta property="og:type" content="article">',
        f'  <meta property="og:url" content="{url}">',
        f'  <meta property="og:site_name" content="Femi Adeniran">',
        f'  <meta name="twitter:card" content="summary_large_image">',
        f'  <meta name="twitter:title" content="{t}">',
        f'  <meta name="twitter:description" content="{d}">',
        f'  <meta name="author" content="Femi Adeniran">',
    ]
    raw = raw.replace("</title>", "</title>" + "\n".join(block), 1)
    (HERE / art["file"]).write_text(raw, encoding="utf-8")
    return True


# --- page render ----------------------------------------------------------

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0a0f; --bg-card:#12121a; --border:#22222e;
  --text:#c8c8d4; --text-dim:#6a6a7e;
  --green:#00ff88; --cyan:#00ddff; --magenta:#ff00aa;
  --orange:#ff8800; --yellow:#ffdd00;
}
html{scroll-behavior:smooth}
body{
  background:var(--bg); color:var(--text);
  font:18px/1.7 'Source Serif 4',Georgia,serif;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace}
a{color:var(--cyan);text-decoration:none}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}

/* scanline */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.16) 0 1px,transparent 1px 3px);
  opacity:.4;
}

/* nav */
nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  background:rgba(10,10,15,.82);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
}
nav .wrap{display:flex;align-items:center;gap:22px;height:52px}
nav a{
  font-family:'JetBrains Mono',monospace;font-size:11px;
  text-transform:uppercase;letter-spacing:.1em;color:var(--text-dim);
}
nav a:hover{color:var(--green)}
nav .home{color:var(--green)}
nav .spacer{flex:1}

/* hero */
header{padding:132px 0 46px;border-bottom:1px solid var(--border)}
h1{
  font-family:'JetBrains Mono',monospace;font-weight:700;
  font-size:clamp(2.1rem,6vw,3.6rem);line-height:1.05;
  color:var(--green);letter-spacing:-.02em;
  text-shadow:0 0 26px rgba(0,255,136,.28);
}
.blink{animation:bl 1.1s step-end infinite;color:var(--green)}
@keyframes bl{50%{opacity:0}}
.lede{margin-top:20px;max-width:64ch;font-size:1.12rem;color:var(--text)}
.lede em{color:var(--cyan);font-style:normal}

.stats{display:flex;flex-wrap:wrap;gap:34px;margin-top:32px}
.stat b{
  display:block;font-family:'JetBrains Mono',monospace;
  font-size:1.7rem;color:var(--yellow);font-weight:600;line-height:1
}
.stat span{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  text-transform:uppercase;letter-spacing:.14em;color:var(--text-dim);
  display:block;margin-top:7px
}

/* filter */
.tools{
  position:sticky;top:52px;z-index:60;
  background:rgba(10,10,15,.94);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);padding:14px 0;
}
#q{
  width:100%;background:var(--bg-card);border:1px solid var(--border);
  color:var(--text);padding:11px 14px;border-radius:5px;
  font-family:'JetBrains Mono',monospace;font-size:13px;
}
#q:focus{outline:none;border-color:var(--green);box-shadow:0 0 0 3px rgba(0,255,136,.08)}
#q::placeholder{color:var(--text-dim)}
#count{
  font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);
  text-transform:uppercase;letter-spacing:.12em;margin-top:9px;display:block
}

/* collections */
section{padding:52px 0 8px}
.sec-head{border-left:2px solid var(--green);padding-left:16px;margin-bottom:26px}
h2{
  font-family:'JetBrains Mono',monospace;font-size:1.32rem;
  color:var(--green);letter-spacing:-.01em
}
.sec-head p{color:var(--text-dim);font-size:.97rem;margin-top:7px;max-width:70ch}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:15px}
.card{
  display:flex;flex-direction:column;
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:7px;padding:19px 19px 15px;
  transition:border-color .16s,transform .16s,background .16s;
}
.card:hover{border-color:var(--green);transform:translateY(-2px);background:#15151f}
.card h3{
  font-family:'JetBrains Mono',monospace;font-size:.97rem;
  color:var(--cyan);line-height:1.35;font-weight:600
}
.card:hover h3{color:var(--green)}
.card .sub{
  font-family:'JetBrains Mono',monospace;font-size:.72rem;
  color:var(--magenta);margin-top:5px;letter-spacing:.01em
}
.card p{font-size:.9rem;color:var(--text-dim);margin-top:11px;flex:1;line-height:1.6}
.card .meta{
  display:flex;gap:13px;margin-top:15px;padding-top:11px;
  border-top:1px solid var(--border);
  font-family:'JetBrains Mono',monospace;font-size:9.5px;
  text-transform:uppercase;letter-spacing:.11em;color:var(--text-dim)
}
.card .meta .w{color:var(--orange)}

.empty{display:none;color:var(--text-dim);padding:46px 0;font-family:'JetBrains Mono',monospace;font-size:13px}
footer{
  margin-top:66px;border-top:1px solid var(--border);padding:30px 0 56px;
  font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim)
}
footer a{color:var(--text-dim)}
footer a:hover{color:var(--green)}
@media(max-width:640px){
  header{padding:104px 0 36px}
  .stats{gap:22px}
  nav .wrap{gap:14px}
}
"""

JS = """
(function(){
  var q=document.getElementById('q'),
      cards=[].slice.call(document.querySelectorAll('.card')),
      secs=[].slice.call(document.querySelectorAll('section')),
      count=document.getElementById('count'),
      empty=document.getElementById('empty'),
      total=cards.length;
  function run(){
    var t=q.value.trim().toLowerCase(), n=0;
    cards.forEach(function(c){
      var hit=!t||c.dataset.s.indexOf(t)>-1;
      c.style.display=hit?'':'none';
      if(hit)n++;
    });
    secs.forEach(function(s){
      var vis=s.querySelectorAll('.card:not([style*="none"])').length;
      s.style.display=vis?'':'none';
    });
    count.textContent=t?(n+' of '+total+' matching "'+q.value.trim()+'"')
                       :(total+' articles');
    empty.style.display=n?'none':'block';
  }
  q.addEventListener('input',run);
  document.addEventListener('keydown',function(e){
    if(e.key==='/'&&document.activeElement!==q){e.preventDefault();q.focus();}
    if(e.key==='Escape'&&document.activeElement===q){q.value='';run();q.blur();}
  });
  run();
})();
"""


def card_html(a: dict) -> str:
    topic, subtitle = split_title(a["title"])
    search = f"{a['title']} {a['desc']}".lower().replace('"', "")
    sub = f'<div class="sub">{H.escape(subtitle)}</div>' if subtitle else ""
    blurb = a["desc"]
    if len(blurb) > 190:
        # cut on a word boundary so cards never end mid-word
        blurb = blurb[:190].rsplit(" ", 1)[0].rstrip(" ,;:—-") + "…"
    return f"""      <a class="card" href="{a["file"]}" data-s="{H.escape(search, quote=True)}">
        <h3>{H.escape(topic)}</h3>{sub}
        <p>{H.escape(blurb)}</p>
        <div class="meta"><span class="w">{a["words"]:,} words</span><span>{a["minutes"]} min</span></div>
      </a>"""


def build() -> None:
    files = {p.name: p for p in sorted(HERE.glob("*.html")) if p.name not in EXCLUDE}
    arts = {name: extract(p) for name, p in files.items()}

    if "--no-meta" not in sys.argv:
        n = sum(backfill_meta(a) for a in arts.values())
        print(f"meta backfilled: {n} articles")

    placed, sections = set(), []
    for title, blurb, names in COLLECTIONS:
        got = [arts[n] for n in names if n in arts]
        placed.update(a["file"] for a in got)
        if got:
            sections.append((title, blurb, got))

    rest = [a for n, a in arts.items() if n not in placed]
    if rest:
        sections.append(
            (
                "Everything Else",
                "Not yet filed into a collection.",
                sorted(rest, key=lambda a: -a["words"]),
            )
        )

    all_arts = [a for _, _, g in sections for a in g]
    total_words = sum(a["words"] for a in all_arts)
    total_hours = round(sum(a["minutes"] for a in all_arts) / 60)

    body = []
    for title, blurb, group in sections:
        cards = "\n".join(card_html(a) for a in group)
        body.append(f"""  <section>
    <div class="sec-head">
      <h2>{H.escape(title)}</h2>
      <p>{H.escape(blurb)}</p>
    </div>
    <div class="grid">
{cards}
    </div>
  </section>""")

    desc = (
        f"{len(all_arts)} long-form technical deep-dives by Femi Adeniran — "
        "LLM inference, GPU compute, memory systems, networking, graphics, and compilers. "
        "Interactive, first-principles, no hand-waving."
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Dives | Femi Adeniran</title>
<meta name="description" content="{H.escape(desc, quote=True)}">
<link rel="canonical" href="{SITE}{BASE}/">
<meta property="og:title" content="Deep Dives | Femi Adeniran">
<meta property="og:description" content="{H.escape(desc, quote=True)}">
<meta property="og:type" content="website">
<meta property="og:url" content="{SITE}{BASE}/">
<meta property="og:site_name" content="Femi Adeniran">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Deep Dives | Femi Adeniran">
<meta name="twitter:description" content="{H.escape(desc, quote=True)}">
<meta name="author" content="Femi Adeniran">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230a0a0f'/%3E%3Ctext x='16' y='23' font-family='monospace' font-size='20' font-weight='700' fill='%2300ff88' text-anchor='middle'%3E%3E%3C/text%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<nav><div class="wrap">
  <a class="home" href="/">FEMI ADENIRAN</a>
  <a href="/deep-dives/">Deep Dives</a>
  <a href="/blog/">Blog</a>
  <a href="/experiments/">Experiments</a>
  <span class="spacer"></span>
  <a href="https://github.com/phenomenon0">GitHub</a>
</div></nav>

<header><div class="wrap">
  <h1>Deep Dives<span class="blink">_</span></h1>
  <p class="lede">
    The average technical post takes four minutes and teaches you what a function does.
    These take forty-five and change how you think about the system underneath it.
    <em>Every abstraction unpacked, down to the hardware.</em>
  </p>
  <div class="stats">
    <div class="stat"><b>{len(all_arts)}</b><span>Articles</span></div>
    <div class="stat"><b>{total_words // 1000}k</b><span>Words</span></div>
    <div class="stat"><b>{total_hours}h</b><span>Reading</span></div>
    <div class="stat"><b>{len(sections)}</b><span>Collections</span></div>
  </div>
</div></header>

<div class="tools"><div class="wrap">
  <input id="q" type="search" placeholder="Filter by topic, title, or idea&nbsp;&nbsp;&nbsp;(press / to focus)" autocomplete="off">
  <span id="count"></span>
</div></div>

<div class="wrap">
{chr(10).join(body)}
  <p class="empty mono" id="empty">No article matches that filter.</p>
</div>

<footer><div class="wrap">
  Written by Femi Adeniran &middot;
  <a href="/">femiadeniran.com</a> &middot;
  <a href="https://github.com/phenomenon0">github.com/phenomenon0</a> &middot;
  Updated {date.today().isoformat()}
</div></footer>

<script>{JS}</script>
</body>
</html>
"""
    (HERE / "index.html").write_text(page, encoding="utf-8")

    today = date.today().isoformat()
    urls = "\n".join(
        f"  <url><loc>{SITE}{BASE}/{a['file']}</loc><lastmod>{today}</lastmod>"
        f"<changefreq>monthly</changefreq><priority>0.8</priority></url>"
        for a in all_arts
    )
    (HERE / "sitemap.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{SITE}{BASE}/</loc><lastmod>{today}</lastmod><changefreq>weekly</changefreq><priority>1.0</priority></url>
{urls}
</urlset>
""",
        encoding="utf-8",
    )

    print(
        f"index.html   -> {len(all_arts)} articles, {total_words:,} words, {len(sections)} collections"
    )
    print(f"sitemap.xml  -> {len(all_arts) + 1} urls")
    for t, _, g in sections:
        print(f"  {len(g):>2}  {t}")
    if rest:
        print("\nUNFILED (add to COLLECTIONS):")
        for a in rest:
            print(f"      {a['file']}")


if __name__ == "__main__":
    build()
