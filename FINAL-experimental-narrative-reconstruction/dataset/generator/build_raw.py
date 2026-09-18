"""Build the raw corpus for Experimental Narrative Reconstruction.

Source: PLOS (Public Library of Science) research articles, fetched via the
public PLOS Search API and article-file endpoint.
Licence: CC BY 4.0 -- every PLOS research article carries the Creative Commons
Attribution licence in its own <license> block.

The task this corpus supports
-----------------------------
A Results section is not a list; it is a CHAIN OF EXPERIMENTS. Each titled
subsection answers a question the previous one raised: "we first showed X is
expressed at the synapse, so we next asked whether X binds Y, and having
found that it does, we tested whether the binding is required for Z". The
order is logical, not stylistic.

We ship each paper's Results subsections SHUFFLED and ask the solver to recover
the authors' original order. Ground truth is the sequence the authors
themselves wrote, recorded in the archive -- not an annotation created for
this dataset, and not a model output.

Why this is hard, and why it is not solvable by surface cues
------------------------------------------------------------
Every Results subsection reports statistics, so numeric density -- the cue
that partially gives away IMRaD move classification -- carries no ordering
information. Position words ("first", "next", "finally") are masked. What
remains is the scientific logic: which claim presupposes which. That requires
reading the content, not counting its features.

What is fetched
---------------
Full JATS XML per article. We locate the top-level Results section, take its
titled subsections in document order, and keep papers with 4-8 subsections
(enough to make ordering non-trivial, few enough to be tractable). Each
subsection's text is its title plus its paragraphs, concatenated.

Output: raw/papers.jsonl, raw/paper_meta.csv, raw/meta.json, raw/ATTRIBUTION.txt
"""
import argparse, collections, csv, json, random, re, sys, time, urllib.request
from pathlib import Path

SEARCH = ("https://api.plos.org/search?q=doc_type:full+AND+article_type:%22Research+Article%22"
          "&rows={rows}&start={start}&fl=id,journal,subject&wt=json")
ARTICLE = "https://journals.plos.org/plosone/article/file?id={doi}&type=manuscript"
UA = {"User-Agent": "Mozilla/5.0 (research dataset build; contact via dataset card)"}
FETCH_DATE = "2026-09-13"

RESULTS_RE = re.compile(r"^(results|findings|principal findings)$", re.I)
MIN_SUBSECTIONS, MAX_SUBSECTIONS = 4, 8
MIN_SUB_CHARS, MAX_SUB_CHARS = 300, 6000

_TAG = re.compile(r"<[^>]+>")
_XREF = re.compile(r"<xref[^>]*>.*?</xref>", re.S | re.I)
_CITE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")

# explicit sequencing language that names position outright
_ORDER = re.compile(
    r"\b(first(ly)?|second(ly)?|third(ly)?|fourth|fifth|next|then|subsequently|"
    r"finally|lastly|initially|previously|above|below|earlier|later|"
    r"having (shown|established|found|demonstrated)|as (shown|described|noted) "
    r"(above|earlier|previously)|we (next|then|first|further) )\b", re.I)
_FIGREF = re.compile(r"\b(fig(ure)?s?\.?|tables?)\s*[\dS]+[a-z]?\b", re.I)


def clean(html):
    t = _XREF.sub(" ", html)
    t = _TAG.sub(" ", t)
    t = _CITE.sub(" ", t)
    t = _FIGREF.sub("<FIG>", t)        # figure numbers are monotone in order -> mask
    t = _ORDER.sub("<SEQ>", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", t).strip()


def fetch(url, timeout=60, retries=4):
    for k in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                          timeout=timeout).read().decode("utf8", "ignore")
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(3 * (k + 1))
    return None


def section_spans(xml):
    """Yield (title, inner, depth) for every <sec>, in DOCUMENT order.

    depth is the nesting level of the section itself (0 = top level). We
    record the depth at open time; an earlier version read len(stack) after
    the pop, which mislabelled direct children and silently dropped most
    Results subsections (2 of 6 on the reference article).
    """
    stack = []                      # (start_offset, depth_at_open)
    out = []
    for m in re.finditer(r"<sec\b[^>]*>|</sec>", xml):
        if m.group(0).startswith("<sec"):
            stack.append((m.end(), len(stack)))
        else:
            if not stack:
                continue
            start, depth = stack.pop()
            inner = xml[start:m.start()]
            t = re.match(r"\s*<title>(.*?)</title>", inner, re.S)
            if t:
                out.append((start, _TAG.sub("", t.group(1)).strip(), inner, depth))
    out.sort(key=lambda x: x[0])   # closing order != opening order; restore doc order
    for _, title, inner, depth in out:
        yield title, inner, depth


def results_subsections(xml):
    """Return the Results section's DIRECT child subsections in document order."""
    body = re.search(r"<body>(.*?)</body>", xml, re.S)
    if not body:
        return None
    body = body.group(1)
    # find the top-level Results section span
    for title, inner, depth in section_spans(body):
        if depth == 0 and RESULTS_RE.match(title):
            subs = []
            for st, sinner, sdepth in section_spans(inner):
                if sdepth != 0:
                    continue                       # only direct children
                paras = re.findall(r"<p>(.*?)</p>", sinner, re.S)
                text = clean(" ".join(paras))
                if MIN_SUB_CHARS <= len(text) <= MAX_SUB_CHARS:
                    subs.append({"title": clean(st), "text": text})
            return subs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/raw")
    ap.add_argument("--articles", type=int, default=6000)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--start", type=int, default=0, help="DOI page offset (parallel workers)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    dois, start = [], a.start
    while len(dois) < a.articles:
        js = fetch(SEARCH.format(rows=200, start=start))
        if not js:
            break
        try:
            docs = json.loads(js)["response"]["docs"]
        except Exception:
            break
        if not docs:
            break
        for d in docs:
            if "/annotation/" not in d["id"]:
                dois.append((d["id"], d.get("subject", [])))
        start += 200
        print(f"  collected {len(dois)} dois", file=sys.stderr, flush=True)
        time.sleep(a.sleep)
    dois = dois[:a.articles]

    papers = []
    ok = 0
    for n, (doi, subjects) in enumerate(dois, 1):
        xml = fetch(ARTICLE.format(doi=doi))
        if not xml:
            continue
        ok += 1
        subs = results_subsections(xml)
        if not subs or not (MIN_SUBSECTIONS <= len(subs) <= MAX_SUBSECTIONS):
            continue
        field = "unknown"
        if subjects:
            parts = [x for x in subjects[0].split("/") if x]
            field = parts[1] if len(parts) >= 2 else (parts[0] if parts else field)
        papers.append({"doi": doi, "field": field, "n_subsections": len(subs),
                       "subsections": subs})      # subs are in AUTHOR ORDER
        if n % 200 == 0:
            print(f"  {n}/{len(dois)} fetched, {len(papers)} usable papers",
                  file=sys.stderr, flush=True)
        time.sleep(a.sleep)

    if len(papers) < 40:
        sys.exit(f"only {len(papers)} usable papers -- raise --articles (min 40)")

    rng.shuffle(papers)
    with (out / "papers.jsonl").open("w", encoding="utf8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (out / "paper_meta.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["doi", "field", "n_subsections"])
        w.writeheader()
        for p in papers:
            w.writerow({k: p[k] for k in ("doi", "field", "n_subsections")})

    dist = collections.Counter(p["n_subsections"] for p in papers)
    meta = {"source": "https://api.plos.org/search + journals.plos.org article XML",
            "fetch_date": FETCH_DATE, "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "seed": a.seed, "articles_requested": len(dois), "articles_fetched": ok,
            "n_papers": len(papers),
            "subsections_per_paper": dict(sorted(dist.items())),
            "total_subsections": sum(p["n_subsections"] for p in papers),
            "field_distribution": dict(collections.Counter(p["field"] for p in papers).most_common(15)),
            "filters": {"results_subsections_min": MIN_SUBSECTIONS,
                        "results_subsections_max": MAX_SUBSECTIONS,
                        "subsection_chars": [MIN_SUB_CHARS, MAX_SUB_CHARS]},
            "masking": "figure/table numbers -> <FIG> (monotone in order); explicit "
                       "sequencing words (first/next/finally/having shown/...) -> <SEQ>; "
                       "bracketed citations removed"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8")
    (out / "ATTRIBUTION.txt").write_text(
        "PLOS (Public Library of Science) research articles\n"
        "Search API:   https://api.plos.org/search\n"
        "Article XML:  https://journals.plos.org/plosone/article/file?id=<doi>&type=manuscript\n"
        f"Fetched: {FETCH_DATE}\n\n"
        "LICENCE\n"
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "Every PLOS research article carries the CC BY licence in its own <license>\n"
        "block: \"This is an open-access article distributed under the terms of the\n"
        "Creative Commons Attribution License, which permits unrestricted use,\n"
        "distribution, and reproduction in any medium, provided the original author\n"
        "and source are credited.\"\n\n"
        "Commercial use is permitted with attribution.\n\n"
        "ATTRIBUTION\n"
        "Each paper is identified by DOI in the raw corpus.\n\n"
        "CONTENT\n"
        "The ground truth is the authors' own ordering of their Results\n"
        "subsections, as published. No annotation was created for this dataset\n"
        "and no content is model-generated.\n", encoding="utf8")
    print(json.dumps({k: meta[k] for k in
                      ("articles_fetched", "n_papers", "subsections_per_paper", "total_subsections")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
