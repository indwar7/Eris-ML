"""Build the raw corpus for Accent-Conditioned Grapheme-to-Phoneme Generation.

Source: English Wiktionary database dump, https://dumps.wikimedia.org/enwiktionary/
Licence: CC BY-SA 4.0 -- confirmed via the MediaWiki siteinfo/rightsinfo API
("Creative Commons Attribution-Share Alike 4.0").

What is fetched
---------------
For each English headword, the {{IPA|en|...|a=<ACCENT>}} templates in its
Pronunciation section. The accent tag and the broad (slash-delimited)
transcription are both the archive's own structured values, written by
Wiktionary editors under the site's pronunciation policy:

    * {{IPA|en|/ˈkæt/|a=RP}}
    * {{IPA|en|/ˈkæt/|a=GA}}
    * {{IPA|en|/ˈkɛt/|a=NZ}}

No annotation is created for this dataset and no content is model-generated.

The task this corpus supports
-----------------------------
Given a SPELLING and a TARGET ACCENT, generate the phoneme sequence. The same
word maps to different sequences under different accents (rhoticity, the
TRAP/BATH split, vowel shifts), so the accent is a genuine conditioning input:
a model that ignores it is measurably capped, and a model that uses it must
learn accent-specific correspondences rather than one fixed G2P mapping.

Why only four accents
---------------------
Wiktionary uses several tags for overlapping dialects (RP/UK, GA/US/GenAm).
Measured on a sample, RP-vs-UK transcriptions of the SAME word agree exactly
only 15% of the time and GA-vs-US 8% -- these are editor-style differences,
not dialect differences, and merging them would ship label noise. We keep the
four tags that name genuinely distinct dialects and are dense enough: RP, GA,
AU, NZ.

Normalisation
-------------
Slashes, primary/secondary stress, length marks and syllable dots are stripped
so the target is the phoneme sequence alone; editor conventions on stress and
syllabification vary and are not the object of the task.

Output: raw/entries.jsonl, raw/accent_meta.csv, raw/meta.json, raw/ATTRIBUTION.txt
"""
import argparse, bz2, collections, csv, json, random, re, sys, urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

DUMP = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
UA = {"User-Agent": "Mozilla/5.0 (research dataset build)"}
FETCH_DATE = "2026-09-18"
KEEP = {"RP", "GA", "AU", "NZ"}
IPA = re.compile(r"\{\{IPA\|en\|([^}]+)\}\}")
NEXTLANG = re.compile(r"\n==[^=]")
WORD = re.compile(r"[a-z]{3,}")
STRIP = re.compile(r"[/ˈˌ.ːˑ\s()]")


def normalise(ipa):
    return STRIP.sub("", ipa)


def english_section(wikitext):
    if "==English==" not in wikitext:
        return None
    eng = wikitext.split("==English==", 1)[1]
    m = NEXTLANG.search(eng)
    return eng[:m.start()] if m else eng


def stream_pages(max_bytes):
    resp = urllib.request.urlopen(urllib.request.Request(DUMP, headers=UA), timeout=3600)
    dec = bz2.BZ2Decompressor()
    buf = b""
    read = 0
    while max_bytes <= 0 or read < max_bytes:
        chunk = resp.read(8_000_000)
        if not chunk:
            break
        read += len(chunk)
        try:
            buf += dec.decompress(chunk)
        except (EOFError, OSError):
            break
        while True:
            i = buf.find(b"<page>")
            j = buf.find(b"</page>")
            if i == -1 or j == -1 or j < i:
                break
            blob = buf[i:j + 7]
            buf = buf[j + 7:]
            try:
                el = ET.fromstring(blob.decode("utf8", "ignore"))
            except ET.ParseError:
                continue
            if el.findtext("ns") != "0":
                continue
            yield el.findtext("title") or "", el.findtext("revision/text") or ""
        if read % 200_000_000 < 8_000_000:
            print(f"  ... {read/1e6:.0f} MB", file=sys.stderr, flush=True)
    resp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/raw")
    ap.add_argument("--max-mb", type=int, default=0, help="0 = whole dump")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    by_word = {}
    scanned = 0
    for title, text in stream_pages(a.max_mb * 1_000_000):
        if not WORD.fullmatch(title):
            continue
        eng = english_section(text)
        if not eng:
            continue
        scanned += 1
        acc = {}
        for m in IPA.finditer(eng):
            parts = m.group(1).split("|")
            tag = next((p[2:] for p in parts if p.startswith("a=")), None)
            trans = [p for p in parts if p.startswith("/") and p.endswith("/")]
            if not tag or not trans:
                continue
            tag = tag.split(",")[0].strip()
            if tag in KEEP and tag not in acc:
                ph = normalise(trans[0])
                if 2 <= len(ph) <= 40:
                    acc[tag] = ph
        if len(acc) >= 2:                 # the conditioning must matter: >=2 accents
            by_word[title] = acc

    if len(by_word) < 500:
        sys.exit(f"only {len(by_word)} multi-accent words -- raise --max-mb")

    words = sorted(by_word)
    rng.shuffle(words)
    with (out / "entries.jsonl").open("w", encoding="utf8") as f:
        for w in words:
            f.write(json.dumps({"word": w, "pronunciations": by_word[w]}, ensure_ascii=False) + "\n")

    acc_count = collections.Counter(a for d in by_word.values() for a in d)
    with (out / "accent_meta.csv").open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=["accent", "n_words"]); wr.writeheader()
        for k, v in sorted(acc_count.items()):
            wr.writerow({"accent": k, "n_words": v})

    meta = {"source": DUMP, "fetch_date": FETCH_DATE, "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "seed": a.seed, "english_entries_scanned": scanned,
            "n_words": len(by_word), "n_pairs": sum(len(d) for d in by_word.values()),
            "accents": sorted(KEEP), "accent_distribution": dict(sorted(acc_count.items())),
            "words_by_accent_count": dict(sorted(collections.Counter(len(d) for d in by_word.values()).items())),
            "filters": {"headword": "lowercase a-z, >=3 chars", "min_accents_per_word": 2,
                        "phoneme_length": [2, 40], "accents_kept": sorted(KEEP)},
            "normalisation": "slashes, stress marks, length marks, syllable dots, parentheses removed"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8")
    (out / "ATTRIBUTION.txt").write_text(
        "English Wiktionary (https://en.wiktionary.org), Wikimedia Foundation\n"
        f"Obtained from the official database dump:\n  {DUMP}\nFetched: {FETCH_DATE}\n\n"
        "LICENCE\nCreative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)\n"
        "https://creativecommons.org/licenses/by-sa/4.0/\n"
        "Confirmed via the MediaWiki siteinfo API (action=query&meta=siteinfo&siprop=rightsinfo):\n"
        '  {"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",\n'
        '   "text": "Creative Commons Attribution-Share Alike 4.0"}\n\n'
        "Commercial use is permitted with attribution and share-alike.\n\n"
        "ATTRIBUTION\nTranscriptions are authored by Wiktionary contributors; per-entry\n"
        "histories are at en.wiktionary.org/wiki/<word>.\n\n"
        "CONTENT\nAccent tags and IPA transcriptions are the archive's own structured\n"
        "template values. No annotation was created for this dataset and no content\n"
        "is model-generated.\n", encoding="utf8")
    print(json.dumps({k: meta[k] for k in ("english_entries_scanned", "n_words", "n_pairs",
                                           "accent_distribution", "words_by_accent_count")}, indent=2))


if __name__ == "__main__":
    main()
