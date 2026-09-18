# Dataset Description — Wiktionary Phoneme Strings With Their Own Editor-Tagged Accent

## Overview

Real pronunciation transcriptions, paired with the accent their own editors
said they represent.

Each record is one English headword from the English Wiktionary together with
its IPA transcription in each of up to four accents — Received Pronunciation,
General American, Australian, New Zealand — as written by Wiktionary
contributors in the entry's Pronunciation section.

Nothing here is annotated, generated, or synthesised for this dataset. The
transcriptions and their accent tags are the archive's own structured
template values:

```
* {{IPA|en|/ˈkæt/|a=RP}}
* {{IPA|en|/ˈkæt/|a=GA}}
* {{IPA|en|/ˈkɛt/|a=NZ}}
```

No one read a word and decided anything about it for this dataset.

## What makes the task learnable — and hard

Accent differences in English are systematic, not random. Rhoticity, the
TRAP–BATH split, the NZ centralised KIT vowel, the AU fronted GOAT vowel:
each is a rule that applies to a class of words, and the class is largely
predictable from spelling. A model can learn *which* correspondences shift
with accent from words where both forms are attested, then apply them to
spellings it has never seen.

It is hard because the shifts interact with the ordinary irregularity of
English spelling, because editors transcribe at slightly different
granularities, and because a wrong accent assumption is wrong at every
affected segment, not just one.

### Dataset at a glance

| Property | Value |
|---|---|
| Unit | one headword with ≥2 accent transcriptions; prepared as directed (src → tgt) pairs |
| Accents | RP, GA, AU, NZ |
| Constraint | every headword has ≥2 accents attested |
| Source | English Wiktionary database dump |
| Licence | **CC BY-SA 4.0** |
| Fetch date | 2026-09-18 |
| Human annotation | None — accent tags and IPA are template values |

Counts are recorded in `raw/meta.json` and, for the prepared split, in
`public/dataset_stats.json`.

## Source

| Role | Source | Endpoint | What is taken from it |
|---|---|---|---|
| PRIMARY | English Wiktionary dump | `https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2` | headword, `{{IPA\|en\|…\|a=…}}` templates in the English section |

### Fetch method

1. Stream the article XML dump; keep mainspace pages whose title is a single
   lowercase a–z word of ≥3 letters.
2. Isolate the `==English==` section (stop at the next language header).
3. Parse every `{{IPA|en|…}}` template: take the first slash-delimited broad
   transcription and the `a=` accent tag. Keep only RP, GA, AU, NZ.
4. Keep headwords with ≥2 such accents; take one transcription per accent.
5. Normalise: remove slashes, primary/secondary stress, length marks,
   syllable dots and parentheses, so the target is the segment sequence.

### Why only four accents

Wiktionary uses several tags for overlapping dialects (RP/UK, GA/US/GenAm).
Measured on a 150 MB sample, transcriptions of the *same word* under RP and
UK agree exactly only 15% of the time, and GA vs US 8%. Those are differences
in editor convention, not dialect. Merging them would ship label noise; we
keep the four tags that name distinct dialects and are dense enough to model.

## File Structure

- `raw/entries.jsonl` — one JSON object per headword
- `raw/accent_meta.csv` — words per accent
- `raw/meta.json` — fetch provenance, filters, normalisation
- `raw/ATTRIBUTION.txt` — source, licence, citation

## Features

### `raw/entries.jsonl`

| Field | Type | Description |
|---|---|---|
| `word` | str | Headword, lowercase a–z |
| `pronunciations` | object | `{accent: phoneme_string}` for each attested accent (≥2) |

### `raw/accent_meta.csv`

| Column | Type | Description |
|---|---|---|
| `accent` | str | RP, GA, AU or NZ |
| `n_words` | int | Headwords attested in that accent |

## Intended Use

**Primary use.** Benchmarking dialect-to-dialect phoneme transfer: given a
word's transcription in one accent, produce it in another. Because every word
is attested in ≥2 accents, the corpus supplies both sides of the transfer for
every item, and the learning target is the context-sensitive rewrite between
accents rather than a spelling-to-sound mapping. Test words are unseen, so
the rules must generalise.

**Also suitable for.** Accent identification from transcription (spelling
withheld), accent-conditioned grapheme-to-phoneme, studying dialectal
correspondence rules from data, and accent-adaptation front ends for TTS.

**Not suitable for.** Speech recognition or audio work (no audio), narrow
phonetic detail (targets are broad transcriptions with stress and length
stripped), dialects outside the four included, or any claim about how a
specific speaker talks — the data records editor-standard forms, not
individuals.

## Limitations

These are properties of the shipped data, stated so results are not over-read.

**Editor conventions vary within an accent.** Two editors transcribing the same
RP word may differ in symbol choice (`/ɹ/` vs `/r/`), diphthong notation, or
granularity. Stress and length are stripped to reduce this, but segment-level
variation remains. Part of the residual error any model shows is this
inconsistency, not modelling failure; a ceiling below 1.0 is inherent.

**Accent coverage is uneven.** RP and GA are attested for far more words than
AU and NZ. Models will see fewer examples of the AU/NZ correspondences and
should be expected to do worse on them; the per-accent distribution is
published.

**Only words attested in ≥2 accents are included.** This biases toward common
and well-edited entries, which are likely the more regular ones. Rare or
technical words with a single transcription are excluded.

**Headwords are restricted to lowercase a–z.** Proper nouns, hyphenated and
multi-word entries, and words with diacritics are excluded.

**Broad transcriptions only.** Phonetic detail in square brackets is discarded.
Allophonic variation within an accent is not represented.

**Four accents is not English.** Indian, Scottish, Irish, Canadian, South
African and other Englishes are present in Wiktionary but too sparse to keep
under the ≥2-accents constraint. Results should not be generalised to them.

**Share-alike licence.** CC BY-SA 4.0 requires derivative datasets to carry the
same licence. Models trained on the data are not derivatives under Wikimedia's
reading, but redistributed data is.

## License

**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
https://creativecommons.org/licenses/by-sa/4.0/

Confirmed programmatically via the MediaWiki siteinfo API
(`action=query&meta=siteinfo&siprop=rightsinfo`), which returns:

```json
{"url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
 "text": "Creative Commons Attribution-Share Alike 4.0"}
```

Commercial use is permitted with attribution and share-alike.

**Attribution:** transcriptions are authored by Wiktionary contributors;
per-entry histories are at `en.wiktionary.org/wiki/<word>`.

Citation: Wiktionary contributors. *Wiktionary, the free dictionary.*
https://en.wiktionary.org. Retrieved from the Wikimedia database dumps,
2026-09-18.
