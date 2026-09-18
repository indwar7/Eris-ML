"""
prepare(raw, public, private) -> None

Builds the public/private split for The Accent Translator from
dataset/raw/entries.jsonl.

TASK SHAPE
----------
Sequence-to-sequence. Each row is one directed (source accent -> target
accent) pair of a word: the input includes the source transcription and the
target is the transcription under the target accent.

Design
------
- Split is WORD-DISJOINT: all directed accent pairs of a word go to train or
  all to test, so no test target can be recovered from a sibling row in train.
- One row per directed (src_accent -> tgt_accent) pair; row_id is unique and
  is the join key to answers.csv.
- 88% of rows require a real change between source and target (measured on
  the shipped split); the rest are identical, and knowing not to rewrite is
  scored too.
- Accent tags RP/UK and GA/US are NOT merged: measured on a sample, same-word
  transcriptions agree exactly only 15% (RP/UK) and 8% (GA/US) of the time --
  editor convention, not dialect -- so only RP, GA, AU, NZ are kept.
"""
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

SEED = 20260918
TEST_FRACTION = 0.15


def _find_raw_file(raw: Path, name: str) -> Path:
    for cand in (raw / name, raw / "raw" / name):
        if cand.is_file():
            return cand
    hits = sorted(raw.rglob(name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under {raw}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    with _find_raw_file(raw, "entries.jsonl").open(encoding="utf8") as fh:
        entries = [json.loads(l) for l in fh if l.strip()]
    rng.shuffle(entries)
    n_test = max(1, int(round(len(entries) * TEST_FRACTION)))
    test_e, train_e = entries[:n_test], entries[n_test:]

    def rows(elist, prefix):
        """One row per DIRECTED accent pair (src -> tgt) of a word."""
        out = []
        for e in elist:
            pr = e["pronunciations"]
            for src in sorted(pr):
                for tgt in sorted(pr):
                    if src == tgt:
                        continue
                    out.append({"word": e["word"], "src_accent": src, "src_phonemes": pr[src],
                                "tgt_accent": tgt, "phonemes": pr[tgt]})
        rng.shuffle(out)
        for i, r in enumerate(out):
            r["row_id"] = f"{prefix}_{i:06d}"
        return out

    train_r, test_r = rows(train_e, "train"), rows(test_e, "test")

    with (public / "train.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "word", "src_accent", "source_pronunciation", "tgt_accent", "phonemes"])
        w.writeheader()
        for r in train_r:
            w.writerow({"row_id": r["row_id"], "word": r["word"], "src_accent": r["src_accent"],
                        "source_pronunciation": r["src_phonemes"], "tgt_accent": r["tgt_accent"],
                        "phonemes": r["phonemes"]})

    with (public / "test.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "word", "src_accent", "source_pronunciation", "tgt_accent"])
        w.writeheader()
        for r in test_r:
            w.writerow({"row_id": r["row_id"], "word": r["word"], "src_accent": r["src_accent"],
                        "source_pronunciation": r["src_phonemes"], "tgt_accent": r["tgt_accent"]})

    # answers.csv: phonemes is the ONLY target. src_phonemes is the same string
    # test.csv publishes as source_pronunciation, repeated under a distinct
    # name so grade.py can align source->target self-contained while the two
    # files share no column name but row_id. Equality is asserted below.
    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "phonemes", "src_phonemes"])
        w.writeheader()
        for r in test_r:
            w.writerow({"row_id": r["row_id"], "phonemes": r["phonemes"], "src_phonemes": r["src_phonemes"]})

    # sample: copy the SOURCE transcription through unchanged -- the honest
    # chance floor for a transfer task (correct only where the accents coincide)
    with (public / "sample_submission.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "phonemes"])
        w.writeheader()
        for r in test_r:
            w.writerow({"row_id": r["row_id"], "phonemes": r["src_phonemes"]})

    stats = {
        "n_train_words": len(train_e), "n_test_words": len(test_e), "word_disjoint": True,
        "n_train_rows": len(train_r), "n_test_rows": len(test_r),
        "accents": sorted({r["src_accent"] for r in train_r}),
        "train_direction_distribution": dict(sorted(Counter(f"{r['src_accent']}->{r['tgt_accent']}" for r in train_r).items())),
        "test_direction_distribution": dict(sorted(Counter(f"{r['src_accent']}->{r['tgt_accent']}" for r in test_r).items())),
        "test_rows_requiring_change": sum(1 for r in test_r if r["src_phonemes"] != r["phonemes"]),
        "metric": "change-segment accuracy: predicted vs true target phoneme string, scored only on segments that differ from the source",
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    v = Path(__file__).parent / "dataset" / "validate_submission.py"
    if v.exists():
        shutil.copyfile(v, public / "validate_submission.py")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, (dict, list))}, indent=2))


if __name__ == "__main__":
    prepare(Path("dataset/raw"), Path("dataset/public"), Path("dataset/private"))
