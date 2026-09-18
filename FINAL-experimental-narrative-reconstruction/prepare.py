"""
prepare(raw, public, private) -> None

Builds the public/private split for Experimental Narrative Reconstruction.

TASK SHAPE
----------
Each item is ONE PAPER. Its Results subsections are shipped in a shuffled order
with opaque slot ids; the solver returns the slots in the order the authors
wrote them. Ground truth is a permutation. One item = one paper, so the split
is paper-disjoint by construction.

Design
------
- Slot ids ("A", "B", "C", ...) are assigned AFTER shuffling and carry no
  positional information. The raw corpus stores subsections in author order;
  that order is the private answer and is never shipped in test.csv.
- Titles are shipped with their subsection text: they are part of what the
  authors wrote, and a title like "X regulates endocytosis of Y" is exactly the
  kind of claim whose logical position a solver must reason about.
- The training split ships the SAME structure plus the answer, so a solver can
  learn ordering directly. Train and test papers are disjoint.
- `field` (the paper's discipline) is published as a feature.
- Figure/table numbers and explicit sequencing words are already masked
  upstream, so the answer cannot be recovered by reading "Fig. 1" or "First".
"""
import csv
import json
import random
import shutil
import string
from collections import Counter
from pathlib import Path

SEED = 20260913
TEST_FRACTION = 0.25
SLOT_LETTERS = string.ascii_uppercase
MAX_SLOTS = 5   # cap so no slot column is sparse; papers with more keep their first 5 (author order)


def _find_raw_file(raw: Path, name: str) -> Path:
    for cand in (raw / name, raw / "raw" / name):
        if cand.is_file():
            return cand
    hits = sorted(raw.rglob(name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under {raw}")


def _shuffle_paper(paper, rng):
    """Return (rows_in_shipped_order, true_order_as_slot_string).

    Papers with more than MAX_SLOTS subsections contribute their FIRST
    MAX_SLOTS in author order -- a valid ordering instance in its own right --
    so every slot column is populated in most rows and no paper is discarded.
    """
    subs = paper["subsections"][:MAX_SLOTS]
    n = len(subs)
    perm = list(range(n))
    rng.shuffle(perm)                       # perm[k] = original index shown at position k
    slots = SLOT_LETTERS[:n]
    rows = []
    for k, orig in enumerate(perm):
        sub = subs[orig]
        rows.append({"slot": slots[k], "title": sub["title"], "text": sub["text"]})
    # true order: for original index 0..n-1, which slot holds it?
    slot_of_orig = {orig: slots[k] for k, orig in enumerate(perm)}
    truth = "".join(slot_of_orig[i] for i in range(n))
    return rows, truth


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    with _find_raw_file(raw, "papers.jsonl").open(encoding="utf8") as fh:
        papers = [json.loads(l) for l in fh if l.strip()]
    rng.shuffle(papers)
    n_test = max(1, int(round(len(papers) * TEST_FRACTION)))
    test_p, train_p = papers[:n_test], papers[n_test:]

    def write_split(plist, prefix, path, with_answer):
        """ONE ROW PER PAPER. paper_id is the unique key and the join key to
        answers.csv. Subsections are packed into fixed columns slot_A_title,
        slot_A_text, ... slot_H_text (blank for papers with fewer slots), so
        every public row has a unique id and the shape is flat."""
        cols = ["paper_id", "field", "n_slots"]
        for L in SLOT_LETTERS[:MAX_SLOTS]:
            cols += [f"slot_{L}_title", f"slot_{L}_text"]
        if with_answer:
            cols.append("true_order")
        truths = {}
        with path.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i, p in enumerate(plist):
                pid = f"{prefix}_{i:05d}"
                rows, truth = _shuffle_paper(p, rng)
                truths[pid] = truth
                row = {"paper_id": pid, "field": p["field"], "n_slots": len(rows)}
                for r in rows:
                    row[f"slot_{r['slot']}_title"] = r["title"]
                    row[f"slot_{r['slot']}_text"] = r["text"]
                if with_answer:
                    row["true_order"] = truth
                w.writerow(row)
        return truths

    write_split(train_p, "train", public / "train.csv", with_answer=True)
    test_truths = write_split(test_p, "test", public / "test.csv", with_answer=False)

    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "true_order"])
        w.writeheader()
        for pid, t in test_truths.items():
            w.writerow({"paper_id": pid, "true_order": t})

    # sample submission: slots in the shipped (shuffled) order, i.e. identity guess
    with (public / "sample_submission.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "predicted_order"])
        w.writeheader()
        for pid, t in test_truths.items():
            w.writerow({"paper_id": pid, "predicted_order": SLOT_LETTERS[:len(t)]})

    sizes = Counter(len(t) for t in test_truths.values())
    stats = {
        "n_train_papers": len(train_p), "n_test_papers": len(test_p),
        "paper_disjoint": True,
        "n_train_subsections": sum(min(p["n_subsections"], MAX_SLOTS) for p in train_p),
        "n_test_subsections": sum(min(p["n_subsections"], MAX_SLOTS) for p in test_p),
        "max_slots_shipped": MAX_SLOTS,
        "slots_per_paper_test": dict(sorted(sizes.items())),
        "field_distribution": dict(Counter(p["field"] for p in papers).most_common(12)),
        "metric": "mean per-paper position accuracy (fraction of subsections at their true position)",
        # a random permutation -- and the shipped shuffled order is one -- places
        # each subsection correctly with probability 1/n, so chance = mean(1/n)
        "chance_level": round(sum(1 / len(t) for t in test_truths.values()) / len(test_truths), 4),
        "random_order_expected_score": "chance_level",
        "identity_order_expected_score": "chance_level",
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    v = Path(__file__).parent / "dataset" / "validate_submission.py"
    if v.exists():
        shutil.copyfile(v, public / "validate_submission.py")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    prepare(Path("dataset/raw"), Path("dataset/public"), Path("dataset/private"))
