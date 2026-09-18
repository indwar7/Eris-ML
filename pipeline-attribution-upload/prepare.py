"""
prepare(raw, public, private) -> None

Builds the public/private split for the Digitization Pipeline Attribution
challenge from dataset/raw/snippets.jsonl + batches_meta.csv.

Design (see DESIGN.md for the full rationale):
- Split is BATCH-DISJOINT: test batches never appear in train. This forces a
  solver to learn a GENERAL OCR-pipeline-fingerprint notion from labeled train
  batches, rather than memorize a small fixed vocabulary of known batch IDs —
  and it blunts the "just globally cluster the whole corpus" shortcut, since
  test batches carry no train-time label a solver could look up.
- train.csv is FLAT and fully labeled (row_id, text, batch_id) — a normal
  supervised training set. row_id is an OPAQUE sequential index
  ("train_00000", ...), never the raw snippet_id: the raw snippet_id is
  formatted "<batch_id>_<lccn>_<date>_<seq>_<block_idx>" and would leak the
  label through the identifier itself if shipped verbatim.
- test.csv is BAG-STRUCTURED and unlabeled: (row_id, text), where row_id is
  "<bag_id>::<opaque per-bag sequence number>" — bag_id groups rows for
  per-bag grading, the sequence number carries no batch information. Each
  bag draws a random subset of 2-4 DISTINCT test batches and 8-14 snippets per
  chosen batch (both randomized per bag — a fixed bag shape would itself be a
  structural tell). Snippet order within a bag is shuffled so position carries
  no signal. Every test snippet is used AT MOST ONCE across the whole test set
  (see "reuse" below).
- Ground truth (private/answers.csv) is ONE ROW PER BAG: (bag_id, batch_id),
  where groups lists the opaque batch label of each snippet in test.csv's
  snippet order. Never exposed to solvers via test.csv. Grading only ever
  needs co-membership within a bag, never the label's name — see grade.py.
  submission.csv has the same shape. One row per bag (rather than per
  snippet) is what keeps a row-wise public/private leaderboard slice from
  cutting bags into fragments — see LEAK HISTORY 4.

LEAK HISTORY (four fixes, each caught by a real solver or reviewer):

  1. row_id leaked the label. An early version shipped row_id as
     "bag_id::snippet_id" (raw snippet_id, batch_id prefix and all). A
     pre-ship agent evaluation caught it: a solver scored a perfect 1.000 by
     string-splitting the answer straight out of row_id. Fixed by making the
     per-row identifier a bare sequence number with zero batch information.

  2. SOURCE-LOOKUP leak. dataset_stats.json published the full list of the 16
     held-out test batch_ids, and the task page named the source archive. A reviewer fetched those batches from the archive, exact-
     matched all 4,457 test rows against the original OCR text, and scored
     1.000 — no modeling at all. Fixed on two axes, because either alone is
     insufficient:
       (a) dataset_stats.json no longer publishes ANY batch identity, train
           or test — only counts. Batch ids are answer-adjacent metadata.
       (b) Shipped text (train and test) is a deterministic WINDOW of the
           source snippet, not the snippet verbatim (see _window below).
           Window bounds derive from a private salt. This closes EXACT
           matching: zero shipped rows are byte-identical to any archive
           region. It does NOT close substring matching against an adversary
           who already holds the source text of the test batches -- no
           signal-preserving transform of a unique document can (see
           DESIGN.md, "Review round 1"). The OCR-noise fingerprint -- the
           actual signal -- is a property of the character stream and
           survives windowing.
       (c) The split is re-drawn under a new seed (v1's test ids are known).
       (d) Every batch_id in public files is an opaque alias ("batch_NN"),
           never the archive's own name; the naming scheme and prefix set
           were themselves an enumeration aid. Real ids live in
           private/split_manifest.json.
       (e) The candidate-selection rule and skipped-batch list are withheld
           from public docs, and the problem statement + rubrics carry an
           explicit no-external-data / no-archive-lookup rule. Against an
           adversary holding the archive, the RULE is the load-bearing
           layer; (a)-(d) make the path non-trivial and cut every link the
           reviewer actually used.

  3. SNIPPET REUSE. An earlier build sampled snippets with replacement ACROSS
     bags: 4,457 test rows drew on only 2,577 distinct snippets, so 70.1% of
     rows came from a snippet that appeared elsewhere in the test set, one of
     them 7 times. A reviewer flagged that repeated examples can dominate the
     leaderboard. Fixed by redesign rather than explanation: sampling is now
     WITHOUT REPLACEMENT globally — a snippet used in any bag is retired from
     the pool, so every test row is a distinct source snippet and each
     contributes exactly once to the mean-per-bag score.

  4. BAG FRAGMENTATION BY THE LEADERBOARD SLICE (review round 2). The
     leaderboard splits answers.csv into public/private slices by ROW. With
     one answers row per snippet, every bag was cut in two and each fragment
     was scored as a bag of its own: 24 public fragments had three rows or
     fewer, where ARI gives 1.0 almost for free, and the all-singleton sample
     scored 0.048 public / 0.000 private (one solver run was exactly that).
     Fixed structurally: answers.csv and submission.csv are now one row per
     bag, so any row-wise slice keeps whole bags on one board.

  5. DIFFICULTY BAND, twice. Round 2 found platform solver runs on v2 (3-5
     groups x 3-8 snippets, reference 0.276) landed at 0.40-0.43, the bottom
     edge of the target band; v3 (3-4 x 6-10, reference 0.326) shipped as the
     fix. Round 3 found the best of three platform solver runs on v3 reached
     only 0.38 (ratio ~1.17x, below the ~1.5x ratio seen on v2/v1) -- short
     of the 0.40 floor. v4 (2-4 x 8-14, reference 0.342, shipped) is the
     second retune; see the N_GROUPS_PER_BAG / N_SNIPPETS_PER_GROUP comments
     below and DESIGN.md "Review round 3" for the full sweep.
"""
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

# v2 seed. The v1 split (seed 20260812) was reviewed with its 16 test batch
# ids published; those ids are now known outside this build, so the split is
# re-drawn rather than reused (leak fix 2c).
SEED = 20260913
# v4 bag shape (review round 3 / difficulty gate, second retune). v3 shipped
# 3-4 groups of 6-10 snippets (reference 0.326); the best of three platform
# solver runs reached only 0.38 (ratio ~1.17x, below the ~1.5x ratio seen on
# v2/v1) -- short of the 0.40 band floor. Measured on the reference
# (RF-probability embedding, per-bag agglomerative), this round swept group
# count DOWN as well as snippets-per-group up (the v3 retune had only tried
# the latter):
#     groups 3-4, per-group 6-10   0.326   (v3, shipped)
#     groups 3-4, per-group 8-14   0.313   (more snippets/group with a fixed
#     groups 3-4, per-group 8-16   0.302    group count made it WORSE, not
#     groups 3-4, per-group 10-16  0.287    better -- larger groups dilute
#                                            per-bag confidence faster than
#                                            they add signal)
#     groups 2-3, per-group 6-12   0.328
#     groups 2-3, per-group 8-12   0.339
#     groups 2-4, per-group 8-12   0.338
#     groups 2-4, per-group 8-14   0.342   <- shipped
#     groups 2-3, per-group 8-14   0.352   (rejected: 53% of bags would be
#                                            exactly 2 groups -- too close to
#                                            a binary-clustering task; same
#                                            "2-group bags are weak instances"
#                                            concern the v3 retune avoided)
#     groups 2-3, per-group 10-16  0.321
# Lowering group count helped consistently; growing snippets-per-group alone
# did not. 2-4 groups keeps a genuine 2/3/4-group spread (33/32/45 bags on
# the shipped data) rather than a narrow or fixed count.
N_GROUPS_PER_BAG = (2, 4)
N_SNIPPETS_PER_GROUP = (8, 14)
N_TEST_BAGS = 200
TEST_BATCH_FRACTION = 0.45

# Windowing (leak fix 2b). Shipped test text is a contiguous character window
# of the source snippet rather than the snippet itself, so a solver holding
# the full source archive cannot exact-match a row back to its
# batch. Bounds are derived from WINDOW_SALT, which is NOT published.
WINDOW_SALT = "pipeline-attribution/window/v2"
# Window length as a FRACTION of the source snippet. Measured on the v2 split
# (reference_solution.py, mean per-bag ARI), every setting below yields zero
# exact matches against the archive; they differ only in how much signal
# survives:
#     absolute 160-400 chars   0.191
#     keep 50-80%              0.239
#     keep 65-95%              0.270
#     keep 80-97%              0.278   <- shipped
#     keep 90-99%              0.278
#     no windowing at all      0.298   (but 100% exact-matchable)
# Windowing's job is to break byte-equality, which any trim achieves; a small
# absolute window additionally destroyed most of the OCR-noise statistics the
# task is about, so the fraction is kept high.
WINDOW_KEEP = (0.80, 0.97)
# Every snippet is windowed, including short ones: a snippet shipped whole is
# exact-matchable against the archive, and short OCR regions are exactly the
# ones a lookup attack finds cheapest to confirm. A snippet too short to
# window (< MIN_SHIPPABLE after trimming) is dropped from the corpus instead.
WINDOW_MIN_TRIM = 12            # minimum chars removed from any snippet
MIN_SHIPPABLE = 40              # matches the corpus's own minimum region length


def _find_raw_file(raw: Path, name: str) -> Path:
    """Locate a raw file whether `raw` is the raw/ dir itself or the extracted
    dataset root that contains raw/ (the platform has passed both)."""
    for cand in (raw / name, raw / "raw" / name):
        if cand.is_file():
            return cand
    hits = sorted(raw.rglob(name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under {raw} (looked in {raw}, {raw / 'raw'}, and recursively)")


def _load_snippets(raw: Path):
    path = _find_raw_file(raw, "snippets.jsonl")
    with path.open(encoding="utf8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    by_batch = {}
    for r in rows:
        by_batch.setdefault(r["batch_id"], []).append(r)
    return by_batch


def _window(snippet_id: str, text: str) -> str:
    """Return a deterministic contiguous character window of `text`.

    Leak fix 2b: the shipped row must not be byte-identical to any region of
    the public archive (substring matching is addressed by the other layers;
    see the module docstring). Offsets are derived by hashing
    (WINDOW_SALT, snippet_id), so they are stable across runs — the same
    snippet always yields the same window, keeping the build reproducible —
    but unguessable without the salt, which is never published.

    EVERY snippet is windowed, long or short. An earlier version shipped
    snippets under 220 chars whole, on the reasoning that a short region is
    weak evidence for a lookup attack; measurement killed that reasoning —
    39.3% of test rows still exact-matched the archive, which is more than
    enough to dominate a leaderboard. A snippet too short to take a real
    window has WINDOW_MIN_TRIM characters removed from a hash-chosen end
    instead, which is enough to break byte-equality; one that would fall
    below MIN_SHIPPABLE after that is returned as None and dropped.

    Windowing preserves the fingerprint: OCR noise (substitution habits, junk
    characters, broken word boundaries) is a property of the character stream,
    not of where the region starts, so a window carries the same signal the
    full snippet does.
    """
    n = len(text)
    h = hashlib.sha256(f"{WINDOW_SALT}|{snippet_id}".encode("utf8")).digest()
    r = random.Random(int.from_bytes(h[:8], "big"))
    lo_f, hi_f = WINDOW_KEEP

    length = min(int(n * r.uniform(lo_f, hi_f)), n - WINDOW_MIN_TRIM)
    if length >= MIN_SHIPPABLE:
        start = r.randint(0, n - length)
        return text[start:start + length]

    # Too short for a fractional window: trim one end so the row is not
    # byte-equal to the archive's text, provided enough of the region survives.
    if n - WINDOW_MIN_TRIM < MIN_SHIPPABLE:
        return None
    return text[WINDOW_MIN_TRIM:] if r.getrandbits(1) else text[:n - WINDOW_MIN_TRIM]


def _split_batches(by_batch, rng):
    batch_ids = sorted(by_batch.keys())
    rng.shuffle(batch_ids)
    n_test = max(1, round(len(batch_ids) * TEST_BATCH_FRACTION))
    test_batches = sorted(batch_ids[:n_test])
    train_batches = sorted(batch_ids[n_test:])
    return train_batches, test_batches


def _anonymize(batch_ids, rng):
    """Map real archive batch ids to opaque labels (leak fix 2d).

    A real archive batch id (of the form "<institution>_<word>_verNN") names
    an institution prefix and a fetchable directory on the public archive. Shipping 19 of them in
    train.csv would tell a solver the naming scheme and the prefix set, which
    is most of what an enumeration attack against the archive needs. The
    label's VALUE carries no signal for the task -- any label that is
    consistent per real batch trains the same model and grades identically --
    so the public files use "batch_00".."batch_NN" instead. The real mapping
    is kept in private/split_manifest.json for auditing.
    """
    ids = list(batch_ids)
    rng.shuffle(ids)
    return {real: f"batch_{i:02d}" for i, real in enumerate(ids)}


def _build_train(by_batch, train_batches, alias):
    # row_id is an OPAQUE sequential index, never the raw snippet_id: the raw
    # snippet_id is formatted "<batch_id>_<lccn>_<date>_<seq>_<block_idx>", so
    # shipping it verbatim would let a solver read the label straight out of
    # the identifier string with a trivial split -- caught by a real solver
    # scoring 1.000 during a pre-ship agent evaluation (leak fix 1).
    #
    # Train text is windowed the same way test text is (leak fix 2b). This is
    # not itself a leak defence -- train labels are published, so there is
    # nothing to protect -- but it keeps train and test drawn from the same
    # text-length distribution. A model trained on full-length snippets and
    # applied to windowed ones would face an avoidable domain shift.
    rows = []
    i = 0
    for b in train_batches:
        for r in by_batch[b]:
            text = _window(r["snippet_id"], r["text"])
            if text is None:
                continue
            rows.append({
                "row_id": f"train_{i:05d}",
                "text": text,
                "batch_id": alias[r["batch_id"]],
            })
            i += 1
    return rows


def _build_bags(by_batch, test_batches, rng, alias, raw_texts):
    """Build bags sampling WITHOUT REPLACEMENT across the whole test set.

    Leak fix 3: a reviewer flagged that the previous with-replacement sampling
    let 2,577 distinct snippets back 4,457 rows, so repeated examples could
    dominate the leaderboard. Each batch now keeps a shuffled pool that is
    consumed as bags are built; a snippet used once is never offered again.

    A batch whose pool runs dry is simply not available to later bags. The
    documented construction -- 2-4 distinct groups per bag, 8-14 snippets per
    group -- is ENFORCED, not merely targeted: a group that cannot reach
    N_SNIPPETS_PER_GROUP[0] snippets is not added, and a bag that cannot reach
    N_GROUPS_PER_BAG[0] groups is discarded rather than shipped. Late in the
    build this yields fewer than N_TEST_BAGS bags (the shipped count is in
    dataset_stats.json); it never yields a bag outside the documented shape.
    """
    pools = {b: rng.sample(by_batch[b], len(by_batch[b])) for b in test_batches}
    # Distinctness is enforced on the shipped TEXT, not just the snippet_id:
    # two different source regions can carry byte-identical OCR text (a
    # masthead, a date line, a standing ad), and one such collision survived
    # the first without-replacement build. A repeated string would be a free
    # co-membership hint as well as a duplicate example.
    seen_text = set()
    bags = []
    for i in range(N_TEST_BAGS):
        n_groups = rng.randint(*N_GROUPS_PER_BAG)
        available = [b for b in test_batches if pools[b]]
        if len(available) < 2:
            break
        chosen_batches = rng.sample(available, min(n_groups, len(available)))
        items = []
        groups_present = 0
        min_per_group = N_SNIPPETS_PER_GROUP[0]
        for b in chosen_batches:
            want = rng.randint(*N_SNIPPETS_PER_GROUP)
            drawn = []
            while len(drawn) < want and pools[b]:
                r = pools[b].pop()
                text = _window(r["snippet_id"], r["text"])
                # `text in raw_texts`: a window can coincide with some OTHER
                # region's full text (repeated boilerplate). Such a row would
                # be exact-matchable, so it is skipped like a duplicate.
                if text is None or text in seen_text or text in raw_texts:
                    continue
                seen_text.add(text)
                drawn.append((r["snippet_id"], text, alias[r["batch_id"]]))
            if len(drawn) < min_per_group:
                # Under-filled group (pool exhausted): do not ship it. Its
                # texts were already marked seen; they are simply unused.
                continue
            items.extend(drawn)
            groups_present += 1
        if groups_present < N_GROUPS_PER_BAG[0]:
            continue
        rng.shuffle(items)
        bags.append((None, items))
    return _balance_order(bags)


def _balance_order(bags):
    """Order bags so that any contiguous run of them is representative.

    The leaderboard slices answers.csv by row (one row per bag), so bag ORDER
    decides what each board sees. Bags are sorted by their difficulty proxy
    (group count, then size) and then spread along the file with a golden-
    ratio permutation, so every contiguous window mixes small and large,
    3-group and 4-group bags. Verified at build time: contiguous fifths of
    the shipped answers score within a few hundredths of the full set for the
    reference. bag_ids are assigned AFTER this ordering, so the id carries
    no information about the bag beyond its file position.
    """
    key = lambda b: (len(set(lab for _, _, lab in b[1])), len(b[1]))
    ranked = sorted(bags, key=key)
    n = len(ranked)
    phi = (5 ** 0.5 - 1) / 2
    slots = sorted(range(n), key=lambda i: (i * phi) % 1.0)
    ordered = [None] * n
    for rank, slot in enumerate(slots):
        ordered[slot] = ranked[rank]
    return [(f"bag_{i:04d}", items) for i, (_, items) in enumerate(ordered)]


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    by_batch = _load_snippets(raw)
    train_batches, test_batches = _split_batches(by_batch, rng)
    alias = _anonymize(train_batches + test_batches, rng)
    raw_texts = {r["text"] for rows in by_batch.values() for r in rows}

    train_rows = _build_train(by_batch, train_batches, alias)
    bags = _build_bags(by_batch, test_batches, rng, alias, raw_texts)

    with (public / "train.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "text", "batch_id"])
        w.writeheader()
        w.writerows(train_rows)

    # row_id is the genuinely unique per-row key AND is deliberately OPAQUE:
    # it encodes bag_id (needed to group rows for per-bag grading) but NEVER
    # the raw snippet_id, because snippet_id is formatted
    # "<batch_id>_<lccn>_<date>_<seq>_<block_idx>" -- shipping it (even inside
    # a compound row_id like "bag_0000::<snippet_id>") lets a solver recover
    # the true batch_id with a trivial string split, no modeling required
    # (leak fix 1). bag_id is safe to expose -- it is not the label -- but the
    # per-row suffix carries no batch information at all.
    def row_id(bag_id, seq_in_bag):
        return f"{bag_id}::{seq_in_bag}"

    with (public / "test.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "text"])
        w.writeheader()
        for bag_id, items in bags:
            for seq, (snippet_id, text, _) in enumerate(items):
                w.writerow({"row_id": row_id(bag_id, seq), "text": text})

    # answers.csv and sample_submission.csv are ONE ROW PER BAG (review round
    # 2): the leaderboard slices answers by row into public/private, and a
    # one-row-per-snippet layout let every bag be cut in two and each fragment
    # scored as its own bag (24 public fragments had <= 3 rows, where ARI is
    # nearly free). `batch_id` holds one label per snippet, whitespace-
    # separated, in the snippet order of test.csv (the integer after "::").
    with (private / "answers.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["bag_id", "batch_id"])
        w.writeheader()
        for bag_id, items in bags:
            w.writerow({"bag_id": bag_id, "batch_id": " ".join(batch_id for _, _, batch_id in items)})

    with (public / "sample_submission.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["bag_id", "batch_id"])
        w.writeheader()
        for bag_id, items in bags:
            # every snippet its own group: valid, scores exactly 0 under ARI
            w.writerow({"bag_id": bag_id, "batch_id": " ".join(str(seq) for seq in range(len(items)))})

    # Leak fix 2a: publish COUNTS, never IDENTITIES. An earlier version listed
    # all 16 held-out test batch_ids here; a reviewer used that list to fetch
    # the batches from the public source archive, exact-matched
    # every test row, and scored 1.000 without modeling anything. Batch ids
    # are answer-adjacent metadata and belong in private/, not public/.
    test_sizes = [len(items) for _, items in bags]
    stats = {
        "n_train_batches": len(train_batches),
        "n_test_batches": len(test_batches),
        "n_train_snippets": len(train_rows),
        "n_test_bags": len(bags),
        "n_test_snippets": sum(test_sizes),
        "bag_size_min": min(test_sizes) if test_sizes else 0,
        "bag_size_max": max(test_sizes) if test_sizes else 0,
        "test_snippets_are_distinct": True,
        "submission_shape": "one row per bag: bag_id, batch_id (one label per snippet, "
                            "whitespace-separated, in test.csv snippet order)",
        "note": (
            "Batch identities (train or test) are deliberately not published: "
            "they are answer-adjacent metadata. Every test row is a distinct "
            "source snippet -- no snippet appears twice in the test set."
        ),
    }
    with (public / "dataset_stats.json").open("w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    # Private-side provenance: the identities the public stats file must not
    # carry still need to live somewhere for auditing and regrading.
    with (private / "split_manifest.json").open("w", encoding="utf8") as f:
        json.dump({
            "seed": SEED,
            "window_salt": WINDOW_SALT,
            "train_batches": train_batches,
            "test_batches": test_batches,
            "batch_alias": alias,
        }, f, indent=2)

    validator_src = Path(__file__).parent / "dataset" / "validate_submission.py"
    if validator_src.exists():
        shutil.copyfile(validator_src, public / "validate_submission.py")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    prepare(Path("dataset/raw"), Path("dataset/public"), Path("dataset/private"))
