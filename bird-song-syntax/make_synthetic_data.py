"""
Generates a small synthetic bird-song dataset for local smoke-testing of the
pipeline (dataset.py / model.py / trainer.py / predict.py / submission.py).

Not part of the competition solution itself -- the real train.json/test.json
are provided by the challenge. This script exists purely so the code can be
exercised end-to-end without network access or the real dataset.

The synthetic generator deliberately includes the three properties called
out in the task brief:
  - per-bird dialect (each bird has its own transition-matrix bias)
  - stochastic branching (transitions are multinomial, not deterministic)
  - a long-range dependency (a phrase can only occur if a specific "trigger"
    phrase occurred earlier in the bout -- a 1st-order Markov chain cannot
    represent this, which is exactly the failure mode the brief describes)
"""

from __future__ import annotations

import json
import random

N_BIRDS = 6
N_PHRASE_TYPES = 12
N_SEQS_PER_BIRD = 120
MIN_LEN, MAX_LEN = 15, 45
SEED = 42


def make_bird_transition_matrix(rng: random.Random, n_types: int):
    matrix = []
    for _ in range(n_types):
        weights = [rng.random() ** 2 for _ in range(n_types)]  # skewed, sparse-ish
        total = sum(weights)
        matrix.append([w / total for w in weights])
    return matrix


def generate_sequence(rng: random.Random, matrix, n_types: int, trigger_phrase: int,
                       unlocked_phrase: int):
    length = rng.randint(MIN_LEN, MAX_LEN)
    seq = [rng.randrange(n_types)]
    triggered = seq[0] == trigger_phrase
    for _ in range(length - 1):
        cur = seq[-1]
        weights = list(matrix[cur])
        if unlocked_phrase < n_types:
            if triggered:
                weights[unlocked_phrase] *= 4.0
            else:
                weights[unlocked_phrase] *= 0.05
        total = sum(weights)
        weights = [w / total for w in weights]
        nxt = rng.choices(range(n_types), weights=weights, k=1)[0]
        seq.append(nxt)
        if nxt == trigger_phrase:
            triggered = True
    return [f"P{p}" for p in seq]


def main():
    rng = random.Random(SEED)
    train_rows = []
    test_rows = []

    for bird_idx in range(N_BIRDS):
        bird_id = f"bird{bird_idx:03d}"
        matrix = make_bird_transition_matrix(rng, N_PHRASE_TYPES)
        trigger_phrase = bird_idx % N_PHRASE_TYPES
        unlocked_phrase = (bird_idx * 3 + 5) % N_PHRASE_TYPES

        for seq_idx in range(N_SEQS_PER_BIRD):
            symbols = generate_sequence(rng, matrix, N_PHRASE_TYPES, trigger_phrase, unlocked_phrase)
            seq_id = f"{bird_id}_seq_{seq_idx:04d}"

            # Hold out the last symbol of ~15% of sequences as a "test" item.
            if rng.random() < 0.15 and len(symbols) > MIN_LEN:
                test_rows.append({
                    "item_id": f"{bird_id}_item_{seq_idx:04d}",
                    "context": symbols[:-1],
                    "_true_answer": symbols[-1],  # kept only for local sanity-checking, not a real field
                })
            else:
                train_rows.append({"seq_id": seq_id, "symbols": symbols})

    with open("data/train.json", "w") as f:
        json.dump(train_rows, f, indent=2)

    # Real test.json wouldn't include _true_answer; write a clean version plus
    # a separate answer key for local accuracy checking.
    clean_test = [{"item_id": r["item_id"], "context": r["context"]} for r in test_rows]
    answer_key = {r["item_id"]: r["_true_answer"] for r in test_rows}

    with open("data/test.json", "w") as f:
        json.dump(clean_test, f, indent=2)
    with open("data/test_answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)

    print(f"Wrote {len(train_rows)} train sequences, {len(clean_test)} test items.")


if __name__ == "__main__":
    main()
