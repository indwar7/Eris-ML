"""
Reference solution for "The Order of Discovery" — GPU (MPS) cross-encoder.

Exact logic of solution.ipynb cells 9-11: a fine-tuned cross-encoder
(SciBERT) over ordered subsection PAIRS. For every training paper, every
ordered pair (i, j) of its subsections becomes one training example: does i
precede j (label 1) or not (label 0), input = "title_i . text_i" paired with
"title_j . text_j" as a sentence-pair classification. At test time, every
pair of a paper's slots is scored, each slot accumulates its total
"precedes" probability across all pairs, and slots are sorted by that
aggregate win-score into the predicted order (same decode as the CPU model).

Uses MPS (Apple GPU) since no CUDA device is present on this machine; the
notebook's own fallback logic only defines a CPU smoke-test budget, so the
real training budget (steps/batch/seq-len) is used here explicitly for MPS
rather than falling back to the tiny CPU budget.
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKBONE = "allenai/scibert_scivocab_uncased"


def papers(df):
    out = {}
    for r in df.itertuples(index=False):
        n = int(r.n_slots)
        slots = "ABCDE"[:n]
        out[r.paper_id] = dict(
            slots=slots,
            titles=[getattr(r, f"slot_{L}_title") or "" for L in slots],
            texts=[getattr(r, f"slot_{L}_text") or "" for L in slots],
            truth=getattr(r, "true_order", None),
        )
    return out


def item(g, i):
    return f"{g['titles'][i]} . {g['texts'][i][:900]}"


class PairDS(Dataset):
    def __init__(self, P):
        self.rows = []
        for g in P.values():
            pos = {s: k for k, s in enumerate(g["truth"])}
            s = g["slots"]
            for i, j in itertools.permutations(range(len(s)), 2):
                self.rows.append((item(g, i), item(g, j), int(pos[s[i]] < pos[s[j]])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        return self.rows[k]


def pos_acc(pred, true):
    return sum(p == t for p, t in zip(pred, true)) / len(true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/public/train.csv")
    ap.add_argument("--test", default="dataset/public/test.csv")
    ap.add_argument("--answers", default="dataset/private/answers.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=4000, help="notebook's intended A10G budget")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  max_len={args.max_len}  batch={args.bs}  steps={args.steps}", file=sys.stderr)

    torch.manual_seed(0)
    np.random.seed(0)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    TR, TE = papers(train), papers(test)

    tok = AutoTokenizer.from_pretrained(BACKBONE)

    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), padding=True, truncation="longest_first",
                  max_length=args.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc

    ds = PairDS(TR)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"{len(ds)} ordered training pairs", file=sys.stderr)

    model.train()
    step = 0
    while step < args.steps:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = F.cross_entropy(model(**batch).logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % max(1, args.steps // 10) == 0:
                print(f"step {step}/{args.steps}  loss {loss.item():.4f}", file=sys.stderr)
            if step >= args.steps:
                break
    model.eval()
    print("training done", file=sys.stderr)

    @torch.no_grad()
    def decode_ce(P, bs=32):
        out = {}
        for p, g in P.items():
            s = g["slots"]
            n = len(s)
            idx = list(itertools.permutations(range(n), 2))
            A = [item(g, i) for i, _ in idx]
            C = [item(g, j) for _, j in idx]
            probs = []
            for k in range(0, len(A), bs):
                enc = tok(A[k:k + bs], C[k:k + bs], padding=True, truncation="longest_first",
                          max_length=args.max_len, return_tensors="pt")
                enc = {kk: v.to(device) for kk, v in enc.items()}
                probs.append(model(**enc).logits.float().softmax(-1)[:, 1].cpu().numpy())
            probs = np.concatenate(probs)
            win = np.zeros(n)
            for (i, j), pr in zip(idx, probs):
                win[i] += pr
            out[p] = "".join(s[k] for k in np.argsort(-win))
        return out

    preds = decode_ce(TE)
    pd.DataFrame({"paper_id": list(preds), "predicted_order": list(preds.values())}
                ).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(preds)} rows", file=sys.stderr)

    try:
        from grade import grade
        sub = pd.read_csv(args.out)
        ans = pd.read_csv(args.answers)
        print(f"local score: {grade(sub, ans):.4f}", file=sys.stderr)
    except Exception as e:
        print(f"(skipped local scoring: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
