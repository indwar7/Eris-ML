"""Builds solution.ipynb for the SHIPPED per-paper layout and executes every
cell in a shared namespace so the notebook carries real outputs.

v2 notes:
- The v1 notebook read a one-row-per-subsection layout (columns slot, title,
  text) that prepare.py does not produce; it had never been executed. This
  version reads the shipped layout (slot_A_title ... slot_E_text, one row per
  paper) and is executed end to end by this script.
- The metric is per-paper position accuracy (see grade.py). Validation in the
  notebook uses the same function.
- The cross-encoder cell is device-aware: on CUDA it trains for MAX_STEPS=4000
  (the intended A10G budget); on CPU -- which is how this script executes it
  -- it runs a short smoke budget so that the code path is proven to run, and
  says so in its output. The GRADED reference in config.yaml remains the
  CPU TF-IDF pairwise model (submission 1), which this script executes fully.

Usage:  python build_notebook.py          (run from the task root)
"""
import io
import contextlib
import traceback
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
out_path = ROOT / "solution.ipynb"

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# The Order of Discovery — Reference Solution

Puts a paper's shuffled Results subsections back in the authors' order. Three
submissions of increasing sophistication, each validated on held-out **whole
papers** with the challenge metric — mean per-paper **position accuracy**
(fraction of subsections at exactly their true position; chance = 1/n).

Data layout: **one row per paper**, subsections in columns `slot_A_title,
slot_A_text, … slot_E_title, slot_E_text`; columns beyond `n_slots` are blank.
""")

code("""
from pathlib import Path
import itertools, random
import numpy as np, pandas as pd

DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/experimental-narrative-reconstruction/public"))
            if (p / "test.csv").exists())
OUT = Path("./working"); OUT.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test  = pd.read_csv(DATA / "test.csv")

def papers(df):
    \"\"\"paper_id -> dict(slots, titles, texts, truth). Slots are the shipped
    (shuffled) letters; truth is the authors' order as a slot string.\"\"\"
    out = {}
    for r in df.itertuples(index=False):
        n = int(r.n_slots); slots = "ABCDE"[:n]
        out[r.paper_id] = dict(
            slots=slots,
            titles=[getattr(r, f"slot_{L}_title") or "" for L in slots],
            texts=[getattr(r, f"slot_{L}_text") or "" for L in slots],
            truth=getattr(r, "true_order", None),
        )
    return out

TR, TE = papers(train), papers(test)
print("train papers", len(TR), "| test papers", len(TE))
print("slots per paper (train):", train.n_slots.value_counts().sort_index().to_dict())
g = next(iter(TR.values()))
print("\\none paper -- shipped order, truth =", g["truth"])
for s, t in zip(g["slots"], g["titles"]):
    print(f"  [{s}] {t[:72]}")
""")

md("""
## Metric, validation split, submission helper

Position accuracy per paper, averaged. Chance is `1/n` per paper (≈0.22 over
the test split). Validation holds out **whole papers** — splitting one
paper's subsections across fit and validation would let a model memorise the
paper's narrative.
""")

code("""
def pos_acc(pred, true):
    return sum(p == t for p, t in zip(pred, true)) / len(true)

def score(pred_dict, P):
    return float(np.mean([pos_acc(pred_dict[p], g["truth"]) for p, g in P.items()]))

rng = random.Random(0)
ids = list(TR); rng.shuffle(ids)
VA  = {p: TR[p] for p in ids[:len(ids) // 5]}
FIT = {p: TR[p] for p in ids[len(ids) // 5:]}
print(f"fit on {len(FIT)} papers, validate on {len(VA)} held-out papers")
print("chance on val (mean 1/n):        ", f"{np.mean([1/len(g['slots']) for g in VA.values()]):.3f}")
print("shipped-order baseline on val:   ", f"{score({p: g['slots'] for p, g in VA.items()}, VA):.3f}")

VAL = {}   # submission name -> validation score; the final cell ships the best

def submit(pred_dict, name, val_score):
    pd.DataFrame({"paper_id": list(pred_dict), "predicted_order": list(pred_dict.values())}
                ).to_csv(OUT / f"{name}.csv", index=False)
    VAL[name] = val_score
""")

md("""
## Submission 1 — pairwise TF-IDF logistic (title + text), aggregate-win decode

For every ordered pair of subsections in a training paper, the feature is the
difference of their TF-IDF vectors and the label is "does the first come
before the second". At test time every pair is scored, each subsection
accumulates its win-probabilities, and the paper is decoded by sorting on
that total. This is the **graded reference** in `config.yaml`.
""")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

def txt(g):   return [t + " . " + x for t, x in zip(g["titles"], g["texts"])]
def title(g): return list(g["titles"])

def pairs(P, featfn, vec):
    X, y = [], []
    for g in P.values():
        pos = {s: i for i, s in enumerate(g["truth"])}; s = g["slots"]
        V = vec.transform(featfn(g))
        for i, j in itertools.combinations(range(len(s)), 2):
            d = V[i] - V[j]
            X.append(d);  y.append(int(pos[s[i]] < pos[s[j]]))
            X.append(-d); y.append(int(pos[s[j]] < pos[s[i]]))
    return vstack(X), np.array(y)

def decode(P, models):
    \"\"\"models: list of (featfn, vec, clf, weight). Aggregate-win decode.\"\"\"
    out = {}
    for p, g in P.items():
        s = g["slots"]; n = len(s); win = np.zeros(n)
        for featfn, vec, clf, w in models:
            V = vec.transform(featfn(g))
            for i, j in itertools.combinations(range(n), 2):
                pr = clf.predict_proba(V[i] - V[j])[0, 1]
                win[i] += w * pr; win[j] += w * (1 - pr)
        out[p] = "".join(s[k] for k in np.argsort(-win))
    return out

vec1 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000,
                       sublinear_tf=True).fit(sum((txt(g) for g in TR.values()), []))
X, y = pairs(FIT, txt, vec1)
m1 = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
v1 = score(decode(VA, [(txt, vec1, m1, 1.0)]), VA)
print("val position accuracy:", f"{v1:.3f}")

Xf, yf = pairs(TR, txt, vec1)
m1f = LogisticRegression(max_iter=1000, C=1.0).fit(Xf, yf)
p1 = decode(TE, [(txt, vec1, m1f, 1.0)])
submit(p1, "submission_1_tfidf_pairwise", v1); print("submission 1 written")
""")

md("""
## Submission 2 — add a title-only model and ensemble

Titles compress the claim ("X regulates endocytosis of Y"). A second pairwise
model on titles alone, ensembled with the first by summing win-scores.
""")

code("""
vec2 = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=50000,
                       sublinear_tf=True).fit(sum((title(g) for g in TR.values()), []))
X2, y2 = pairs(FIT, title, vec2)
m2 = LogisticRegression(max_iter=1000, C=1.0).fit(X2, y2)
print("title-only val:", f"{score(decode(VA, [(title, vec2, m2, 1.0)]), VA):.3f}")
v2 = score(decode(VA, [(txt, vec1, m1, 1.0), (title, vec2, m2, 1.0)]), VA)
print("ensemble val:  ", f"{v2:.3f}")

X2f, y2f = pairs(TR, title, vec2)
m2f = LogisticRegression(max_iter=1000, C=1.0).fit(X2f, y2f)
p2 = decode(TE, [(txt, vec1, m1f, 1.0), (title, vec2, m2f, 1.0)])
submit(p2, "submission_2_ensemble", v2); print("submission 2 written")
""")

md("""
## Submission 3 (final) — fine-tuned cross-encoder over subsection pairs

A bi-encoder embeds each subsection alone and cannot see whether one claim
*presupposes* another. A cross-encoder reads both texts jointly. SciBERT is
fine-tuned on every ordered pair from the training papers to predict "A comes
before B", then decoded by aggregate win-score.

**Device-aware budget.** On the challenge's A10G this trains for
`MAX_STEPS = 4000`. This notebook was *executed on CPU* to prove the code
path end to end, so it ran a short smoke budget — the printed score is that
smoke run's, not the GPU result. Set `MAX_STEPS` back to 4000 on GPU.
""")

code("""
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "allenai/scibert_scivocab_uncased"
if DEVICE == "cuda":
    MAX_LEN, BS, LR, MAX_STEPS = 384, 16, 2e-5, 4000     # intended A10G budget
else:
    MAX_LEN, BS, LR, MAX_STEPS = 192, 8, 2e-5, 30        # CPU smoke budget (code-path proof only)
print(f"device={DEVICE}  max_len={MAX_LEN}  batch={BS}  steps={MAX_STEPS}")
torch.manual_seed(0); np.random.seed(0)
tok = AutoTokenizer.from_pretrained(BACKBONE)

def item(g, i):   # title carries the claim; keep it first, truncate body
    return f"{g['titles'][i]} . {g['texts'][i][:900]}"

class PairDS(Dataset):
    def __init__(self, P):
        self.rows = []
        for g in P.values():
            pos = {s: k for k, s in enumerate(g["truth"])}; s = g["slots"]
            for i, j in itertools.permutations(range(len(s)), 2):
                self.rows.append((item(g, i), item(g, j), int(pos[s[i]] < pos[s[j]])))
    def __len__(self): return len(self.rows)
    def __getitem__(self, k): return self.rows[k]

def collate(batch):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), padding=True, truncation="longest_first",
              max_length=MAX_LEN, return_tensors="pt")
    enc["labels"] = torch.tensor(y)
    return enc

ds = PairDS(TR)
dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate)
model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=2).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
print(f"{len(ds)} ordered training pairs")
""")

code("""
model.train(); step = 0
while step < MAX_STEPS:
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        loss = F.cross_entropy(model(**batch).logits, labels)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if step % max(1, MAX_STEPS // 10) == 0: print(f"step {step}/{MAX_STEPS}  loss {loss.item():.4f}")
        if step >= MAX_STEPS: break
model.eval(); print("training done")
""")

code("""
@torch.no_grad()
def decode_ce(P, bs=32):
    out = {}
    for p, g in P.items():
        s = g["slots"]; n = len(s)
        idx = list(itertools.permutations(range(n), 2))
        A = [item(g, i) for i, _ in idx]; C = [item(g, j) for _, j in idx]
        probs = []
        for k in range(0, len(A), bs):
            enc = tok(A[k:k+bs], C[k:k+bs], padding=True, truncation="longest_first",
                      max_length=MAX_LEN, return_tensors="pt")
            enc = {kk: v.to(DEVICE) for kk, v in enc.items()}
            probs.append(model(**enc).logits.float().softmax(-1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs)
        win = np.zeros(n)
        for (i, j), pr in zip(idx, probs): win[i] += pr
        out[p] = "".join(s[k] for k in np.argsort(-win))
    return out

v3 = score(decode_ce(VA), VA)
print("cross-encoder val position accuracy:", f"{v3:.3f}",
      "(CPU smoke budget -- not the GPU number)" if DEVICE == "cpu" else "")
p3 = decode_ce(TE)
submit(p3, "submission_3_cross_encoder", v3); print("submission 3 written")
""")

md("""
## Final submission: the best-validated of the three

Selected on held-out whole-paper validation, never on test. On the A10G the
cross-encoder is expected to win; on a CPU smoke run it will not, and the
TF-IDF pairwise model (the graded reference) ships instead. Then a structural
check with `validate_submission.py`, which needs no answers.
""")

code("""
import shutil, subprocess, sys
best = max(VAL, key=VAL.get)
print("validation scores:", {k: round(v, 3) for k, v in VAL.items()})
print("shipping:", best)
shutil.copyfile(OUT / f"{best}.csv", OUT / "submission.csv")
v = DATA / "validate_submission.py"
if v.exists():
    print(subprocess.run([sys.executable, str(v), str(OUT / "submission.csv"), str(DATA / "test.csv")],
                         capture_output=True, text=True).stdout)
sub = pd.read_csv(OUT / "submission.csv"); print(sub.head()); print("rows:", len(sub), "| test papers:", len(TE))
""")

# ---- assemble, execute, save ----
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
ns = {"__name__": "__main__"}
import os
os.chdir(ROOT)
for i, cell in enumerate(cells):
    if cell.cell_type != "code":
        continue
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(cell.source, f"<cell {i}>", "exec"), ns)
    except Exception:
        buf.write(traceback.format_exc())
        cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=buf.getvalue())]
        nb.cells = cells
        nbf.write(nb, out_path)
        raise SystemExit(f"cell {i} FAILED -- see notebook output")
    out = buf.getvalue()
    if out:
        cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=out)]
    cell.execution_count = i
nb.cells = cells
nbf.write(nb, out_path)
print(f"\nwrote {out_path}, {len(nb.cells)} cells, all executed successfully")
