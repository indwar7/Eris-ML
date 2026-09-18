"""
Reference solution for "The Accent Translator" — CPU model.

Approach: per-segment context rewrite table (submission-1/2 logic from
solution.ipynb), with backoff and no-change gating. This is the model whose
score (0.698) is published as reference_solution_score in config.yaml.

Given a source phoneme string and a (src_accent, tgt_accent) direction, learn
what each phoneme becomes in that direction from 3-segment context, backing
off to a context-free per-direction table, and finally to identity (copy
through) when no rule has ever been observed for that segment in that
direction — this also covers the ~12% of rows where nothing should change.
"""
import argparse
import collections
import csv
import sys
import unicodedata

_STRIP = set("/ˈˌ.ːˑ ()[]")


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFC", str(s)) if c not in _STRIP)


def align(a, b):
    """Levenshtein traceback -> list of (a_char|'', b_char|'') ops."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((a[i - 1], "")); i -= 1
        else:
            ops.append(("", b[j - 1])); j -= 1
    return ops[::-1]


def build_rules(rows):
    """3-segment-context table per direction: (src_accent,tgt_accent) -> {context: Counter(target_segment)}."""
    R = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for s, sp, t, tp in rows:
        a = norm(sp)
        pa = "#" + a + "#"
        k = 0
        for x, y in align(a, norm(tp)):
            if x:
                R[(s, t)][pa[k] + x + pa[k + 2]][y] += 1
                k += 1
    return R


def build_backoff(rows):
    """Context-free fallback per direction: (src_accent,tgt_accent) -> {segment: Counter(target_segment)}."""
    B = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for s, sp, t, tp in rows:
        a = norm(sp)
        for x, y in align(a, norm(tp)):
            if x:
                B[(s, t)][x][y] += 1
    return B


def apply_rules(R, B, s, sp, t):
    """context rules -> backoff -> identity (copy through)."""
    a = norm(sp)
    pa = "#" + a + "#"
    out = []
    for k, x in enumerate(a):
        c = R[(s, t)].get(pa[k] + x + pa[k + 2]) or B[(s, t)].get(x)
        out.append(c.most_common(1)[0][0] if c else x)
    return "".join(out)


def read_csv(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/public/train.csv")
    ap.add_argument("--test", default="dataset/public/test.csv")
    ap.add_argument("--answers", default="dataset/private/answers.csv",
                     help="optional; if present, prints the local score before writing")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    train = read_csv(args.train)
    test = read_csv(args.test)

    train_rows = [(r["src_accent"], r["source_pronunciation"], r["tgt_accent"], r["phonemes"]) for r in train]
    R = build_rules(train_rows)
    B = build_backoff(train_rows)

    preds = {}
    for r in test:
        preds[r["row_id"]] = apply_rules(R, B, r["src_accent"], r["source_pronunciation"], r["tgt_accent"])

    with open(args.out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "phonemes"])
        w.writeheader()
        for row_id, ph in preds.items():
            w.writerow({"row_id": row_id, "phonemes": ph})
    print(f"wrote {args.out}: {len(preds)} rows", file=sys.stderr)

    try:
        from grade import grade
        import pandas as pd
        sub = pd.read_csv(args.out, keep_default_na=False)
        ans = pd.read_csv(args.answers, keep_default_na=False)
        print(f"local score: {grade(sub, ans):.4f}", file=sys.stderr)
    except Exception as e:
        print(f"(skipped local scoring: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
