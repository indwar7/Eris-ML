"""
Data pipeline for the bird-song phrase-prediction task.

Responsibilities:
  1. Parse train.json ({seq_id, symbols}) / test.json ({item_id, context}).
  2. Build a closed phrase vocabulary + a per-bird "dialect" id vocabulary
     (inferred from seq_id/item_id prefixes when no explicit bird field is
     given — see `infer_bird_id`).
  3. Expand each training sequence into (left-context -> next-token)
     examples for next-token prediction, left-truncated to max_context_len.
  4. Fit a smoothed variable-order (interpolated Kneser-Ney) Markov model
     directly on the token-id sequences — this is the non-neural half of
     the final ensemble.
  5. Provide a torch Dataset + collate_fn that left-pads variable-length
     contexts and returns tensors ready for the Transformer.

Design note on padding direction: contexts are LEFT-padded (pad tokens
before the real tokens) so that position `-1` is always the most recent
real token, which keeps the "predict next token" position fixed at the
final sequence index regardless of context length — simpler than tracking
per-example lengths at inference time.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

class Vocab:
    def __init__(self, tokens: list, specials: list):
        self.itos = list(specials) + [t for t in tokens if t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tok, unk_tok):
        return self.stoi.get(tok, self.stoi[unk_tok])

    def encode_seq(self, toks, unk_tok):
        return [self.encode(t, unk_tok) for t in toks]


def build_phrase_vocab(sequences: list, specials: list) -> Vocab:
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    # Sort by frequency desc then lexicographic for determinism.
    tokens = sorted(counter.keys(), key=lambda t: (-counter[t], str(t)))
    return Vocab(tokens, specials)


# Matches a leading alphabetic-prefix + digit-run (e.g. "bird017"), ignoring
# whatever separator/suffix structure follows (e.g. "_seq_0000", "-bout-004").
_BIRD_PREFIX_RE = re.compile(r"^([A-Za-z]+\d+)")


def infer_bird_id(seq_or_item_id: str) -> str:
    """Best-effort extraction of a bird/individual identifier from an id string.

    Real datasets in this family typically encode bird identity as an id
    prefix, e.g. "bird017_bout_004" -> "bird017". If the id doesn't match a
    recognizable pattern, the whole id is treated as its own bird bucket
    (degrades gracefully to "no sharing," never crashes).
    """
    m = _BIRD_PREFIX_RE.match(seq_or_item_id)
    if m:
        return m.group(1)
    return seq_or_item_id


# --------------------------------------------------------------------------
# Raw data loading
# --------------------------------------------------------------------------

@dataclass
class RawExample:
    context: list       # list of token ids, chronological order
    target: int          # token id to predict (None for test-time)
    bird_id: int          # integer bird/dialect id


def load_train_json(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("sequences", [raw]))
    return raw


def load_test_json(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("items", [raw]))
    return raw


# --------------------------------------------------------------------------
# Interpolated Kneser-Ney variable-order Markov model
# --------------------------------------------------------------------------

def _new_ctx_counter_dict():
    """Module-level factory (not a lambda) so KneserNeyBackoff pickles cleanly."""
    return defaultdict(Counter)


def _new_ctx_int_dict():
    """Module-level factory (not a lambda) so KneserNeyBackoff pickles cleanly."""
    return defaultdict(int)


class KneserNeyBackoff:
    """Variable-order Markov chain with modified Kneser-Ney smoothing.

    Why this over a plain n-gram: a plain fixed-order MLE table assigns
    zero probability to any unseen (context, next) pair, which is fatal at
    test time against novel long contexts. Kneser-Ney recursively backs off
    from order `k` to order `k-1` using a *continuation* count (how many
    distinct contexts precede a symbol) rather than raw frequency, which is
    what makes it good at estimating the probability of symbols in contexts
    it has never seen fully spelled out — precisely the long-tail regime a
    small discrete birdsong vocabulary with finite training bouts lives in.

    Fit per-bird AND globally; at query time we back off bird-specific ->
    global if the bird-specific table lacks the context entirely, which
    handles birds/dialects with very little individual training data.
    """

    def __init__(self, vocab_size: int, max_order: int, discount: float = 0.75):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.discount = discount
        # counts[order][context_tuple][next_id] = count, order 0..max_order-1 (context length)
        self.counts = defaultdict(_new_ctx_counter_dict)
        # continuation counts: for order>0, count of distinct left-extensions of (context[1:], next)
        self.continuation = defaultdict(_new_ctx_counter_dict)
        self.context_totals = defaultdict(_new_ctx_int_dict)
        self._fitted = False

    def fit(self, token_sequences: list) -> "KneserNeyBackoff":
        for seq in token_sequences:
            n = len(seq)
            for i in range(n):
                target = seq[i]
                max_ctx = min(self.max_order, i)
                for order in range(0, max_ctx + 1):
                    ctx = tuple(seq[i - order:i])
                    self.counts[order][ctx][target] += 1
                    self.context_totals[order][ctx] += 1
        # Precompute continuation counts (distinct preceding contexts) per order>=1,
        # used for the lower-order smoothed distribution in the KN recursion.
        for order in range(1, self.max_order + 1):
            seen_pairs = set()
            for ctx, next_counts in self.counts[order].items():
                shorter_ctx = ctx[1:]
                for next_id in next_counts:
                    key = (shorter_ctx, next_id)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        self.continuation[order - 1][shorter_ctx][next_id] += 1
        self._fitted = True
        return self

    def _prob(self, context: tuple, order: int) -> torch.Tensor:
        """Recursive interpolated KN probability distribution over the vocab, for a
        context truncated to `order` tokens. Returns a dense (vocab_size,) tensor.
        """
        if order == 0:
            # Unigram level: back off to continuation-count-based distribution,
            # falling back to uniform if totally unseen.
            cont = self.continuation[0][()]
            total = sum(cont.values())
            if total == 0:
                return torch.full((self.vocab_size,), 1.0 / self.vocab_size)
            probs = torch.zeros(self.vocab_size)
            for tok, c in cont.items():
                probs[tok] = c / total
            return probs

        ctx = context[-order:] if order > 0 else ()
        counts = self.counts[order].get(ctx)
        total = self.context_totals[order].get(ctx, 0)

        lower = self._prob(context, order - 1)

        if total == 0:
            return lower

        d = min(self.discount, total)  # guard: discount can't exceed total mass
        n_distinct_next = len(counts)
        lam = (d * n_distinct_next) / total  # backoff weight, standard KN formula

        probs = lower * lam
        for tok, c in counts.items():
            probs[tok] += max(c - d, 0.0) / total
        return probs

    def predict_log_probs(self, context: list) -> torch.Tensor:
        """Log-probabilities over the vocab given a (possibly long) left context.
        Only the last `max_order` tokens matter to the recursion.
        """
        ctx = tuple(context[-self.max_order:]) if self.max_order > 0 else ()
        probs = self._prob(ctx, min(self.max_order, len(ctx)))
        probs = probs.clamp_min(1e-9)
        probs = probs / probs.sum()
        return probs.log()


def fit_kn_models(train_examples_by_bird: dict, vocab_size: int, max_order: int,
                   discount: float) -> dict:
    """Fit one global KN model plus one per-bird KN model.

    Returns {"__global__": KneserNeyBackoff, bird_id_0: KneserNeyBackoff, ...}
    """
    models = {}
    all_seqs = []
    for bird_id, seqs in train_examples_by_bird.items():
        all_seqs.extend(seqs)
        model = KneserNeyBackoff(vocab_size, max_order, discount).fit(seqs)
        models[bird_id] = model
    models["__global__"] = KneserNeyBackoff(vocab_size, max_order, discount).fit(all_seqs)
    return models


# --------------------------------------------------------------------------
# Torch Dataset
# --------------------------------------------------------------------------

class PhrasePredictionDataset(Dataset):
    """One example = (left context window, bird id, target next-token id).

    MASK/UNK handling: any raw symbol not seen in the training vocabulary
    collapses to <UNK> rather than crashing or silently dropping the
    sequence — required because test-time contexts can contain symbols
    rare enough to be absent from a train/val split.
    """

    def __init__(self, examples: list, pad_id: int, max_context_len: int):
        self.examples = examples
        self.pad_id = pad_id
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ctx = ex.context[-self.max_context_len:]
        return {
            "context": ctx,
            "bird_id": ex.bird_id,
            "target": ex.target,
        }


def collate_fn(batch: list, pad_id: int, max_context_len: int):
    """Left-pad contexts to the batch max length (capped at max_context_len).

    A boolean attention/padding mask is returned (True = real token) so the
    model can correctly ignore PAD positions in both self-attention and the
    final pooled representation.
    """
    lengths = [len(b["context"]) for b in batch]
    max_len = min(max(lengths), max_context_len)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    pad_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)  # True where PAD
    bird_ids = torch.zeros(len(batch), dtype=torch.long)
    targets = torch.full((len(batch),), -100, dtype=torch.long)  # ignore_index default

    for i, b in enumerate(batch):
        ctx = b["context"][-max_len:]
        L = len(ctx)
        input_ids[i, max_len - L:] = torch.tensor(ctx, dtype=torch.long)
        pad_mask[i, :max_len - L] = True
        bird_ids[i] = b["bird_id"]
        if b["target"] is not None:
            targets[i] = b["target"]

    return {
        "input_ids": input_ids,
        "pad_mask": pad_mask,
        "bird_ids": bird_ids,
        "targets": targets,
        "raw_contexts": [b["context"] for b in batch],  # kept for the KN branch (variable length, unpadded)
    }


def build_examples(sequences_by_id: dict, bird_of_seq: dict, phrase_vocab: Vocab,
                    bird_vocab: Vocab, unk_token: str, min_context_len: int) -> list:
    """Expand each (seq_id -> token sequence) into one example per position
    i >= min_context_len, predicting symbols[i] from symbols[:i].
    """
    examples = []
    for seq_id, symbols in sequences_by_id.items():
        ids = phrase_vocab.encode_seq(symbols, unk_token)
        bird_str = bird_of_seq[seq_id]
        bird_id = bird_vocab.encode(bird_str, bird_vocab.itos[0])
        for i in range(min_context_len, len(ids)):
            examples.append(RawExample(context=ids[:i], target=ids[i], bird_id=bird_id))
    return examples
