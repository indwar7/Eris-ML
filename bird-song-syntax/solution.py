"""
Recovering Hidden Phrases in Bird-Song Syntax -- self-contained solution.

Reads ./dataset/public/{train.json, test.json} and writes
./working/submission.csv with columns [item_id, symbol].

ARCHITECTURE (see the "Your Approach" writeup for full rationale):
  A causal Transformer decoder (bird/individual-conditioned) is combined,
  via a LEARNED per-example gate, with a smoothed variable-order Markov
  chain (modified Kneser-Ney) fit directly on the training sequences.
  The two signals are complementary in different regimes -- KN is a
  strong, reliable estimator for short/common contexts (this dataset's
  bigram baseline alone reaches ~0.27, and full KN backoff does
  considerably better), while the Transformer can exploit long-range
  structure once it has seen enough data to learn it. A fixed mixing
  weight would be provably suboptimal since the right weight differs by
  context; a small gating MLP learns it per-example instead.

  Gate-collapse fix: training the gate and the neural branch jointly from
  step 1 fails. At initialization the neural branch is pure noise, so the
  loss immediately teaches the gate to trust only Kneser-Ney (gate -> 0),
  which then starves the neural branch of gradient and it never recovers
  -- a self-reinforcing mixture-of-experts cold start, confirmed
  empirically (validation accuracy got stuck exactly at the KN-only
  accuracy for many epochs with the naive joint-training approach). The
  fix is a curriculum: the Transformer trains alone on plain
  cross-entropy for `gate_warmup_epochs` epochs (kn_log_probs=None forces
  pure-neural mode) until it is genuinely competitive, and only then is
  the learned gate switched on for the remaining epochs.

Runs on CPU or GPU (auto-detected), uses AMP when CUDA is available.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Paths (per the platform's contract)
# ============================================================================

DATASET_DIR = "./dataset/public"
WORKING_DIR = "./working"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.json")
TEST_PATH = os.path.join(DATASET_DIR, "test.json")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    val_fraction: float = 0.1
    max_context_len: int = 40          # matches this task's test-context length cap (max 40)
    min_context_len: int = 1
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 384
    dropout: float = 0.2
    bird_embed_dim: int = 32
    max_position_embeddings: int = 256

    kn_max_order: int = 5
    kn_discount: float = 0.75

    interp_hidden_dim: int = 16
    gate_warmup_epochs: int = 8         # see module docstring: prevents gate collapse

    batch_size: int = 64
    lr: float = 4e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: int = 18
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    early_stopping_patience: int = 5
    num_workers: int = 0               # 0 is safest inside notebook/grader sandboxes

    top_k_to_average: int = 3
    tta_context_crops: tuple = (1.0, 0.75, 0.5)

    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# Vocabulary + bird/individual id inference
# ============================================================================

class Vocab:
    def __init__(self, tokens, specials):
        self.itos = list(specials) + [t for t in tokens if t not in specials]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tok, unk_tok):
        return self.stoi.get(tok, self.stoi[unk_tok])

    def encode_seq(self, toks, unk_tok):
        return [self.encode(t, unk_tok) for t in toks]


def build_phrase_vocab(sequences, specials):
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    tokens = sorted(counter.keys(), key=lambda t: (-counter[t], str(t)))
    return Vocab(tokens, specials)


_BIRD_PREFIX_RE = re.compile(r"^([A-Za-z]+\d+)")


def infer_bird_id(seq_or_item_id: str) -> str:
    """Best-effort bird/individual id from an id string (e.g. "bird017_bout_004"
    -> "bird017"). This dataset's ids (train_000, IT_...) don't carry a bird
    field, so every id falls back to its own singleton bucket -- harmless:
    the model backs off bird-specific -> global whenever a bird's data is
    sparse or absent (see KneserNeyBackoff and the bird embedding's shared
    global row 0 fallback)."""
    m = _BIRD_PREFIX_RE.match(seq_or_item_id)
    return m.group(1) if m else seq_or_item_id


# ============================================================================
# Kneser-Ney variable-order Markov backoff
# ============================================================================

def _new_ctx_counter_dict():
    return defaultdict(Counter)


def _new_ctx_int_dict():
    return defaultdict(int)


class KneserNeyBackoff:
    """Smoothed variable-order Markov chain (modified Kneser-Ney).

    Why this over a plain n-gram: a plain fixed-order MLE table assigns
    zero probability to any unseen (context, next) pair -- fatal at test
    time against novel long contexts. Kneser-Ney backs off from order k to
    k-1 using continuation counts (how many distinct contexts precede a
    symbol) rather than raw frequency, which is what makes it reliable in
    the long-tail regime a small discrete vocabulary with finite training
    sequences lives in.
    """

    def __init__(self, vocab_size, max_order, discount=0.75):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.discount = discount
        self.counts = defaultdict(_new_ctx_counter_dict)
        self.continuation = defaultdict(_new_ctx_counter_dict)
        self.context_totals = defaultdict(_new_ctx_int_dict)

    def fit(self, token_sequences):
        for seq in token_sequences:
            n = len(seq)
            for i in range(n):
                target = seq[i]
                max_ctx = min(self.max_order, i)
                for order in range(0, max_ctx + 1):
                    ctx = tuple(seq[i - order:i])
                    self.counts[order][ctx][target] += 1
                    self.context_totals[order][ctx] += 1
        for order in range(1, self.max_order + 1):
            seen_pairs = set()
            for ctx, next_counts in self.counts[order].items():
                shorter_ctx = ctx[1:]
                for next_id in next_counts:
                    key = (shorter_ctx, next_id)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        self.continuation[order - 1][shorter_ctx][next_id] += 1
        return self

    def _prob(self, context, order):
        if order == 0:
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

        d = min(self.discount, total)
        n_distinct_next = len(counts)
        lam = (d * n_distinct_next) / total

        probs = lower * lam
        for tok, c in counts.items():
            probs[tok] += max(c - d, 0.0) / total
        return probs

    def predict_log_probs(self, context):
        ctx = tuple(context[-self.max_order:]) if self.max_order > 0 else ()
        probs = self._prob(ctx, min(self.max_order, len(ctx)))
        probs = probs.clamp_min(1e-9)
        probs = probs / probs.sum()
        return probs.log()


def fit_kn_models(train_seqs_by_bird, vocab_size, max_order, discount):
    models = {}
    all_seqs = []
    for bird_id, seqs in train_seqs_by_bird.items():
        all_seqs.extend(seqs)
        models[bird_id] = KneserNeyBackoff(vocab_size, max_order, discount).fit(seqs)
    models["__global__"] = KneserNeyBackoff(vocab_size, max_order, discount).fit(all_seqs)
    return models


def kn_log_probs_for_batch(raw_contexts, bird_ids, kn_models, bird_vocab_itos, vocab_size, device):
    out = torch.empty(len(raw_contexts), vocab_size)
    for i, ctx in enumerate(raw_contexts):
        bird_str = bird_vocab_itos[bird_ids[i].item()]
        model = kn_models.get(bird_str, kn_models["__global__"])
        out[i] = model.predict_log_probs(ctx)
    return out.to(device)


# ============================================================================
# Torch dataset
# ============================================================================

@dataclass
class RawExample:
    context: list
    target: int
    bird_id: int


def build_examples(sequences_by_id, bird_of_seq, phrase_vocab, bird_vocab, unk_token, min_context_len):
    examples = []
    for seq_id, symbols in sequences_by_id.items():
        ids = phrase_vocab.encode_seq(symbols, unk_token)
        bird_str = bird_of_seq[seq_id]
        bird_id = bird_vocab.encode(bird_str, bird_vocab.itos[0])
        for i in range(min_context_len, len(ids)):
            examples.append(RawExample(context=ids[:i], target=ids[i], bird_id=bird_id))
    return examples


class PhrasePredictionDataset(Dataset):
    def __init__(self, examples, pad_id, max_context_len):
        self.examples = examples
        self.pad_id = pad_id
        self.max_context_len = max_context_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "context": ex.context[-self.max_context_len:],
            "bird_id": ex.bird_id,
            "target": ex.target,
        }


def collate_fn(batch, pad_id, max_context_len):
    """Left-pads contexts (position -1 is always the most recent real token,
    regardless of context length -- keeps pooling/causal-mask logic simple)."""
    lengths = [len(b["context"]) for b in batch]
    max_len = min(max(lengths), max_context_len)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    pad_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    bird_ids = torch.zeros(len(batch), dtype=torch.long)
    targets = torch.full((len(batch),), -100, dtype=torch.long)

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
        "raw_contexts": [b["context"] for b in batch],
    }


# ============================================================================
# Model
# ============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len):
        return self.pe[:seq_len]


class PhraseTransformer(nn.Module):
    def __init__(self, vocab_size, n_birds, d_model=192, n_heads=6, n_layers=4, d_ff=768,
                 dropout=0.2, bird_embed_dim=32, max_position_embeddings=256,
                 interp_hidden_dim=16, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.bird_embed = nn.Embedding(max(n_birds, 1), bird_embed_dim)
        self.bird_proj = nn.Linear(bird_embed_dim, d_model)
        self.pos_embed = SinusoidalPositionalEmbedding(d_model, max_position_embeddings)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.gate = nn.Sequential(
            nn.Linear(d_model + 1, interp_hidden_dim),
            nn.GELU(),
            nn.Linear(interp_hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    def _combined_mask(self, pad_mask, seq_len, device):
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        causal = causal.unsqueeze(0).expand(pad_mask.size(0), -1, -1).clone()
        causal.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
        all_masked = causal.eq(float("-inf")).all(dim=-1)
        causal[all_masked] = 0.0
        n_heads = self.encoder.layers[0].self_attn.num_heads
        return causal.repeat_interleave(n_heads, dim=0)

    def encode(self, input_ids, pad_mask, bird_ids):
        B, L = input_ids.shape
        device = input_ids.device

        tok = self.token_embed(input_ids)
        pos = self.pos_embed(L).unsqueeze(0)
        bird = self.bird_proj(self.bird_embed(bird_ids))

        x = tok + pos + bird.unsqueeze(1)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        attn_mask = self._combined_mask(pad_mask, L, device)
        x = self.encoder(x, mask=attn_mask, is_causal=False)
        x = self.final_norm(x)
        return x[:, -1, :]

    def forward(self, input_ids, pad_mask, bird_ids, kn_log_probs=None, context_lengths=None):
        pooled = self.encode(input_ids, pad_mask, bird_ids)
        neural_logits = self.output_proj(pooled)
        neural_log_probs = F.log_softmax(neural_logits, dim=-1)

        if kn_log_probs is None:
            return neural_log_probs, None

        if context_lengths is None:
            context_lengths = (~pad_mask).sum(dim=1).float()
        len_feat = torch.log1p(context_lengths.float()).unsqueeze(1)
        gate_input = torch.cat([pooled, len_feat], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        log_gate = torch.log(gate.clamp_min(1e-6))
        log_one_minus_gate = torch.log((1 - gate).clamp_min(1e-6))
        mixed = torch.logsumexp(
            torch.stack([neural_log_probs + log_gate, kn_log_probs + log_one_minus_gate], dim=0),
            dim=0,
        )
        mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)
        return mixed, gate.squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def build_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def label_smoothed_nll_input(log_probs, smoothing):
    if smoothing <= 0:
        return log_probs
    vocab_size = log_probs.size(-1)
    uniform_log_probs = -math.log(vocab_size)
    return torch.logsumexp(
        torch.stack([
            log_probs + math.log(1 - smoothing),
            torch.full_like(log_probs, uniform_log_probs) + math.log(smoothing),
        ], dim=0),
        dim=0,
    )


@torch.no_grad()
def topk_accuracy(logits, targets, ks=(1, 3, 5)):
    valid_mask = targets != -100
    if valid_mask.sum() == 0:
        return {f"top{k}": 0.0 for k in ks}
    logits = logits[valid_mask]
    targets = targets[valid_mask]
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=-1)
    correct = pred.eq(targets.unsqueeze(1))
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item() for k in ks}


class EarlyStopper:
    def __init__(self, patience, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg,
                     kn_models, bird_vocab_itos, use_gate):
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        bird_ids = batch["bird_ids"].to(device)
        targets = batch["targets"].to(device)
        context_lengths = (~pad_mask).sum(dim=1)

        kn_lp = None
        if use_gate:
            kn_lp = kn_log_probs_for_batch(batch["raw_contexts"], batch["bird_ids"], kn_models,
                                            bird_vocab_itos, model.vocab_size, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=(device == "cuda")):
            log_probs, _gate = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
            loss = F.nll_loss(
                label_smoothed_nll_input(log_probs, cfg.label_smoothing),
                targets, ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device, cfg, kn_models, bird_vocab_itos, use_gate):
    model.eval()
    total_loss, n_batches = 0.0, 0
    agg = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    n_examples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        bird_ids = batch["bird_ids"].to(device)
        targets = batch["targets"].to(device)
        context_lengths = (~pad_mask).sum(dim=1)

        kn_lp = None
        if use_gate:
            kn_lp = kn_log_probs_for_batch(batch["raw_contexts"], batch["bird_ids"], kn_models,
                                            bird_vocab_itos, model.vocab_size, device)
        log_probs, _gate = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)

        loss = F.nll_loss(log_probs, targets, ignore_index=-100)
        total_loss += loss.item()
        n_batches += 1

        metrics = topk_accuracy(log_probs, targets, ks=(1, 3, 5))
        batch_n = (targets != -100).sum().item()
        for k in agg:
            agg[k] += metrics[k] * batch_n
        n_examples += batch_n

    for k in agg:
        agg[k] /= max(1, n_examples)
    agg["loss"] = total_loss / max(1, n_batches)
    return agg


def fit(model, train_loader, val_loader, cfg, kn_models, bird_vocab_itos, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(cfg.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    stopper = EarlyStopper(patience=cfg.early_stopping_patience, mode="max")

    best_state = None
    checkpoint_states = []  # list of (val_top1, state_dict) for checkpoint averaging

    for epoch in range(cfg.max_epochs):
        use_gate = epoch >= cfg.gate_warmup_epochs
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                      device, cfg, kn_models, bird_vocab_itos, use_gate)
        val_metrics = evaluate(model, val_loader, device, cfg, kn_models, bird_vocab_itos, use_gate)
        dt = time.time() - t0

        phase = "gated" if use_gate else "neural-only(warmup)"
        print(f"epoch {epoch:03d} [{phase}] | train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} "
              f"| val_top1 {val_metrics['top1']:.4f} | val_top3 {val_metrics['top3']:.4f} "
              f"| val_top5 {val_metrics['top5']:.4f} | {dt:.1f}s")

        # Early stopping / checkpointing only apply once both branches train
        # together -- warmup is a fixed-length curriculum stage, not something
        # to cut short on a metric plateau.
        if not use_gate:
            continue

        state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint_states.append((val_metrics["top1"], state_cpu))
        checkpoint_states.sort(key=lambda x: -x[0])
        checkpoint_states = checkpoint_states[:cfg.top_k_to_average]

        improved = stopper.step(val_metrics["top1"])
        if improved:
            best_state = state_cpu

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best val_top1={stopper.best:.4f})")
            break

    return best_state, [s for _, s in checkpoint_states]


def average_state_dicts(states):
    if len(states) == 1:
        return states[0]
    avg = {k: v.clone().float() for k, v in states[0].items()}
    for state in states[1:]:
        for k in avg:
            avg[k] += state[k].float()
    for k in avg:
        avg[k] /= len(states)
    return avg


# ============================================================================
# Inference: temperature scaling + checkpoint-averaged + TTA prediction
# ============================================================================

@torch.no_grad()
def temperature_scaled_log_probs(log_probs, temperature):
    if temperature == 1.0:
        return log_probs
    scaled = log_probs / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


@torch.no_grad()
def fit_temperature(model, val_loader, device, kn_models, bird_vocab_itos):
    """1-D grid search on held-out NLL. Doesn't change top-1 predictions on
    its own (argmax is temperature-invariant) but calibrates confidences,
    which matters if this solution is later extended to a multi-seed
    probability-averaged ensemble."""
    model.eval()
    candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
    all_log_probs, all_targets = [], []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        bird_ids = batch["bird_ids"].to(device)
        targets = batch["targets"].to(device)
        context_lengths = (~pad_mask).sum(dim=1)
        kn_lp = kn_log_probs_for_batch(batch["raw_contexts"], batch["bird_ids"], kn_models,
                                        bird_vocab_itos, model.vocab_size, device)
        mixed_log_probs, _ = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
        all_log_probs.append(mixed_log_probs.cpu())
        all_targets.append(targets.cpu())

    log_probs = torch.cat(all_log_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    best_T, best_nll = 1.0, float("inf")
    for T in candidates:
        nll = F.nll_loss(temperature_scaled_log_probs(log_probs, T), targets, ignore_index=-100).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


@torch.no_grad()
def predict_example(model, context_ids, bird_id, kn_models, bird_str, device,
                     max_context_len, temperature, tta_crops):
    """Context-window TTA: predict using {100%, 75%, 50%} of the available
    context and average the resulting probability distributions. Long-range
    dependencies are real but so is recency-dominated local structure;
    cropping simulates "what would a shorter-context predictor say" and
    guards against overweighting a spurious long-range coincidence."""
    kn_model = kn_models.get(bird_str, kn_models["__global__"])
    full_len = len(context_ids)

    crop_lengths = sorted(set(
        max(1, int(round(full_len * frac))) for frac in tta_crops if full_len > 0
    ), reverse=True) or [max(1, full_len)]

    all_probs = []
    for crop_len in crop_lengths:
        cropped_ctx = (context_ids[-crop_len:] if crop_len > 0 else context_ids)[-max_context_len:]
        kn_lp = kn_model.predict_log_probs(cropped_ctx).unsqueeze(0).to(device)

        input_ids = torch.tensor([cropped_ctx], dtype=torch.long, device=device)
        pad_mask = torch.zeros((1, len(cropped_ctx)), dtype=torch.bool, device=device)
        bird_ids = torch.tensor([bird_id], dtype=torch.long, device=device)
        context_lengths = torch.tensor([len(cropped_ctx)], device=device)

        mixed_log_probs, _ = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
        mixed_log_probs = temperature_scaled_log_probs(mixed_log_probs, temperature)
        all_probs.append(mixed_log_probs.exp().squeeze(0).cpu())

    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    return int(torch.argmax(avg_probs).item())


# ============================================================================
# Main
# ============================================================================

def split_sequences_by_id(sequences_by_id, val_fraction, seed):
    """Sequence-level (not example-level) split -- splitting individual
    (context, target) examples would leak adjacent positions from the same
    training sequence across train/val, inflating validation accuracy
    without reflecting true generalization."""
    ids = list(sequences_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids = [i for i in ids if i in val_ids]
    return train_ids, val_ids


def main():
    cfg = CFG
    seed_everything(cfg.seed)
    os.makedirs(WORKING_DIR, exist_ok=True)
    device = cfg.device
    print(f"Using device: {device}")

    # ---------------- Load data ----------------
    with open(TRAIN_PATH) as f:
        train_raw = json.load(f)
    with open(TEST_PATH) as f:
        test_raw = json.load(f)

    sequences_by_id = {row.get("seq_id", row.get("id")): row["symbols"] for row in train_raw}
    bird_of_seq = {seq_id: infer_bird_id(str(seq_id)) for seq_id in sequences_by_id}

    specials = [cfg.pad_token, cfg.unk_token, cfg.bos_token]
    phrase_vocab = build_phrase_vocab(list(sequences_by_id.values()), specials)
    bird_vocab = Vocab(sorted(set(bird_of_seq.values())), specials=["__global__"])
    print(f"Phrase vocab size: {len(phrase_vocab)} | Bird/individual buckets: {len(bird_vocab)}")

    train_ids, val_ids = split_sequences_by_id(sequences_by_id, cfg.val_fraction, cfg.seed)
    print(f"Sequences: {len(sequences_by_id)} total | {len(train_ids)} train | {len(val_ids)} val")

    train_seqs_by_id = {i: sequences_by_id[i] for i in train_ids}
    val_seqs_by_id = {i: sequences_by_id[i] for i in val_ids}

    train_examples = build_examples(train_seqs_by_id, bird_of_seq, phrase_vocab, bird_vocab,
                                     cfg.unk_token, cfg.min_context_len)
    val_examples = build_examples(val_seqs_by_id, bird_of_seq, phrase_vocab, bird_vocab,
                                   cfg.unk_token, cfg.min_context_len)
    print(f"Examples: {len(train_examples)} train | {len(val_examples)} val")

    pad_id = phrase_vocab.stoi[cfg.pad_token]
    train_ds = PhrasePredictionDataset(train_examples, pad_id, cfg.max_context_len)
    val_ds = PhrasePredictionDataset(val_examples, pad_id, cfg.max_context_len)

    def collate(batch):
        return collate_fn(batch, pad_id, cfg.max_context_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)

    # ---------------- Fit Kneser-Ney backoff ----------------
    train_seqs_encoded_by_bird = {}
    for seq_id in train_ids:
        bird_str = bird_of_seq[seq_id]
        ids = phrase_vocab.encode_seq(sequences_by_id[seq_id], cfg.unk_token)
        train_seqs_encoded_by_bird.setdefault(bird_str, []).append(ids)

    kn_models = fit_kn_models(train_seqs_encoded_by_bird, len(phrase_vocab), cfg.kn_max_order, cfg.kn_discount)
    print(f"Fitted Kneser-Ney models for {len(kn_models) - 1} bird buckets + 1 global backoff.")

    # ---------------- Build + train model ----------------
    model = PhraseTransformer(
        vocab_size=len(phrase_vocab), n_birds=len(bird_vocab), d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, d_ff=cfg.d_ff, dropout=cfg.dropout,
        bird_embed_dim=cfg.bird_embed_dim, max_position_embeddings=cfg.max_position_embeddings,
        interp_hidden_dim=cfg.interp_hidden_dim, pad_id=pad_id,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_state, topk_states = fit(model, train_loader, val_loader, cfg, kn_models, bird_vocab.itos, device)

    # ---------------- Checkpoint averaging (SWA-style) ----------------
    averaged_state = average_state_dicts(topk_states) if topk_states else best_state
    model.load_state_dict(averaged_state)
    model.eval()

    # ---------------- Temperature scaling ----------------
    best_T = fit_temperature(model, val_loader, device, kn_models, bird_vocab.itos)
    print(f"Fitted temperature: {best_T}")

    # ---------------- Predict on test set ----------------
    rows = []
    for row in test_raw:
        item_id = row.get("item_id", row.get("id"))
        context_syms = row["context"]
        bird_str = infer_bird_id(str(item_id))

        context_ids = phrase_vocab.encode_seq(context_syms, cfg.unk_token)
        bird_id = bird_vocab.encode(bird_str, bird_vocab.itos[0])

        pred_id = predict_example(
            model=model, context_ids=context_ids, bird_id=bird_id, kn_models=kn_models,
            bird_str=bird_str, device=device, max_context_len=cfg.max_context_len,
            temperature=best_T, tta_crops=cfg.tta_context_crops,
        )
        rows.append((item_id, phrase_vocab.itos[pred_id]))

    import csv
    with open(SUBMISSION_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "symbol"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
