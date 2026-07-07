"""
Main training entrypoint.

Usage:
    python train.py --config configs/base.json
    python train.py --train_path data/train.json --max_epochs 40

Pipeline:
  1. Load train.json, infer per-bird ids, build phrase vocab.
  2. Split sequences (not raw examples!) into train/val so validation
     truly measures generalization to unseen bouts, not just unseen
     positions within a bout the model has already partially seen.
  3. Fit the Kneser-Ney backoff models on the TRAIN split only.
  4. Build the PhraseTransformer + interpolation gate.
  5. Train with AMP/grad-clip/cosine-schedule/early-stopping via trainer.fit.
  6. Persist vocab, bird vocab, KN models, and config alongside checkpoints
     so predict.py / submission.py can reconstruct everything deterministically.
"""

from __future__ import annotations

import argparse
import functools
import pickle
import random

import torch
from torch.utils.data import DataLoader

from dataset import (
    Vocab,
    PhrasePredictionDataset,
    build_examples,
    build_phrase_vocab,
    collate_fn,
    fit_kn_models,
    infer_bird_id,
    load_train_json,
)
from model import PhraseTransformer
from trainer import fit
from utils import Config, seed_everything, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    # Allow overriding any Config field from the CLI for quick hparam sweeps.
    for field_name, field_val in Config().__dict__.items():
        if isinstance(field_val, bool):
            p.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=None)
        elif isinstance(field_val, (int, float, str)):
            p.add_argument(f"--{field_name}", type=type(field_val), default=None)
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def split_sequences_by_id(sequences_by_id: dict, val_fraction: float, seed: int):
    """Sequence-level (not example-level) train/val split — prevents leakage
    of near-duplicate positions from the same bout across the split."""
    ids = list(sequences_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids = [i for i in ids if i in val_ids]
    return train_ids, val_ids


def main():
    args = parse_args()
    cfg = build_config(args)
    seed_everything(cfg.seed)
    ensure_dir(cfg.checkpoint_dir)

    device = cfg.device
    print(f"Using device: {device}")

    # ---------------- Load & prepare data ----------------
    raw = load_train_json(cfg.train_path)
    sequences_by_id = {}
    for row in raw:
        seq_id = row.get("seq_id", row.get("id"))
        symbols = row["symbols"]
        sequences_by_id[seq_id] = symbols

    bird_of_seq = {seq_id: infer_bird_id(str(seq_id)) for seq_id in sequences_by_id}

    specials = [cfg.pad_token, cfg.unk_token, cfg.bos_token]
    phrase_vocab = build_phrase_vocab(list(sequences_by_id.values()), specials)
    bird_vocab = Vocab(sorted(set(bird_of_seq.values())), specials=["__global__"])

    print(f"Phrase vocab size: {len(phrase_vocab)} | Bird/dialect count: {len(bird_vocab)}")

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

    collate = functools.partial(collate_fn, pad_id=pad_id, max_context_len=cfg.max_context_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)

    # ---------------- Fit Kneser-Ney backoff (non-neural half of the ensemble) ----------------
    train_seqs_encoded_by_bird = {}
    for seq_id in train_ids:
        bird_str = bird_of_seq[seq_id]
        ids = phrase_vocab.encode_seq(sequences_by_id[seq_id], cfg.unk_token)
        train_seqs_encoded_by_bird.setdefault(bird_str, []).append(ids)

    kn_models = fit_kn_models(train_seqs_encoded_by_bird, len(phrase_vocab), cfg.kn_max_order,
                               cfg.kn_discount)
    print(f"Fitted Kneser-Ney models for {len(kn_models) - 1} birds + 1 global backoff.")

    # ---------------- Build model ----------------
    model = PhraseTransformer(
        vocab_size=len(phrase_vocab),
        n_birds=len(bird_vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        bird_embed_dim=cfg.bird_embed_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        interp_hidden_dim=cfg.interp_hidden_dim,
        pad_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---------------- Train ----------------
    history, top_k_checkpoint_paths = fit(model, train_loader, val_loader, cfg, kn_models,
                                           bird_vocab.itos, device)

    # ---------------- Persist everything needed for inference ----------------
    with open(f"{cfg.checkpoint_dir}/phrase_vocab.pkl", "wb") as f:
        pickle.dump(phrase_vocab, f)
    with open(f"{cfg.checkpoint_dir}/bird_vocab.pkl", "wb") as f:
        pickle.dump(bird_vocab, f)
    with open(f"{cfg.checkpoint_dir}/kn_models.pkl", "wb") as f:
        pickle.dump(kn_models, f)
    with open(f"{cfg.checkpoint_dir}/top_k_checkpoints.pkl", "wb") as f:
        pickle.dump(top_k_checkpoint_paths, f)
    cfg.save(f"{cfg.checkpoint_dir}/config.json")

    print("Training complete. Artifacts saved to:", cfg.checkpoint_dir)
    print("Top checkpoints for averaging:", top_k_checkpoint_paths)


if __name__ == "__main__":
    main()
