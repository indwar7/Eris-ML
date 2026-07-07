"""
Generate the final submission file from a trained checkpoint directory.

Usage:
    python submission.py --checkpoint_dir checkpoints --test_path data/test.json \
        --output outputs/submission.csv

Steps:
  1. Load persisted vocab, bird vocab, KN models, and config from training.
  2. Build model(s): checkpoint-averaged single model by default; pass
     multiple --checkpoint_dir values (comma-separated) to ensemble across
     independently trained runs/seeds as well.
  3. Fit global temperature scaling on the training run's validation split
     (recomputed here from train.json using the same split seed, so no
     extra artifact is required).
  4. For each test item: map its context through the phrase vocab (unseen
     symbols -> <UNK>), infer its bird id (falling back to the global KN
     model + bird-agnostic embedding row 0 for unseen birds), and predict
     via ensembled, temperature-scaled, TTA-averaged probabilities.
  5. Write a CSV with columns [item_id, prediction].
"""

from __future__ import annotations

import argparse
import csv
import functools
import pickle

import torch
from torch.utils.data import DataLoader

from dataset import (
    PhrasePredictionDataset,
    build_examples,
    collate_fn,
    infer_bird_id,
    load_test_json,
    load_train_json,
)
from predict import fit_temperature, load_averaged_model, predict_example
from trainer import kn_log_probs_for_batch
from train import split_sequences_by_id
from utils import Config, ensure_dir, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Comma-separated list of checkpoint dirs to ensemble across independent runs.")
    p.add_argument("--test_path", type=str, default=None)
    p.add_argument("--output", type=str, default="outputs/submission.csv")
    return p.parse_args()


def load_run_artifacts(ckpt_dir: str):
    cfg = Config.load(f"{ckpt_dir}/config.json")
    with open(f"{ckpt_dir}/phrase_vocab.pkl", "rb") as f:
        phrase_vocab = pickle.load(f)
    with open(f"{ckpt_dir}/bird_vocab.pkl", "rb") as f:
        bird_vocab = pickle.load(f)
    with open(f"{ckpt_dir}/kn_models.pkl", "rb") as f:
        kn_models = pickle.load(f)
    with open(f"{ckpt_dir}/top_k_checkpoints.pkl", "rb") as f:
        top_k_paths = pickle.load(f)
    return cfg, phrase_vocab, bird_vocab, kn_models, top_k_paths


def rebuild_val_loader(cfg, phrase_vocab, bird_vocab):
    """Reconstruct the exact validation split used during training (same
    seed + same split function) purely to fit temperature scaling — avoids
    needing to persist the val set itself as a separate artifact."""
    raw = load_train_json(cfg.train_path)
    sequences_by_id = {row.get("seq_id", row.get("id")): row["symbols"] for row in raw}
    bird_of_seq = {seq_id: infer_bird_id(str(seq_id)) for seq_id in sequences_by_id}
    _, val_ids = split_sequences_by_id(sequences_by_id, cfg.val_fraction, cfg.seed)
    val_seqs_by_id = {i: sequences_by_id[i] for i in val_ids}

    val_examples = build_examples(val_seqs_by_id, bird_of_seq, phrase_vocab, bird_vocab,
                                   cfg.unk_token, cfg.min_context_len)
    pad_id = phrase_vocab.stoi[cfg.pad_token]
    val_ds = PhrasePredictionDataset(val_examples, pad_id, cfg.max_context_len)
    collate = functools.partial(collate_fn, pad_id=pad_id, max_context_len=cfg.max_context_len)
    return DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)


def main():
    args = parse_args()
    ckpt_dirs = [d.strip() for d in args.checkpoint_dir.split(",") if d.strip()]

    ensembled_models = []
    # Use the first run's artifacts as canonical (vocab/bird_vocab/kn_models/test path
    # must be shared across ensembled runs — they were all trained on the same train.json).
    cfg, phrase_vocab, bird_vocab, kn_models, _ = load_run_artifacts(ckpt_dirs[0])
    seed_everything(cfg.seed)
    device = cfg.device
    test_path = args.test_path or cfg.test_path

    best_temperature = 1.0
    for i, ckpt_dir in enumerate(ckpt_dirs):
        run_cfg, run_phrase_vocab, run_bird_vocab, run_kn_models, top_k_paths = load_run_artifacts(ckpt_dir)
        model_kwargs = dict(
            vocab_size=len(run_phrase_vocab),
            n_birds=len(run_bird_vocab),
            d_model=run_cfg.d_model,
            n_heads=run_cfg.n_heads,
            n_layers=run_cfg.n_layers,
            d_ff=run_cfg.d_ff,
            dropout=run_cfg.dropout,
            bird_embed_dim=run_cfg.bird_embed_dim,
            max_position_embeddings=run_cfg.max_position_embeddings,
            interp_hidden_dim=run_cfg.interp_hidden_dim,
            pad_id=run_phrase_vocab.stoi[run_cfg.pad_token],
        )
        model = load_averaged_model(top_k_paths, model_kwargs, device)
        ensembled_models.append(model)

        if i == 0:
            val_loader = rebuild_val_loader(run_cfg, run_phrase_vocab, run_bird_vocab)
            best_temperature = fit_temperature(model, val_loader, device, run_cfg, run_kn_models,
                                                run_bird_vocab.itos, kn_log_probs_for_batch)
            print(f"Fitted temperature: {best_temperature}")

    # ---------------- Predict on test set ----------------
    test_raw = load_test_json(test_path)
    ensure_dir("outputs")

    rows = []
    for row in test_raw:
        item_id = row.get("item_id", row.get("id"))
        context_syms = row["context"]
        bird_str = infer_bird_id(str(item_id))

        context_ids = phrase_vocab.encode_seq(context_syms, cfg.unk_token)
        bird_id = bird_vocab.encode(bird_str, bird_vocab.itos[0])

        pred_id = predict_example(
            models=ensembled_models,
            context_ids=context_ids,
            bird_id=bird_id,
            kn_models=kn_models,
            bird_str=bird_str,
            vocab_size=len(phrase_vocab),
            device=device,
            max_context_len=cfg.max_context_len,
            temperature=best_temperature,
            tta_crops=cfg.tta_context_crops,
        )
        pred_symbol = phrase_vocab.itos[pred_id]
        rows.append((item_id, pred_symbol))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([cfg.submission_id_column, cfg.submission_prediction_column])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
