"""
Training loop: mixed precision, gradient clipping, cosine LR schedule with
linear warmup, label smoothing, early stopping on validation top-1 accuracy,
and per-epoch top-k metric logging.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import Config, EarlyStopper, ensure_dir, topk_accuracy


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to ~0. Standard, robust choice for
    Transformer training at this scale — warmup prevents the large early
    gradients (from randomly-initialized attention) from destabilizing
    training, cosine decay avoids the accuracy cliff a step schedule can
    cause right at the decay boundary.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def kn_log_probs_for_batch(raw_contexts: list, bird_ids: torch.Tensor, kn_models: dict,
                            bird_vocab_itos: list, vocab_size: int, device) -> torch.Tensor:
    """Query the fitted Kneser-Ney models for each example's context.

    Falls back bird-specific -> global model whenever the bird-specific
    table has literally zero observations for the deepest available order
    of this context (handled inside KneserNeyBackoff via its own recursion,
    so here we just choose which top-level model object to query).
    """
    out = torch.empty(len(raw_contexts), vocab_size)
    for i, ctx in enumerate(raw_contexts):
        bird_str = bird_vocab_itos[bird_ids[i].item()]
        model = kn_models.get(bird_str, kn_models["__global__"])
        out[i] = model.predict_log_probs(ctx)
    return out.to(device)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg: Config,
                     kn_models: dict, bird_vocab_itos: list, use_gate: bool):
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        bird_ids = batch["bird_ids"].to(device)
        targets = batch["targets"].to(device)
        context_lengths = (~pad_mask).sum(dim=1)

        # During the gate warmup phase, skip the KN branch entirely (kn_lp=None
        # forces model.forward into pure-neural mode) -- see Config.gate_warmup_epochs
        # for why: judging the untrained neural branch against KN from step 1
        # causes the learned gate to collapse to ~0 and never recover.
        kn_lp = None
        if use_gate:
            kn_lp = kn_log_probs_for_batch(batch["raw_contexts"], batch["bird_ids"], kn_models,
                                            bird_vocab_itos, model.vocab_size, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                             enabled=cfg.use_amp and device == "cuda"):
            log_probs, _gate = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
            loss = F.nll_loss(
                _label_smoothed_nll_input(log_probs, cfg.label_smoothing),
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


def _label_smoothed_nll_input(log_probs: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Blend the model's log-probs with a uniform distribution's log-probs so
    F.nll_loss effectively computes a label-smoothed cross-entropy, even
    though our final distribution is already a mixture (so we can't use
    nn.CrossEntropyLoss(label_smoothing=...) directly on raw logits).

    Why label smoothing at all here: the grammar is explicitly stochastic
    (task description), so several next-phrases are frequently *all*
    valid at a given position. Training with hard one-hot targets pushes
    the model to be falsely overconfident about the single observed
    sample at each context; smoothing keeps calibration honest and reduces
    overfitting to which particular branch happened to be sampled in the
    (finite) training bouts.
    """
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
def evaluate(model, loader, device, cfg: Config, kn_models: dict, bird_vocab_itos: list,
             use_gate: bool):
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


def fit(model, train_loader, val_loader, cfg: Config, kn_models: dict, bird_vocab_itos: list,
        device: str):
    ensure_dir(cfg.checkpoint_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")
    stopper = EarlyStopper(patience=cfg.early_stopping_patience, mode="max")

    history = []
    saved_checkpoints = []

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
        history.append({"epoch": epoch, "train_loss": train_loss, "use_gate": use_gate, **val_metrics})

        # Early stopping is only meaningful once both branches are being
        # trained together; the warmup phase is a fixed-length curriculum
        # stage, not something to cut short on a metric plateau.
        if not use_gate:
            continue

        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch{epoch:03d}_top1{val_metrics['top1']:.4f}.pt")
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_top1": val_metrics["top1"]},
                   ckpt_path)
        saved_checkpoints.append((val_metrics["top1"], ckpt_path))

        improved = stopper.step(val_metrics["top1"])
        if improved:
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_top1": val_metrics["top1"]},
                       os.path.join(cfg.checkpoint_dir, "best.pt"))

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best val_top1={stopper.best:.4f})")
            break

    # Keep only the top-K checkpoints on disk (by val_top1) to save space,
    # plus best.pt which is already separate.
    saved_checkpoints.sort(key=lambda x: -x[0])
    keep = set(p for _, p in saved_checkpoints[:cfg.top_k_to_average])
    for _, p in saved_checkpoints:
        if p not in keep and os.path.exists(p):
            os.remove(p)

    top_k_paths = [p for _, p in saved_checkpoints[:cfg.top_k_to_average]]
    return history, top_k_paths
