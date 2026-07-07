"""
Inference utilities: temperature scaling, checkpoint averaging (SWA-style),
multi-model ensembling, and context-window test-time augmentation (TTA).

Temperature scaling note: since the final training-time output is already
a normalized log-probability mixture (neural + KN in log-space), temperature
is applied to the *mixed* log-probs before the final softmax renormalization,
which calibrates the combined distribution rather than just the neural half.
Because evaluation is top-1 accuracy (argmax-only, order-preserving under
any positive temperature), temperature scaling on its own cannot change
which single-model prediction wins — its value here is strictly for making
per-example confidences comparable across the ensemble members before
averaging their probabilities (a poorly-calibrated component would
otherwise dominate a naive logit-average even when it's less accurate).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset import KneserNeyBackoff
from model import PhraseTransformer
from utils import Config, average_checkpoints


@torch.no_grad()
def temperature_scaled_log_probs(log_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return log_probs
    scaled = log_probs / temperature
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


@torch.no_grad()
def fit_temperature(model, val_loader, device, cfg: Config, kn_models: dict,
                     bird_vocab_itos: list, kn_log_probs_for_batch_fn) -> float:
    """Grid-search a single global temperature on held-out validation NLL.

    A 1-parameter grid search is preferred over gradient-based temperature
    fitting here: it's exactly as effective for a scalar and removes any
    risk of a second optimization loop silently diverging inside a
    submission notebook.
    """
    model.eval()
    candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
    best_T, best_nll = 1.0, float("inf")

    all_log_probs, all_targets = [], []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        bird_ids = batch["bird_ids"].to(device)
        targets = batch["targets"].to(device)
        context_lengths = (~pad_mask).sum(dim=1)
        kn_lp = kn_log_probs_for_batch_fn(batch["raw_contexts"], batch["bird_ids"], kn_models,
                                           bird_vocab_itos, model.vocab_size, device)
        mixed_log_probs, _ = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
        all_log_probs.append(mixed_log_probs.cpu())
        all_targets.append(targets.cpu())

    log_probs = torch.cat(all_log_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    for T in candidates:
        scaled = temperature_scaled_log_probs(log_probs, T)
        nll = F.nll_loss(scaled, targets, ignore_index=-100).item()
        if nll < best_nll:
            best_nll, best_T = nll, T

    return best_T


def load_averaged_model(checkpoint_paths: list, model_kwargs: dict, device: str) -> PhraseTransformer:
    """Build a fresh model and load SWA-averaged weights from the given checkpoints."""
    model = PhraseTransformer(**model_kwargs).to(device)
    if len(checkpoint_paths) == 1:
        state = torch.load(checkpoint_paths[0], map_location="cpu")["model_state"]
    else:
        state = average_checkpoints(checkpoint_paths)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_example(
    models: list,
    context_ids: list,
    bird_id: int,
    kn_models: dict,
    bird_str: str,
    vocab_size: int,
    device: str,
    max_context_len: int,
    temperature: float = 1.0,
    tta_crops=(1.0, 0.75, 0.5),
) -> int:
    """Ensemble + TTA prediction for a single test example.

    Ensembling strategy:
      - Model ensemble: average PROBABILITIES (not logits) across `models`
        (e.g. several checkpoint-averaged folds or seeds) — probability
        averaging is the right combination rule when models can disagree
        in *direction* (favor different modes of a genuinely multi-modal
        stochastic grammar), since it lets a confident minority model still
        contribute mass to its favored answer rather than being swamped in
        logit space by an unrelated model's unrelated large logits.
      - Context-window TTA: re-run prediction using only the most recent
        {100%, 75%, 50%} of the available context and average those
        probability distributions too. Rationale: long-range dependencies
        are real but so is recency-dominated local structure; cropping the
        context simulates "what would a shorter-context predictor say" and
        guards against the full-context model overweighting a spurious
        long-range coincidence for contexts where the short-range signal is
        actually cleaner. This is only applied when the full context has
        enough tokens that a crop is meaningfully different.
    """
    kn_model = kn_models.get(bird_str, kn_models["__global__"])
    full_len = len(context_ids)

    crop_lengths = sorted(set(
        max(1, int(round(full_len * frac))) for frac in tta_crops if full_len > 0
    ), reverse=True)
    if not crop_lengths:
        crop_lengths = [max(1, full_len)]

    all_probs = []
    for crop_len in crop_lengths:
        cropped_ctx = context_ids[-crop_len:] if crop_len > 0 else context_ids
        cropped_ctx = cropped_ctx[-max_context_len:]

        kn_lp = kn_model.predict_log_probs(cropped_ctx).unsqueeze(0).to(device)

        input_ids = torch.tensor([cropped_ctx], dtype=torch.long, device=device)
        pad_mask = torch.zeros((1, len(cropped_ctx)), dtype=torch.bool, device=device)
        bird_ids = torch.tensor([bird_id], dtype=torch.long, device=device)
        context_lengths = torch.tensor([len(cropped_ctx)], device=device)

        for model in models:
            mixed_log_probs, _ = model(input_ids, pad_mask, bird_ids, kn_lp, context_lengths)
            mixed_log_probs = temperature_scaled_log_probs(mixed_log_probs, temperature)
            all_probs.append(mixed_log_probs.exp().squeeze(0).cpu())

    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    return int(torch.argmax(avg_probs).item())
