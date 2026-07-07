"""
Model architecture: causal Transformer decoder over phrase tokens, with
bird/dialect conditioning and a learned interpolation gate that blends the
neural distribution with the Kneser-Ney backoff distribution.

Why interpolate inside the model rather than as a post-hoc ensemble:
the two signals are complementary in *different regimes of the same
example* (short/common contexts vs. long/rare contexts), not just noisy
estimates of the same thing — so a fixed global mixing weight is provably
suboptimal. Making the mixing weight a function of the current context
(via a small gating MLP fed a handful of context statistics) lets the
model learn e.g. "context length < 2 -> trust KN more" without hand-tuning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positions.

    Chosen over learned absolute position embeddings because contexts are
    left-padded and variable length (up to max_context_len) — sinusoidal
    embeddings extrapolate cleanly to context lengths seen rarely during
    training without needing extra learnable parameters that would only see
    sparse gradient signal at the longest positions.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class PhraseTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_birds: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        d_ff: int = 768,
        dropout: float = 0.2,
        bird_embed_dim: int = 32,
        max_position_embeddings: int = 256,
        interp_hidden_dim: int = 16,
        pad_id: int = 0,
    ):
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
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: materially more stable to train at this depth without a long LR warmup tail
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size)

        # Gate: takes [pooled hidden state, log(context_length+1)] -> scalar in (0, 1)
        # deciding how much weight the neural distribution gets vs. the KN distribution.
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

    def _combined_mask(self, pad_mask: torch.Tensor, seq_len: int, device) -> torch.Tensor:
        # Additive float mask, one per batch element: 0 where attention is
        # allowed, -inf above the diagonal (causal) OR wherever the key
        # position is PAD. Merging padding into the same float mask (rather
        # than passing a separate bool src_key_padding_mask) avoids relying
        # on PyTorch's causal+padding mask-merging path, whose dtype
        # coercion is deprecated as of recent torch versions.
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        causal = causal.unsqueeze(0).expand(pad_mask.size(0), -1, -1).clone()  # (B, L, L)
        # pad_mask: True = PAD key position -> block attending to it from every query row.
        causal.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
        # Guard fully-padded query rows (all-(-inf) rows) from producing NaN softmax;
        # such rows are never read from (they're padding on the query side too).
        all_masked = causal.eq(float("-inf")).all(dim=-1)
        causal[all_masked] = 0.0
        # Repeat per attention head: nn.MultiheadAttention expects (B*num_heads, L, L)
        # when given a 3D mask.
        n_heads = self.encoder.layers[0].self_attn.num_heads
        return causal.repeat_interleave(n_heads, dim=0)

    def encode(self, input_ids: torch.Tensor, pad_mask: torch.Tensor, bird_ids: torch.Tensor):
        """Returns the pooled hidden state at the last (rightmost, most-recent) position."""
        B, L = input_ids.shape
        device = input_ids.device

        tok = self.token_embed(input_ids)                      # (B, L, D)
        pos = self.pos_embed(L).unsqueeze(0)                    # (1, L, D)
        bird = self.bird_proj(self.bird_embed(bird_ids))        # (B, D)

        x = tok + pos + bird.unsqueeze(1)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        attn_mask = self._combined_mask(pad_mask, L, device)
        x = self.encoder(x, mask=attn_mask, is_causal=False)
        x = self.final_norm(x)

        pooled = x[:, -1, :]  # last position = most recent real token (contexts are left-padded)
        return pooled

    def forward(self, input_ids, pad_mask, bird_ids, kn_log_probs=None, context_lengths=None):
        """
        kn_log_probs: optional (B, V) precomputed Kneser-Ney log-probabilities
                      for this batch's contexts. If provided, the model
                      returns interpolated log-probabilities (log-space
                      mixture) instead of raw logits — use this path for
                      loss computation and final predictions.
        context_lengths: (B,) true (unpadded) context lengths, used as a
                      cheap, informative gate feature (short context ->
                      lean on KN; long context -> lean on neural model).
        """
        pooled = self.encode(input_ids, pad_mask, bird_ids)
        neural_logits = self.output_proj(pooled)                # (B, V)
        neural_log_probs = F.log_softmax(neural_logits, dim=-1)

        if kn_log_probs is None:
            return neural_log_probs, None

        if context_lengths is None:
            context_lengths = (~pad_mask).sum(dim=1).float()
        len_feat = torch.log1p(context_lengths.float()).unsqueeze(1)  # (B, 1)
        gate_input = torch.cat([pooled, len_feat], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))              # (B, 1), weight on the neural branch

        # log-space mixture: log(gate * p_neural + (1-gate) * p_kn), computed
        # stably via logsumexp instead of exponentiating both branches naively.
        log_gate = torch.log(gate.clamp_min(1e-6))
        log_one_minus_gate = torch.log((1 - gate).clamp_min(1e-6))
        mixed = torch.logsumexp(
            torch.stack([neural_log_probs + log_gate, kn_log_probs + log_one_minus_gate], dim=0),
            dim=0,
        )
        mixed = mixed - torch.logsumexp(mixed, dim=-1, keepdim=True)  # renormalize for numerical safety
        return mixed, gate.squeeze(-1)
