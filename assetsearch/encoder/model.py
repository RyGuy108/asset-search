"""Bi-encoder that maps search queries and generation prompts into one vector space.

word embedding + mean of hashed char n-gram embeddings -> small Transformer -> masked
mean pool -> linear -> L2 normalise. Queries and documents share all weights, so a
query and the prompt that produced an asset land close together when they describe
the same thing, even with no words in common ("pup on the sand" vs "a dog lying on a
beach").
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .tokenize import NGRAM_BUCKETS


class AssetEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        out_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 96,
        bag_only: bool = False,
    ):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, dim=dim, out_dim=out_dim, layers=layers,
            heads=heads, max_len=max_len, bag_only=bag_only,
        )
        self.word = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.ngram = nn.EmbeddingBag(NGRAM_BUCKETS, dim, mode="sum", padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.bag_only = bag_only
        if not bag_only:
            layer = nn.TransformerEncoderLayer(
                dim, heads, dim_feedforward=dim * 4, dropout=0.1,
                batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.in_norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.05)))
        nn.init.normal_(self.word.weight, std=0.02)
        nn.init.normal_(self.ngram.weight, std=0.02)

    def forward(self, words: torch.Tensor, ngrams: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = words.shape
        G = ngrams.shape[-1]
        flat = ngrams.reshape(B * T, G)
        n = (flat != 0).sum(-1, keepdim=True).clamp(min=1)
        x = self.word(words) + (self.ngram(flat) / n).reshape(B, T, -1)
        if not self.bag_only:
            # The Transformer refines the bag-of-words vectors rather than replacing them.
            # The embeddings start tiny (std 0.02); normalising them first stops the
            # Transformer's own residual stream from swamping them, which otherwise
            # collapses every input to almost the same vector.
            h = self.in_norm(x) + self.pos(torch.arange(T, device=words.device))[None]
            x = self.in_norm(x) + self.encoder(h, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        x = (self.norm(x) * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.proj(x), dim=-1)


def info_nce(q: torch.Tensor, d: torch.Tensor, scale: torch.Tensor, d_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric in-batch contrastive loss.

    d_ids marks which documents describe the same source image; those are not
    treated as negatives for each other (a batch can hold two captions of one image).
    """
    logits = scale.exp().clamp(max=100) * q @ d.T
    target = torch.arange(len(q), device=q.device)
    if d_ids is not None:
        same = d_ids[:, None] == d_ids[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, -1e4)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
