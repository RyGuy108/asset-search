"""Tokenizer for the asset encoder.

Each word becomes a word id (from a vocabulary built on the training captions) plus a
bag of hashed character n-grams, fastText style. The n-grams are what let the encoder
cope with typos ("watercolr"), plurals and words it never saw in training
("bioluminescent"), which matters because users type search queries fast and sloppily.
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PAD, UNK = 0, 1
NGRAM_BUCKETS = 1 << 17
NGRAMS = (3, 4)
MAX_NGRAMS_PER_WORD = 12


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _ngram_ids(word: str) -> list[int]:
    w = f"<{word}>"
    out: list[int] = []
    for n in NGRAMS:
        for i in range(len(w) - n + 1):
            # crc32 is stable across processes, unlike hash()
            out.append(1 + zlib.crc32(w[i : i + n].encode()) % (NGRAM_BUCKETS - 1))
    return out[:MAX_NGRAMS_PER_WORD]


class Tokenizer:
    def __init__(self, vocab: list[str], max_len: int = 96):
        self.itos = ["<pad>", "<unk>", *vocab]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len
        self._ngram_cache: dict[str, list[int]] = {}

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 3, max_len: int = 96) -> Tokenizer:
        counts = Counter(w for t in texts for w in words(t))
        vocab = sorted(w for w, c in counts.items() if c >= min_count)
        return cls(vocab, max_len)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"vocab": self.itos[2:], "max_len": self.max_len}))

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        d = json.loads(path.read_text())
        return cls(d["vocab"], d["max_len"])

    def ngrams(self, w: str) -> list[int]:
        ids = self._ngram_cache.get(w)
        if ids is None:
            ids = _ngram_ids(w)
            if len(self._ngram_cache) < 200_000:
                self._ngram_cache[w] = ids
        return ids

    def encode_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Returns word ids [B, T], n-gram ids [B, T, G] and a padding mask [B, T]."""
        toks = [words(t)[: self.max_len] or ["<unk>"] for t in texts]
        T = max(len(t) for t in toks)
        wid = torch.zeros(len(texts), T, dtype=torch.long)
        gid = torch.zeros(len(texts), T, MAX_NGRAMS_PER_WORD, dtype=torch.long)
        for b, ws in enumerate(toks):
            for t, w in enumerate(ws):
                wid[b, t] = self.stoi.get(w, UNK)
                g = self.ngrams(w)
                gid[b, t, : len(g)] = torch.tensor(g)
        return {"words": wid, "ngrams": gid, "mask": wid != PAD}
