"""Load the trained encoder and turn text into vectors, with a Redis cache for queries."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import config
from .encoder.model import AssetEncoder
from .encoder.tokenize import Tokenizer

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class Embedder:
    def __init__(self, model_dir: str | Path = config.MODEL_DIR, redis=None):
        model_dir = Path(model_dir)
        ckpt = torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=True)
        self.model = AssetEncoder(**ckpt["config"])
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state"].items()})
        self.model.eval()
        self.tok = Tokenizer.load(model_dir / "tokenizer.json")
        self.dim = ckpt["config"]["out_dim"]
        self.version = hashlib.sha1((model_dir / "encoder.pt").read_bytes()).hexdigest()[:10]
        self.redis = redis

    @torch.inference_mode()
    def encode(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        # Sort by length so each batch pads to a similar size.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(texts), batch):
            idx = order[s : s + batch]
            out.append((idx, self.model(**self.tok.encode_batch([texts[i] for i in idx])).numpy()))
        res = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, v in out:
            res[idx] = v
        return res

    def query(self, text: str) -> np.ndarray:
        if self.redis is None:
            return self.encode([text])[0]
        key = f"qv:{self.version}:{hashlib.sha1(text.strip().lower().encode()).hexdigest()}"
        hit = self.redis.get(key)
        if hit is not None:
            return np.frombuffer(hit, dtype=np.float32)
        v = self.encode([text])[0]
        self.redis.set(key, v.astype(np.float32).tobytes(), ex=86400)
        return v


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    import redis

    try:
        r = redis.Redis.from_url(config.REDIS_URL)
        r.ping()
    except Exception:
        r = None
    return Embedder(redis=r)
