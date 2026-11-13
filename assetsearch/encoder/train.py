"""Train the asset encoder on COCO caption pairs.

    python -m assetsearch.encoder.train --captions data/captions_val2014.json --out models/encoder

COCO's 40,504 val2014 images are split by image id into train / dev / test (the test
split is also what scripts/eval_coco.py scores), so no image contributes captions to
both training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from .augment import make_prompt, make_query, random_style
from .model import AssetEncoder, info_nce
from .tokenize import Tokenizer


def split_coco(path: Path) -> dict[str, dict[int, list[str]]]:
    d = json.loads(path.read_text())
    caps: dict[int, list[str]] = defaultdict(list)
    for a in d["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    ids = sorted(caps)
    random.Random(0).shuffle(ids)
    test, dev = set(ids[:3000]), set(ids[3000:4000])
    out: dict[str, dict[int, list[str]]] = {"train": {}, "dev": {}, "test": {}}
    for i in ids:
        out["test" if i in test else "dev" if i in dev else "train"][i] = caps[i]
    return out


def make_pairs(split: dict[int, list[str]], rng: random.Random) -> tuple[list[str], list[str], list[int]]:
    qs, ds, ids = [], [], []
    for img, caps in split.items():
        if len(caps) < 2:
            continue
        a, b = rng.sample(caps, 2)
        style = random_style(rng)
        qs.append(make_query(a, rng, style))
        ds.append(make_prompt(b, rng, style))
        ids.append(img)
    return qs, ds, ids


@torch.no_grad()
def recall_at(model: AssetEncoder, tok: Tokenizer, split: dict[int, list[str]], k: int = 10) -> float:
    model.eval()
    qs, ds, _ = make_pairs(split, random.Random(1))
    enc = lambda xs: torch.cat([model(**tok.encode_batch(xs[i : i + 512])) for i in range(0, len(xs), 512)])
    q, d = enc(qs), enc(ds)
    top = (q @ d.T).topk(k, dim=1).indices
    hit = (top == torch.arange(len(qs))[:, None]).any(1).float().mean().item()
    model.train()
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=Path, default=Path("data/captions_val2014.json"))
    ap.add_argument("--out", type=Path, default=Path("models/encoder"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bag-only", action="store_true", help="ablation: no Transformer")
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    splits = split_coco(args.captions)
    train_caps = [c for cs in splits["train"].values() for c in cs]
    tok = Tokenizer.build(train_caps)
    model = AssetEncoder(tok.vocab_size, dim=args.dim, out_dim=args.dim, layers=args.layers, bag_only=args.bag_only)
    dense = [p for n, p in model.named_parameters() if not n.startswith(("word", "ngram"))]
    sparse = [model.word.weight, model.ngram.weight]
    opt = torch.optim.AdamW([{"params": dense, "weight_decay": 0.01}, {"params": sparse, "weight_decay": 0.0}], lr=args.lr)
    # Each epoch resamples one fresh (query, prompt) pair per image, so 12 epochs see
    # 12 different augmentations of every training image.
    steps_per_epoch = len(splits["train"]) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    print(f"vocab {tok.vocab_size:,}  train images {len(splits['train']):,}  params {sum(p.numel() for p in model.parameters()):,}")

    rng = random.Random(0)
    best = 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        qs, ds, ids = make_pairs(splits["train"], rng)
        order = list(range(len(qs)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = order[s * args.batch : (s + 1) * args.batch]
            q = model(**tok.encode_batch([qs[i] for i in idx]))
            d = model(**tok.encode_batch([ds[i] for i in idx]))
            loss = info_nce(q, d, model.logit_scale, torch.tensor([ids[i] for i in idx]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        r = recall_at(model, tok, splits["dev"])
        print(f"epoch {epoch + 1:2d}  loss {tot / steps_per_epoch:.3f}  dev R@10 (1k imgs) {r:.3f}  {time.time() - t0:.0f}s", flush=True)
        if r > best:
            best = r
            torch.save({"config": model.config, "state": model.state_dict()}, args.out / "encoder.pt")
            tok.save(args.out / "tokenizer.json")
    print(f"best dev R@10 {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
