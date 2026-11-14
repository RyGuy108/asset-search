"""Collect every real Magic Hour generation made in this workspace into one asset library.

These are actual renders (images, face swaps, image-to-video and text-to-video clips)
from five earlier projects, with the prompt or inputs that produced each one. They are
the "past generations" a user searches in the evaluation.

    python scripts/import_library.py --root /mnt/project-files --out data/library.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def moderation(root: Path):
    base = root / "content-moderation"
    rows = [json.loads(line) for line in open(base / "data/gen/ledger.jsonl")]
    paths = {r["id"]: r["paths"] for r in rows if "paths" in r}
    by_key, out = {}, []
    for r in rows:
        if r.get("kind") == "image":
            ps = paths.get(r["id"], [])
            for i, p in enumerate(ps or [None]):
                a = dict(key=f"{r['key']}_{i}", mh_id=r["id"], kind="image", tool="ai_image_generator",
                         prompt=r["prompt"], path=str(base / p) if p else None, project="content-moderation",
                         credits=r["credits"] / max(1, len(ps)))
                out.append(a)
                if p:
                    by_key[Path(p).stem] = a
    for r in rows:
        if r.get("kind") != "swap":
            continue
        src, dst = by_key.get(Path(r["src"]).stem), by_key.get(Path(r["dst"]).stem)
        out.append(dict(
            key=r["key"], mh_id=r["id"], kind="image", tool="face_swap_photo", prompt=None,
            # A face swap has no prompt; it inherits the text of both inputs.
            derived=f"face swap. face: {src['prompt'] if src else 'uploaded photo'}. "
                    f"scene: {dst['prompt'] if dst else 'uploaded photo'}",
            parents=[x["key"] for x in (src, dst) if x],
            path=str(base / paths[r["id"]][0]) if paths.get(r["id"]) else None, project="content-moderation", credits=r["credits"],
        ))
    return out


def book_scenes(root: Path):
    base = root / "book-scenes"
    src = (base / "tools/style.py").read_text()
    mod = ast.parse(src)
    consts = {}
    for node in mod.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    style = consts.get("STYLE", "")
    out = []
    for k, p in consts.get("REFS", {}).items():
        out.append(dict(key=f"ref_{k}", kind="image", tool="ai_image_generator", prompt=f"{p} {style}",
                        path=str(base / f"public/books/time-machine/refs/{k}.jpg"), project="book-scenes"))
    scenes = json.loads((base / "data/time-machine/scenes.json").read_text())
    for s in scenes:
        out.append(dict(key=f"scene_{s['id']}", kind="image", tool="ai_image_editor", prompt=f"{s['prompt']} {style}",
                        parents=[f"ref_{r}" for r in s.get("refs", [])],
                        path=str(base / f"public/books/time-machine/scenes/{s['id']}.jpg"), project="book-scenes"))
    motion = {}
    m = re.search(r"MOTION=(\{.*?\n\})", (base / "tools/animate.py").read_text(), re.S)
    if m:
        motion = ast.literal_eval(m.group(1))
    for clip in sorted((base / "public/books/time-machine/clips").glob("*.mp4")):
        sid = clip.stem
        parent = next((s for s in scenes if s["id"] == sid), None)
        out.append(dict(key=f"clip_{sid}", kind="video", tool="image_to_video", model="ltx-2",
                        prompt=motion.get(sid), derived=parent["prompt"] if parent else None,
                        parents=[f"scene_{sid}"], path=str(clip), project="book-scenes"))
    return out


def bio(root: Path):
    base = root / "bio-explainer"
    out = []
    for plan in sorted((base / "plans").glob("*.json")):
        p = json.loads(plan.read_text())
        name = plan.stem
        for i, kf in enumerate(p["keyframes"]):
            out.append(dict(key=f"{name}_key{i}", kind="image", tool="ai_image_editor", model="gpt-image-2",
                            prompt=f"{p['concept']}: {kf}", parents=["bloop_ref"],
                            path=str(base / f"out/{name}/key{i}.png"), project="bio-explainer"))
        for i, mo in enumerate(p["motion"]):
            out.append(dict(key=f"{name}_shot{i}", kind="video", tool="image_to_video", prompt=mo,
                            derived=f"{p['concept']}, {p['subtitle']}", parents=[f"{name}_key{i}", f"{name}_key{i + 1}"],
                            path=str(base / f"out/{name}/shot{i}.mp4"), project="bio-explainer"))
        out.append(dict(key=f"{name}_music", kind="audio", tool="ai_music_generator",
                        prompt=f"background music for a cartoon science explainer about {p['concept'].lower()}",
                        path=str(base / f"out/{name}/music.mp3"), project="bio-explainer"))
    out.append(dict(key="bloop_ref", kind="image", tool="ai_image_generator",
                    prompt="mascot reference sheet: Bloop, a cute translucent jelly blob cell character with a curl, "
                           "round glasses and blush, hand-drawn cartoon on paper",
                    path=str(base / "mascot/bloop_ref.png"), project="bio-explainer"))
    return out


def strava(root: Path):
    base = root / "strava-wrapped"
    ids = json.loads((base / "magic_hour_project_ids.json").read_text())
    out = []
    for node in ast.walk(ast.parse((base / "shots.py").read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict":
            kw = {k.arg: k.value for k in node.keywords}
            if "prompt" in kw and isinstance(kw["prompt"], ast.Constant) and kw["prompt"].value and isinstance(kw.get("id"), ast.Constant):
                sid = kw["id"].value
                out.append(dict(key=f"run_{sid}", mh_id=ids.get(sid), kind="video", tool="text_to_video", model="ltx-2",
                                prompt=kw["prompt"].value, path=None, project="strava-wrapped"))
    return out


def outfits(root: Path):
    base = root / "outfit-planner"
    closet = json.loads((base / "data/closet.json").read_text())
    out = []
    for c in closet:
        out.append(dict(key=f"closet_{c['id']}", kind="image", tool="ai_image_generator",
                        prompt=f"product photo of a {c['name'].lower()} ({c['category']}, {c['color']}), laid flat on a plain background",
                        path=str(base / "public" / c["photo"].lstrip("/")), project="outfit-planner"))
        out.append(dict(key=f"cutout_{c['id']}", kind="image", tool="image_background_remover", prompt=None,
                        derived=f"background removed cutout of {c['name'].lower()}", parents=[f"closet_{c['id']}"],
                        path=str(base / "public" / c["image"].lstrip("/")), project="outfit-planner"))
    out.append(dict(key="standin", kind="image", tool="ai_image_generator",
                    prompt="full-length photo of a man standing in a bedroom facing the camera, neutral pose, phone photo",
                    path=str(base / "public/me/standin.png"), project="outfit-planner"))
    for f in sorted((base / "looks/fits").glob("*.png")):
        out.append(dict(key=f"fit_{f.stem}", kind="image", tool="ai_image_editor", model="nano-banana-2", prompt=None,
                        derived="virtual try-on: the stand-in person wearing an outfit from the closet",
                        parents=["standin"], path=str(f), project="outfit-planner"))
    for f in sorted((base / "looks/clips").glob("*.mp4")):
        out.append(dict(key=f"spin_{f.stem}", kind="video", tool="image_to_video", model="ltx-2",
                        prompt="a person checking their outfit in the mirror, turning slightly",
                        parents=[f"fit_{f.stem}"], path=str(f), project="outfit-planner"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/project-files"))
    ap.add_argument("--out", type=Path, default=Path("data/library.jsonl"))
    args = ap.parse_args()
    assets = []
    for fn in (moderation, book_scenes, bio, strava, outfits):
        got = fn(args.root)
        print(f"{fn.__name__:12s} {len(got):4d}")
        assets += got
    keys = [a["key"] for a in assets]
    assert len(keys) == len(set(keys)), "duplicate keys"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for a in assets:
            f.write(json.dumps(a) + "\n")
    print(f"total        {len(assets):4d} -> {args.out}")


if __name__ == "__main__":
    main()
