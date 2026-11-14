"""Regenerate variants of a past generation through the Magic Hour API.

Flow: POST /assets/{id}/variants -> a variant_jobs row + an RQ job -> the worker plans N
prompts, submits them to Magic Hour, records each project id *before* waiting on it
(so a crashed or retried worker polls the renders it already paid for instead of
submitting them again), downloads the outputs and inserts them as new assets whose
parent is the source. New assets are searchable as soon as the job finishes.

Planning is retrieval-augmented: with `style_from` the variant keeps the source's
subject and borrows the style clauses of another of the user's assets; without an
instruction it varies the axes the original prompt leaves open (time of day, framing,
medium) instead of rerolling the same prompt.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import httpx
from psycopg import Connection

from . import config, db
from .encoder.augment import QUALITY

AXES = {
    "time": ["at golden hour", "at blue hour", "at night under streetlights", "on an overcast morning", "at sunrise"],
    "framing": ["wide establishing shot", "close-up", "low angle shot", "overhead top-down view", "medium shot, eye level"],
    "medium": ["realistic photo", "watercolor painting", "3d render, soft studio light", "anime key visual",
               "vintage 35mm film photo", "isometric diorama", "pencil sketch"],
}
AXIS_WORDS = {
    "time": r"\b(golden hour|sunset|sunrise|dawn|dusk|night|morning|noon|moonlit|blue hour)\b",
    "framing": r"\b(close-up|wide|aerial|low angle|overhead|medium shot|telephoto|full-length|waist-up|selfie)\b",
    "medium": r"\b(photo|photograph|painting|watercolou?r|render|anime|illustration|sketch|lithograph|cartoon|pixel art|drawing)\b",
}
STYLE_WORDS = re.compile(
    r"\b(lighting|light|lens|\d+mm|film|grain|palette|style|render\w*|painting|watercolou?r|illustration|photo\w*|sketch|"
    r"cinematic|anamorphic|bokeh|depth of field|detailed|8k|4k|hdr|shot|sharp focus|realistic|lithograph|inks?)\b", re.I)


def split_subject_style(prompt: str) -> tuple[str, list[str]]:
    """Split "a fox in snow, watercolor painting, soft light" into subject and style clauses."""
    clauses = [c.strip() for c in re.split(r",(?![^()]*\))|(?<=[.!?])\s+", prompt) if c.strip(" .")]
    subject = [c for i, c in enumerate(clauses) if i == 0 or not STYLE_WORDS.search(c)]
    style = [c for i, c in enumerate(clauses) if i > 0 and STYLE_WORDS.search(c)]
    return ", ".join(subject), style


def plan_variants(prompt: str, n: int = 4, instruction: str | None = None, style_from: str | None = None,
                  seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    subject, style = split_subject_style(prompt)
    if style_from:
        _, borrowed = split_subject_style(style_from)
        base = ", ".join([subject, *borrowed]) if borrowed else f"{subject}, in the style of: {style_from}"
        out = [base]
        open_axes = [a for a in ("time", "framing") if not re.search(AXIS_WORDS[a], base, re.I)]
        while len(out) < n:
            extra = rng.choice(AXES[rng.choice(open_axes)]) if open_axes else rng.choice(QUALITY)
            out.append(f"{base}, {extra}")
        return list(dict.fromkeys(out))[:n] if len(set(out)) >= n else out[:n]
    if instruction:
        return [f"{prompt.rstrip('. ')}. Change: {instruction}"] + [
            f"{subject}, {instruction}, {c}" for c in rng.sample(AXES["framing"], max(0, n - 1))
        ][: n - 1]
    open_axes = [a for a in AXES if not re.search(AXIS_WORDS[a], prompt, re.I)] or ["framing"]
    out = []
    for i in range(n):
        axis = open_axes[i % len(open_axes)]
        choice = rng.choice(AXES[axis])
        out.append(", ".join([subject, choice, *style]) if axis != "medium" else ", ".join([subject, choice]))
    return out


def similar_style_source(conn: Connection, user_id: str, source_id: int, style_query: str) -> str | None:
    """Pick the user's asset that best matches a style description ("like my lithograph ones")."""
    from .embed import default_embedder
    from .search import hybrid

    emb = default_embedder()
    hits = hybrid(conn, user_id, style_query, emb.query(style_query), k=3)
    for h in hits:
        if h.id != source_id:
            row = conn.execute("SELECT prompt FROM assets WHERE id = %s AND prompt IS NOT NULL", (h.id,)).fetchone()
            if row:
                return row[0]
    return None


EDIT_AXES = {
    "time": ["golden hour sunlight", "blue hour dusk light", "night, lit by warm streetlights", "soft overcast daylight"],
    "setting": ["a clean white studio sweep", "a sunlit concrete rooftop", "a rustic wooden table", "a marble countertop"],
    "medium": ["a watercolor painting", "a 3d clay render", "an anime key visual", "a vintage 35mm film photo"],
}
KEEP = ("Keep the main subject exactly as it is: same shape, proportions, colors, materials, logos and details. "
        "Do not add text.")


def plan_edits(prompt: str, n: int = 4, instruction: str | None = None, style_from: str | None = None,
               seed: int = 0) -> list[str]:
    """Edit instructions for the image editor. Each one says what to keep and what to change."""
    rng = random.Random(seed)
    if style_from:
        _, borrowed = split_subject_style(style_from)
        style = ", ".join(borrowed) or style_from
        base = (f"Redraw this exact image in this style: {style}. Keep the main subject's shape, proportions, "
                "details and the composition. Do not add text.")
        extras = ["", " Keep the original composition.", " Slightly closer crop.", " Slightly wider crop."]
        return [base + extras[i % len(extras)] for i in range(n)]
    if instruction:
        return [f"{instruction.rstrip('.')}. {KEEP}"] + [
            f"{instruction.rstrip('.')}, with {c}. {KEEP}" for c in rng.sample(EDIT_AXES["time"], max(0, n - 1))
        ][: n - 1]
    out = []
    axes = ["setting", "time", "medium"]
    for i in range(n):
        axis = axes[i % len(axes)]
        choice = rng.choice(EDIT_AXES[axis])
        verb = {"setting": "Place it on", "time": "Relight the scene with", "medium": "Render the whole image as"}[axis]
        out.append(f"{verb} {choice}. {KEEP}")
    return list(dict.fromkeys(out)) + out[len(set(out)):]


# ---------------------------------------------------------------- worker side

def _client():
    from magic_hour import Client

    if not config.MAGIC_HOUR_API_KEY:
        raise RuntimeError("MAGIC_HOUR_API_KEY is not set")
    return Client(token=config.MAGIC_HOUR_API_KEY)


DEFAULT_MODELS = {"regenerate": "z-image-turbo", "edit": "nano-banana-2"}


def run_variant_job(job_id: int, poll_s: float = 3.0, timeout_s: float = 900) -> dict:
    """RQ entry point. Safe to re-run: submitted project ids are persisted before polling."""
    from .embed import default_embedder
    from .ingest import insert_assets

    conn = db.connect()
    job = conn.execute(
        "SELECT user_id, source_id, prompts, mh_ids, status, mode, model FROM variant_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    if job is None or job[4] == "done":
        return {"status": job[4] if job else "missing"}
    user_id, source_id, prompts, mh_ids, _, mode, model = job
    model = model or DEFAULT_MODELS[mode]
    src = conn.execute("SELECT kind, key, path, url, search_text FROM assets WHERE id = %s", (source_id,)).fetchone()
    client = _client()
    conn.execute("UPDATE variant_jobs SET status = 'rendering' WHERE id = %s", (job_id,))

    mh_ids = list(mh_ids)
    upload = None
    for p in prompts[len(mh_ids):]:
        if mode == "edit":
            if upload is None:
                upload = client.v1.files.upload_file(src[2]) if src[2] else src[3]
            r = client.v1.ai_image_editor.create(assets={"image_file_paths": [upload]}, style={"prompt": p},
                                                 model=model, aspect_ratio="auto", image_count=1,
                                                 name=f"variant job {job_id}")
        else:
            r = client.v1.ai_image_generator.create(image_count=1, style={"prompt": p}, model=model,
                                                    aspect_ratio="1:1", resolution="1k", name=f"variant job {job_id}")
        mh_ids.append(r.id)
        conn.execute("UPDATE variant_jobs SET mh_ids = %s WHERE id = %s", (mh_ids, job_id))

    out_dir = Path(config.MEDIA_DIR) / user_id
    out_dir.mkdir(parents=True, exist_ok=True)
    new_assets, credits, deadline = [], 0.0, time.time() + timeout_s
    for i, (pid, p) in enumerate(zip(mh_ids, prompts)):
        while True:
            proj = client.v1.image_projects.get(id=pid)
            if proj.status in ("complete", "error", "canceled") or time.time() > deadline:
                break
            time.sleep(poll_s)
        if proj.status != "complete":
            continue
        credits += proj.credits_charged or 0
        url = proj.downloads[0].url
        path = out_dir / f"{pid}.png"
        if not path.exists():
            path.write_bytes(httpx.get(url, timeout=60).content)
        new_assets.append(dict(
            key=f"variant-{pid}", mh_id=pid, kind=src[0], model=model, path=str(path), url=url,
            credits=proj.credits_charged, parents=[src[1]], prompt=p,
            tool="ai_image_editor" if mode == "edit" else "ai_image_generator",
            # An edit instruction alone ("Relight the scene...") says nothing about the subject,
            # so edited variants also carry the source's text.
            derived=src[4] if mode == "edit" else None,
        ))

    # parents are resolved by key inside insert_assets; include the source row's key mapping.
    ids = insert_assets(conn, user_id, new_assets, default_embedder()) if new_assets else {}
    for k in ids.values():
        conn.execute("UPDATE assets SET parent_ids = ARRAY[%s]::bigint[] WHERE id = %s", (source_id, k))
    status = "done" if len(ids) == len(prompts) else ("partial" if ids else "failed")
    conn.execute(
        "UPDATE variant_jobs SET status = %s, result_ids = %s, credits = %s, finished_at = now() WHERE id = %s",
        (status, list(ids.values()), credits, job_id),
    )
    return {"status": status, "result_ids": list(ids.values()), "credits": credits}


def create_job(conn: Connection, user_id: str, source_id: int, n: int, instruction: str | None,
               style_query: str | None, mode: str = "regenerate") -> tuple[int, list[str]]:
    row = conn.execute("SELECT prompt, search_text FROM assets WHERE id = %s AND user_id = %s",
                       (source_id, user_id)).fetchone()
    if row is None:
        raise KeyError(source_id)
    prompt = row[0] or row[1]
    style_from = similar_style_source(conn, user_id, source_id, style_query) if style_query else None
    planner = plan_edits if mode == "edit" else plan_variants
    prompts = planner(prompt, n=n, instruction=instruction, style_from=style_from, seed=source_id)
    job_id = conn.execute(
        "INSERT INTO variant_jobs (user_id, source_id, instruction, prompts, mode, model) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
        (user_id, source_id, instruction or style_query, json.dumps(prompts), mode, DEFAULT_MODELS[mode]),
    ).fetchone()[0]
    return job_id, prompts
