"""HTTP API.

    uvicorn assetsearch.api:app --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from . import config, db, search
from .embed import default_embedder
from .ingest import insert_assets

pool: ConnectionPool | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pool
    with db.connect() as c:
        db.migrate(c)
    pool = ConnectionPool(config.DATABASE_URL, min_size=2, max_size=16, kwargs={"autocommit": True},
                          configure=register_vector, open=True)
    default_embedder()  # load the model before the first request
    yield
    pool.close()


app = FastAPI(title="asset-search", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

COLS = "id, key, mh_id, kind, tool, model, prompt, search_text, parent_ids, path, url, credits, created_at"


def _row(r) -> dict:
    d = dict(zip([c.strip() for c in COLS.split(",")], r))
    d["thumb"] = f"/media/{d['id']}" if d["path"] else d["url"]
    d.pop("path")
    return d


class AssetIn(BaseModel):
    key: str
    kind: Literal["image", "video", "audio"]
    tool: str
    prompt: str | None = None
    derived: str | None = Field(None, description="text for assets with no prompt, e.g. face swaps")
    model: str | None = None
    mh_id: str | None = None
    url: str | None = None
    credits: float | None = None
    parents: list[str] = []


@app.post("/users/{user_id}/assets")
def add_assets(user_id: str, assets: list[AssetIn]):
    with pool.connection() as c:
        ids = insert_assets(c, user_id, [a.model_dump() for a in assets], default_embedder())
    return {"ids": ids}


@app.get("/users/{user_id}/search")
def search_assets(user_id: str, q: str = Query(..., min_length=1), k: int = Query(24, le=100),
                  kind: Literal["image", "video", "audio"] | None = None,
                  mode: Literal["hybrid", "vector", "keyword"] = "hybrid"):
    t0 = time.perf_counter()
    emb = default_embedder()
    with pool.connection() as c:
        if mode == "keyword":
            hits = search.keyword(c, user_id, q, k, kind=kind)
        else:
            qv = emb.query(q)
            hits = search.vector(c, user_id, qv, k, kind) if mode == "vector" else search.hybrid(c, user_id, q, qv, k, kind)
        rows = {r[0]: r for r in c.execute(f"SELECT {COLS} FROM assets WHERE id = ANY(%s)", ([h.id for h in hits],))}
    results = [{**_row(rows[h.id]), "score": round(h.score, 4)} for h in hits if h.id in rows]
    return {"q": q, "mode": mode, "ms": round((time.perf_counter() - t0) * 1000, 1), "results": results}


@app.get("/users/{user_id}/assets/{asset_id}")
def get_asset(user_id: str, asset_id: int):
    with pool.connection() as c:
        r = c.execute(f"SELECT {COLS} FROM assets WHERE id = %s AND user_id = %s", (asset_id, user_id)).fetchone()
        if not r:
            raise HTTPException(404)
        kids = c.execute(f"SELECT {COLS} FROM assets WHERE %s = ANY(parent_ids) ORDER BY created_at DESC", (asset_id,)).fetchall()
        parents = c.execute(f"SELECT {COLS} FROM assets WHERE id = ANY(%s)", (r[8],)).fetchall()
    return {**_row(r), "parents": [_row(p) for p in parents], "children": [_row(k) for k in kids]}


class VariantReq(BaseModel):
    n: int = Field(4, ge=1, le=8)
    instruction: str | None = Field(None, description='e.g. "make it night time"')
    style_like: str | None = Field(None, description='borrow the style of your best match, e.g. "my lithograph scenes"')
    dry_run: bool = False


@app.post("/users/{user_id}/assets/{asset_id}/variants")
def make_variants(user_id: str, asset_id: int, req: VariantReq):
    from .variants import create_job, plan_variants

    with pool.connection() as c:
        if req.dry_run:
            r = c.execute("SELECT coalesce(prompt, search_text) FROM assets WHERE id = %s AND user_id = %s",
                          (asset_id, user_id)).fetchone()
            if not r:
                raise HTTPException(404)
            return {"prompts": plan_variants(r[0], req.n, req.instruction, seed=asset_id)}
        try:
            job_id, prompts = create_job(c, user_id, asset_id, req.n, req.instruction, req.style_like)
        except KeyError:
            raise HTTPException(404)
    from redis import Redis
    from rq import Queue

    Queue("variants", connection=Redis.from_url(config.REDIS_URL)).enqueue(
        "assetsearch.variants.run_variant_job", job_id, job_id=f"variant-{job_id}", job_timeout=900, retry=None)
    return {"job_id": job_id, "prompts": prompts}


@app.get("/jobs/{job_id}")
def job_status(job_id: int):
    with pool.connection() as c:
        r = c.execute("SELECT id, user_id, source_id, prompts, status, mh_ids, result_ids, credits, error FROM variant_jobs WHERE id = %s",
                      (job_id,)).fetchone()
    if not r:
        raise HTTPException(404)
    keys = ["id", "user_id", "source_id", "prompts", "status", "mh_ids", "result_ids", "credits", "error"]
    return dict(zip(keys, r))


@app.get("/media/{asset_id}")
def media(asset_id: int):
    with pool.connection() as c:
        r = c.execute("SELECT path FROM assets WHERE id = %s", (asset_id,)).fetchone()
    if not r or not r[0] or not Path(r[0]).exists():
        raise HTTPException(404)
    return FileResponse(r[0])


@app.get("/healthz")
def healthz():
    with pool.connection() as c:
        n = c.execute("SELECT count(*) FROM assets").fetchone()[0]
    return {"ok": True, "assets": n, "encoder": default_embedder().version}
