"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Asset = {
  id: number;
  key: string;
  kind: "image" | "video" | "audio";
  tool: string;
  model: string | null;
  prompt: string | null;
  search_text: string;
  thumb: string | null;
  score?: number;
  created_at: string;
};
type Detail = Asset & { parents: Asset[]; children: Asset[] };
type Job = { id: number; status: string; prompts: string[]; result_ids: number[]; credits: number | null };

function media(a: Asset) {
  if (!a.thumb) return null;
  return a.thumb.startsWith("/") ? `${API}${a.thumb}` : a.thumb;
}

function Thumb({ a, onClick }: { a: Asset; onClick?: () => void }) {
  const src = media(a);
  return (
    <button className="card" onClick={onClick} title={a.search_text}>
      {src && a.kind === "image" && <img src={src} alt="" loading="lazy" />}
      {src && a.kind === "video" && <video src={src} muted loop autoPlay playsInline />}
      {(!src || a.kind === "audio") && <div className="ph">{a.kind}</div>}
      <span className="tag">{a.tool.replaceAll("_", " ")}</span>
    </button>
  );
}

export default function Page() {
  const [user, setUser] = useState("ryan");
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [compare, setCompare] = useState(false);
  const [res, setRes] = useState<{ hybrid: Asset[]; keyword: Asset[]; ms: number } | null>(null);
  const [open, setOpen] = useState<Detail | null>(null);
  const [instruction, setInstruction] = useState("");
  const [styleLike, setStyleLike] = useState("");
  const [variantMode, setVariantMode] = useState<"edit" | "regenerate">("edit");
  const [job, setJob] = useState<Job | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  async function run(query: string) {
    if (!query.trim()) return setRes(null);
    const params = (mode: string) =>
      new URLSearchParams({ q: query, mode, k: "24", ...(kind ? { kind } : {}) }).toString();
    const [h, k] = await Promise.all([
      fetch(`${API}/users/${user}/search?${params("hybrid")}`).then((r) => r.json()),
      compare ? fetch(`${API}/users/${user}/search?${params("keyword")}`).then((r) => r.json()) : { results: [] },
    ]);
    setRes({ hybrid: h.results, keyword: k.results, ms: h.ms });
  }

  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => run(q), 150);
  }, [q, kind, compare, user]);

  async function openAsset(id: number) {
    setJob(null);
    setOpen(await fetch(`${API}/users/${user}/assets/${id}`).then((r) => r.json()));
  }

  async function variants() {
    if (!open) return;
    const r = await fetch(`${API}/users/${user}/assets/${open.id}/variants`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ n: 4, mode: variantMode, instruction: instruction || null, style_like: styleLike || null }),
    }).then((r) => r.json());
    const poll = async () => {
      const j: Job = await fetch(`${API}/jobs/${r.job_id}`).then((r) => r.json());
      setJob(j);
      if (["queued", "rendering"].includes(j.status)) setTimeout(poll, 2500);
      else openAsset(open.id);
    };
    poll();
  }

  return (
    <main>
      <header>
        <h1>Find anything you've generated</h1>
        <p>Describe it the way you remember it. Search runs over your past Magic Hour renders.</p>
        <div className="bar">
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="that rainy neon street one…" />
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">all</option>
            <option value="image">images</option>
            <option value="video">videos</option>
            <option value="audio">audio</option>
          </select>
          <input className="user" value={user} onChange={(e) => setUser(e.target.value)} aria-label="user" />
        </div>
        <label className="cmp">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} /> compare with keyword search
        </label>
      </header>

      {res && (
        <section className={compare ? "cols" : ""}>
          <div>
            <h2>
              Hybrid <small>{res.ms} ms</small>
            </h2>
            <div className="grid">
              {res.hybrid.map((a) => (
                <Thumb key={a.id} a={a} onClick={() => openAsset(a.id)} />
              ))}
            </div>
          </div>
          {compare && (
            <div>
              <h2>Keyword</h2>
              {res.keyword.length === 0 && <p className="empty">No results</p>}
              <div className="grid">
                {res.keyword.map((a) => (
                  <Thumb key={a.id} a={a} onClick={() => openAsset(a.id)} />
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {open && (
        <aside onClick={() => setOpen(null)}>
          <div className="panel" onClick={(e) => e.stopPropagation()}>
            <button className="x" onClick={() => setOpen(null)}>×</button>
            <Thumb a={open} />
            <p className="prompt">{open.prompt ?? open.search_text}</p>
            <p className="meta">
              {open.tool} {open.model ? `· ${open.model}` : ""} · {new Date(open.created_at).toLocaleDateString()}
            </p>
            {open.kind === "image" && (
              <div className="variants">
                <h3>Make variants</h3>
                <input value={instruction} onChange={(e) => setInstruction(e.target.value)} placeholder="change… (optional, e.g. at night)" />
                <input value={styleLike} onChange={(e) => setStyleLike(e.target.value)} placeholder="in the style of my… (optional)" />
                <select value={variantMode} onChange={(e) => setVariantMode(e.target.value as "edit" | "regenerate")}>
                  <option value="edit">Keep this exact subject (image editor, ~50 credits each)</option>
                  <option value="regenerate">New takes on the prompt (text-to-image, ~5 credits each)</option>
                </select>
                <button onClick={variants} disabled={!!job && ["queued", "rendering"].includes(job.status)}>
                  Generate 4 with Magic Hour
                </button>
                {job && (
                  <p className="meta">
                    {job.status}
                    {job.credits ? ` · ${job.credits} credits` : ""}
                  </p>
                )}
              </div>
            )}
            {open.parents.length > 0 && (
              <>
                <h3>Made from</h3>
                <div className="grid sm">
                  {open.parents.map((a) => (
                    <Thumb key={a.id} a={a} onClick={() => openAsset(a.id)} />
                  ))}
                </div>
              </>
            )}
            {open.children.length > 0 && (
              <>
                <h3>Variants and edits</h3>
                <div className="grid sm">
                  {open.children.map((a) => (
                    <Thumb key={a.id} a={a} onClick={() => openAsset(a.id)} />
                  ))}
                </div>
              </>
            )}
          </div>
        </aside>
      )}
    </main>
  );
}
