"""How much of the source does each variant mode keep?

For every finished variant job, compare each output with its source image:

* SIFT inliers: local features matched (Lowe ratio test) and verified with a RANSAC
  homography. A product photo edited in place keeps dozens to hundreds of geometric
  correspondences with its source; a fresh render of the same prompt keeps almost none.
* palette similarity: correlation of 3-D colour histograms (HSV, 8x8x8), 1.0 = same palette.

    python scripts/eval_variants.py --user ryan
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assetsearch import db  # noqa: E402

SIFT = cv2.SIFT_create(nfeatures=2000)


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = 768 / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def sift_inliers(a: np.ndarray, b: np.ndarray) -> int:
    ka, da = SIFT.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db_ = SIFT.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db_ is None or len(ka) < 4 or len(kb) < 4:
        return 0
    pairs = cv2.BFMatcher().knnMatch(da, db_, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    return int(mask.sum()) if mask is not None else 0


def palette(a: np.ndarray, b: np.ndarray) -> float:
    h = [cv2.normalize(cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 8],
                                    [0, 180, 0, 256, 0, 256]), None).flatten() for x in (a, b)]
    return float(cv2.compareHist(h[0], h[1], cv2.HISTCMP_CORREL))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="ryan")
    ap.add_argument("--out", type=Path, default=Path("results/variants_eval.json"))
    args = ap.parse_args()
    conn = db.connect()
    jobs = conn.execute(
        """SELECT j.id, j.mode, j.model, j.instruction, j.credits, s.path, s.key, j.result_ids
           FROM variant_jobs j JOIN assets s ON s.id = j.source_id
           WHERE j.user_id = %s AND j.status IN ('done', 'partial') ORDER BY j.id""", (args.user,)).fetchall()
    rows = []
    for jid, mode, model, instr, credits, spath, skey, results in jobs:
        src = load(spath)
        for rid in results:
            p = conn.execute("SELECT path FROM assets WHERE id = %s", (rid,)).fetchone()[0]
            out = load(p)
            rows.append({"job": jid, "mode": mode, "model": model, "source": skey, "style": bool(instr and mode),
                         "sift_inliers": sift_inliers(src, out), "palette": round(palette(src, out), 3),
                         "credits_per_image": round((credits or 0) / max(1, len(results)), 1)})
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "images": len(rs),
            "median_sift_inliers": statistics.median(r["sift_inliers"] for r in rs),
            "share_with_20+_inliers": round(sum(r["sift_inliers"] >= 20 for r in rs) / len(rs), 2),
            "median_palette": round(statistics.median(r["palette"] for r in rs), 3),
            "credits_per_image": statistics.median(r["credits_per_image"] for r in rs),
        }
        print(mode, summary[mode])
    args.out.write_text(json.dumps({"summary": summary, "images": rows}, indent=1))


if __name__ == "__main__":
    main()
