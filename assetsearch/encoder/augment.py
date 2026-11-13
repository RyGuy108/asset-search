"""Turn COCO captions into (search query, generation prompt) training pairs.

A Magic Hour prompt is a long caption plus style boilerplate ("cinematic lighting, 35mm,
ultra detailed"); a search query is a few words typed from memory, often with a
different word for the same thing and sometimes the style. COCO gives five independent
captions per image, which supplies the paraphrase signal; this module adds the
prompt-vs-query shape on top.
"""

from __future__ import annotations

import random

from .tokenize import words

# Each family: (phrases a prompt might use, words a user might search with).
STYLE_FAMILIES: dict[str, tuple[list[str], list[str]]] = {
    "photo": (["realistic photo, sharp focus", "photorealistic, 50mm lens", "dslr photograph, natural light"],
              ["photo", "realistic", "photograph", "real"]),
    "cinematic": (["cinematic lighting, film still, anamorphic", "cinematic shot, dramatic lighting, 35mm film",
                   "movie still, moody color grade"], ["cinematic", "movie", "film", "dramatic"]),
    "watercolor": (["watercolor painting, soft washes on paper", "loose watercolour illustration"],
                   ["watercolor", "watercolour", "painted", "painting"]),
    "oil": (["oil painting on canvas, thick brushstrokes", "classical oil painting, impasto"],
            ["oil painting", "painting", "painted", "brushstrokes"]),
    "anime": (["anime style, cel shaded, studio quality", "japanese animation style key visual"],
              ["anime", "cartoon", "manga", "animated"]),
    "pixel": (["16-bit pixel art, limited palette", "retro pixel art sprite scene"],
              ["pixel art", "pixel", "8 bit", "retro game"]),
    "render3d": (["3d render, octane, soft global illumination", "cgi render, clay materials, studio light"],
                 ["3d", "render", "cgi", "3d render"]),
    "sketch": (["pencil sketch, cross hatching, graphite on paper", "ink line drawing, black and white"],
               ["sketch", "drawing", "pencil", "line art"]),
    "isometric": (["isometric illustration, clean vector, diorama", "isometric miniature diorama"],
                  ["isometric", "diorama", "miniature"]),
    "vintage": (["vintage 1970s film photo, faded colors, grain", "polaroid photo, retro, light leaks"],
                ["vintage", "retro", "old photo", "70s", "polaroid"]),
    "neon": (["neon cyberpunk night, glowing signs, rain", "synthwave neon lighting, magenta and cyan"],
             ["neon", "cyberpunk", "synthwave", "night"]),
}
QUALITY = ["ultra detailed", "8k", "highly detailed", "trending on artstation", "masterpiece", "4k", "hdr"]
STOP = set("a an the of on in at with and to is are its it his her their there this that some".split())


BOILERPLATE = [
    "no text, no watermark, no border, consistent character design, keep the same style as the reference",
    "high quality, balanced composition, rule of thirds, soft shadows, rich detail in the background, professional color grading",
    "static camera, slow push-in, subtle motion only, keep the illustration style, no new objects, no text",
]


def make_long_prompt(captions: list[str], rng: random.Random, style: str | None) -> str:
    """Real prompts run to 50-150 words: several facts about one scene plus boilerplate."""
    k = rng.randint(2, min(4, len(captions)))
    body = ". ".join(c.strip().rstrip(".") for c in rng.sample(captions, k))
    out = make_prompt(body, rng, style)
    if rng.random() < 0.5:
        out += ". " + rng.choice(BOILERPLATE)
    return out


def make_prompt(caption: str, rng: random.Random, style: str | None) -> str:
    text = caption.strip().rstrip(".")
    if rng.random() < 0.5:
        text = text[0].lower() + text[1:]
    parts = [text]
    if style:
        phrase = rng.choice(STYLE_FAMILIES[style][0])
        parts = [phrase, text] if rng.random() < 0.3 else [text, phrase]
    if rng.random() < 0.5:
        parts += rng.sample(QUALITY, rng.randint(1, 3))
    return ", ".join(parts)


def make_query(caption: str, rng: random.Random, style: str | None) -> str:
    ws = words(caption)
    r = rng.random()
    if r < 0.25:
        q = ws
    else:
        content = [w for w in ws if w not in STOP] or ws
        k = min(len(content), rng.randint(2, 5))
        start = rng.randint(0, max(0, len(content) - k)) if rng.random() < 0.3 else 0
        q = content[start : start + k] if rng.random() < 0.5 else sorted(rng.sample(content, k), key=content.index)
    q = list(q)
    if rng.random() < 0.1 and q:  # typo
        i = rng.randrange(len(q))
        w = q[i]
        if len(w) > 4:
            j = rng.randrange(1, len(w) - 1)
            q[i] = w[:j] + w[j + 1 :]
    if style and rng.random() < 0.45:
        tag = rng.choice(STYLE_FAMILIES[style][1])
        q = [tag, *q] if rng.random() < 0.5 else [*q, tag]
    return " ".join(q)


def random_style(rng: random.Random) -> str | None:
    return None if rng.random() < 0.3 else rng.choice(list(STYLE_FAMILIES))
