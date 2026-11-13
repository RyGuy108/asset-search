import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assetsearch.encoder.tokenize import Tokenizer, words  # noqa: E402
from assetsearch.search import Hit, rrf  # noqa: E402
from assetsearch.variants import plan_variants, split_subject_style  # noqa: E402
from scripts.eval_library import expand, metrics  # noqa: E402


def test_tokenizer_handles_unknown_and_typos():
    tok = Tokenizer.build(["a dog on a beach", "a dog in the snow"], min_count=1)
    b = tok.encode_batch(["dgo beach", "bioluminescent jellyfish"])
    assert b["words"].shape[0] == 2
    assert (b["ngrams"][1, 0] > 0).sum() > 0  # unseen words still get n-gram ids
    assert words("Watercolor, 35mm!") == ["watercolor", "35mm"]


def test_rrf_rewards_agreement():
    a = [Hit(1, 0), Hit(2, 0), Hit(3, 0)]
    b = [Hit(3, 0), Hit(4, 0)]
    fused = [h.id for h in rrf(a, b, k=4)]
    assert fused[0] in (1, 3) and set(fused) == {1, 2, 3, 4}
    assert fused.index(3) < fused.index(2)


def test_split_subject_style():
    subj, style = split_subject_style("a fox in the snow, watercolor painting, soft morning light, 8k")
    assert subj == "a fox in the snow"
    assert "watercolor painting" in style


def test_plan_variants_varies_open_axes_only():
    ps = plan_variants("a fox in the snow, watercolor painting", n=4, seed=1)
    assert len(ps) == 4 and len(set(ps)) == 4
    assert all(p.startswith("a fox in the snow") for p in ps)


def test_plan_variants_style_from_borrows_style():
    ps = plan_variants("a fox in the snow", n=2, style_from="a castle, two-colour lithograph illustration, fine ink lines")
    assert "lithograph" in ps[0] and ps[0].startswith("a fox in the snow")


def test_relevance_expands_to_same_prompt_and_children():
    lib = [
        {"key": "a0", "prompt": "a red car"}, {"key": "a1", "prompt": "a red car"},
        {"key": "swap", "derived": "face swap onto a red car", "parents": ["a1"]},
        {"key": "b", "prompt": "a blue car"},
    ]
    assert expand(lib, ["a0"]) == {"a0", "a1", "swap"}


def test_metrics():
    m = metrics(["x", "r1", "y"], {"r1", "r2"})
    assert m["MRR"] == 0.5 and m["R@1"] == 0 and m["R@5"] == 0.5 and m["hit@10"] == 1
