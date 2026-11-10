"""
OttoLatin - error analysis (automatic, over all predictions)

Usage:
  python scripts/error_analysis.py --model_dir preds/rule
  python scripts/error_analysis.py --model_dir preds/rule preds/qwen_fewshot   # several models

What it does:
  Classifies the character edit operations (Levenshtein) between normalized reference and prediction:
    vowel_front_back    : i<->ı, ü<->u, ö<->o (front/back vowel confusion)
    vowel_other         : other vowel errors (a<->e, o<->u, missing/extra vowel ...)
    consonant_ambiguous : consonants whose Ottoman letter has several readings (k/g/ğ/n, t/d, p/b, c/ç, s/z, h)
    consonant_other     : other consonant errors
    word_boundary       : inserted/deleted spaces (joined vs. separate spelling)
    other               : digits, punctuation, etc.
  Also:
    case_diacritic_share_pct : share of raw CER that disappears after normalization (case, circumflex, hyphens)
    label_noise (test_2k only): last Ottoman word is an infinitive (-مق/-مك) but the Turkish reference does not end in -mak/-mek

Output: <model_dir>/error_analysis.json, error_analysis.png (per model), out/error_analysis_compare.png (several models)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import turkish_norm  # noqa: E402

from rapidfuzz.distance import Levenshtein

SETS = ["test_2k", "riqa", "naskh", "ot2tr"]
VOWELS = set("aeıioöuü")
FRONT_BACK = [{"i", "ı"}, {"ü", "u"}, {"ö", "o"}]
AMBIG_CONS = [set("kgğn"), set("td"), set("pb"), set("cç"), set("sz"), set("h")]
CATS = ["vowel_front_back", "vowel_other", "consonant_ambiguous", "consonant_other", "word_boundary", "other"]
CAT_LABEL = {"vowel_front_back": "Vowel: front/back (i/ı, ü/u, ö/o)", "vowel_other": "Vowel: other",
             "consonant_ambiguous": "Consonant: ambiguous letter (k/g/ğ, t/d …)", "consonant_other": "Consonant: other",
             "word_boundary": "Word boundary (space)", "other": "Other (digits, punctuation)"}
RE_PUNCT = re.compile(r"[^\w]+$")


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def classify(op, a, b):
    """op: replace/insert/delete ; a: ref char (or ''), b: hyp char (or '')"""
    chars = {c for c in (a, b) if c}
    if " " in chars:
        return "word_boundary"
    if chars and chars <= VOWELS:
        if op == "replace" and any(chars <= fb for fb in FRONT_BACK):
            return "vowel_front_back"
        return "vowel_other"
    if chars & VOWELS:                       # vowel <-> consonant swap
        return "vowel_other"
    if all(c.isalpha() for c in chars):
        if op == "replace" and any(chars <= g for g in AMBIG_CONS):
            return "consonant_ambiguous"
        if op != "replace" and chars <= set("hğ"):   # silent h / ğ inserted or deleted
            return "consonant_ambiguous"
        return "consonant_other"
    return "other"


def analyse(ref_rows, preds):
    cnt = Counter()
    raw_err = raw_len = norm_err = norm_len = 0
    for r in ref_rows:
        ref, hyp = r["turkish"], preds.get(str(r["id"]), "")
        raw_err += Levenshtein.distance(ref, hyp)
        raw_len += len(ref)
        rn, hn = turkish_norm(ref), turkish_norm(hyp)
        norm_err += Levenshtein.distance(rn, hn)
        norm_len += len(rn)
        for op in Levenshtein.editops(rn, hn):
            a = rn[op.src_pos] if op.tag in ("replace", "delete") else ""
            b = hn[op.dest_pos] if op.tag in ("replace", "insert") else ""
            cnt[classify(op.tag, a, b)] += 1
    total = sum(cnt.values()) or 1
    raw_cer, norm_cer = 100 * raw_err / max(raw_len, 1), 100 * norm_err / max(norm_len, 1)
    return {
        "n": len(ref_rows),
        "cer_raw": round(raw_cer, 2), "cer_norm": round(norm_cer, 2),
        "case_diacritic_share_pct": round(100 * max(raw_cer - norm_cer, 0) / raw_cer, 1) if raw_cer else 0.0,
        "error_ops": total,
        "share_pct": {k: round(100 * cnt[k] / total, 1) for k in CATS},
    }


def label_noise(ref_rows):
    hits = []
    for r in ref_rows:
        ow = RE_PUNCT.sub("", r["ottoman"].split()[-1]) if r["ottoman"].split() else ""
        tw = RE_PUNCT.sub("", r["turkish"].split()[-1]).lower() if r["turkish"].split() else ""
        if ow.endswith(("مق", "مك")) and not tw.endswith(("mak", "mek", "mık", "mik", "muk", "mük")):
            hits.append({"id": r["id"], "ottoman": r["ottoman"], "turkish": r["turkish"]})
    return {"n": len(hits), "pct": round(100 * len(hits) / max(len(ref_rows), 1), 2), "examples": hits[:10]}


def plot(results, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed, chart skipped")
        return
    colors = ["#2a6f97", "#61a5c2", "#c75146", "#e9a29b", "#8a8a8a", "#cfcfcf"]
    labels = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(labels) + 1.8))
    left = [0.0] * len(labels)
    for cat, col in zip(CATS, colors):
        vals = [results[l]["share_pct"][cat] for l in labels]
        ax.barh(labels, vals, left=left, color=col, label=CAT_LABEL[cat], edgecolor="white", linewidth=0.6)
        for i, v in enumerate(vals):
            if v >= 7:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if cat in ("vowel_front_back", "consonant_ambiguous", "other") else "black")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Share of character errors (%, on normalized CER)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[i] Chart: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", nargs="+", required=True)
    ap.add_argument("--eval_dir", default="data/eval")
    args = ap.parse_args()

    refs = {s: load(os.path.join(args.eval_dir, f"{s}.jsonl")) for s in SETS
            if os.path.exists(os.path.join(args.eval_dir, f"{s}.jsonl"))}
    compare = {}
    for md in args.model_dir:
        name = os.path.basename(os.path.normpath(md))
        out = {}
        print(f"\n=== {name} ===")
        print(f"{'set':8s} {'CERn':>6s} {'case%':>8s}  " + "  ".join(f"{c[:12]:>12s}" for c in CATS))
        for s, rows in refs.items():
            pp = os.path.join(md, f"{s}.jsonl")
            if not os.path.exists(pp):
                continue
            preds = {str(p["id"]): p.get("prediction", "") or "" for p in load(pp)}
            res = analyse(rows, preds)
            out[s] = res
            compare[f"{name} · {s}"] = res
            d = res["share_pct"]
            print(f"{s:8s} {res['cer_norm']:6.2f} {res['case_diacritic_share_pct']:7.1f}%  "
                  + "  ".join(f"{d[c]:11.1f}%" for c in CATS))
        if "test_2k" in refs:
            out["label_noise_test_2k"] = label_noise(refs["test_2k"])
        with open(os.path.join(md, "error_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        plot({s: v for s, v in out.items() if s in SETS}, os.path.join(md, "error_analysis.png"),
             f"Error breakdown – {name}")

    if "test_2k" in refs:
        ln = label_noise(refs["test_2k"])
        print(f"\n[i] test_2k label noise (infinitive -> imperative/other): {ln['n']} rows ({ln['pct']}%)")
        for e in ln["examples"][:5]:
            print(f"     {e['ottoman'][:50]}  ->  {e['turkish'][:50]}")
    if len(args.model_dir) > 1:
        os.makedirs("out", exist_ok=True)
        plot(compare, "out/error_analysis_compare.png", "Error breakdown – model comparison")


if __name__ == "__main__":
    main()
