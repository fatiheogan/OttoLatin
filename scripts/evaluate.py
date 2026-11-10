"""
OttoLatin - Degerlendirme scripti

Tahmin dosyasi formati (jsonl): {"id": "...", "prediction": "..."}
Referans dosyasi: data/eval/<set>.jsonl  ({"id", "ottoman", "turkish"})

Kullanim:
  # tek model, tek set
  python scripts/evaluate.py --ref data/eval/test_2k.jsonl --pred preds/rule/test_2k.jsonl

  # tek model, tum setler (preds/<model>/<set>.jsonl duzeni)
  python scripts/evaluate.py --model_dir preds/rule

  # iki modeli karsilastir (paired bootstrap)
  python scripts/evaluate.py --model_dir preds/byt5 --compare preds/rule

Ciktilar: terminalde tablo + <model_dir>/results.json + <model_dir>/worst_<set>.txt

Metrikler (korpus seviyesi, yuzde):
  CER  = toplam karakter edit mesafesi / toplam referans karakteri
  WER  = toplam kelime edit mesafesi / toplam referans kelimesi
  chrF = sacrebleu chrF (0-100, yuksek iyi)
  *_norm: hem tahmin hem referans text_norm.turkish_norm()'dan gecirildikten sonra
  CI  : 1000 ornekli bootstrap, %95 guven araligi
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import turkish_norm  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein
    def lev(a, b):
        return Levenshtein.distance(a, b)
except ImportError:  # yavas ama bagimliliksiz
    def lev(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

try:
    from sacrebleu.metrics import CHRF
    _chrf = CHRF()
except ImportError:
    _chrf = None

SETS = ["test_2k", "riqa", "naskh", "ot2tr"]
N_BOOT = 1000
SEED = 12345


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def per_item(refs, hyps):
    """Her ornek icin (char_err, char_len, word_err, word_len) ham ve norm."""
    items = []
    for r, h in zip(refs, hyps):
        rn, hn = turkish_norm(r), turkish_norm(h)
        items.append((
            lev(r, h), len(r), lev(r.split(), h.split()), len(r.split()),
            lev(rn, hn), len(rn), lev(rn.split(), hn.split()), len(rn.split()),
        ))
    return items


def agg(items, idx=None):
    idx = range(len(items)) if idx is None else idx
    s = [0] * 8
    for i in idx:
        for k in range(8):
            s[k] += items[i][k]
    pct = lambda e, n: 100 * e / n if n else 0.0
    return {"cer": pct(s[0], s[1]), "wer": pct(s[2], s[3]),
            "cer_norm": pct(s[4], s[5]), "wer_norm": pct(s[6], s[7])}


def bootstrap_ci(items, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    N = len(items)
    samples = {k: [] for k in ["cer", "wer", "cer_norm", "wer_norm"]}
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        a = agg(items, idx)
        for k in samples:
            samples[k].append(a[k])
    ci = {}
    for k, v in samples.items():
        v.sort()
        ci[k] = (v[int(0.025 * n)], v[int(0.975 * n) - 1])
    return ci


def paired_bootstrap(items_a, items_b, key="cer_norm", n=N_BOOT, seed=SEED):
    """H0: A, B'den iyi degil. p = A'nin B'den kotu/esit oldugu orneklem orani (dusuk CER iyi)."""
    rng = random.Random(seed)
    N = len(items_a)
    worse = 0
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        if agg(items_a, idx)[key] >= agg(items_b, idx)[key]:
            worse += 1
    return worse / n


def align(ref_rows, pred_rows):
    pred = {str(p["id"]): p.get("prediction", "") or "" for p in pred_rows}
    missing = [r["id"] for r in ref_rows if str(r["id"]) not in pred]
    refs = [r["turkish"] for r in ref_rows]
    hyps = [pred.get(str(r["id"]), "") for r in ref_rows]   # eksik tahmin = bos string (cezalandirilir)
    return refs, hyps, missing


def length_buckets(ref_rows, items):
    buckets = {"kisa (<40)": [], "orta (40-120)": [], "uzun (>120)": []}
    for i, r in enumerate(ref_rows):
        L = len(r["turkish"])
        key = "kisa (<40)" if L < 40 else ("orta (40-120)" if L <= 120 else "uzun (>120)")
        buckets[key].append(i)
    return {k: {"n": len(v), **({m: round(x, 2) for m, x in agg(items, v).items()} if v else {})}
            for k, v in buckets.items()}


def write_worst(path, ref_rows, refs, hyps, items, k=30):
    order = sorted(range(len(items)), key=lambda i: -(items[i][4] / max(items[i][5], 1)))[:k]
    with open(path, "w", encoding="utf-8") as f:
        for i in order:
            cer = 100 * items[i][4] / max(items[i][5], 1)
            f.write(f"### {ref_rows[i]['id']}   norm-CER={cer:.1f}\n"
                    f"OT  : {ref_rows[i]['ottoman']}\nREF : {refs[i]}\nPRED: {hyps[i]}\n\n")


def eval_set(ref_path, pred_path, worst_path=None):
    ref_rows = load_jsonl(ref_path)
    refs, hyps, missing = align(ref_rows, load_jsonl(pred_path))
    items = per_item(refs, hyps)
    res = {k: round(v, 2) for k, v in agg(items).items()}
    res["ci95"] = {k: [round(a, 2), round(b, 2)] for k, (a, b) in bootstrap_ci(items).items()}
    if _chrf:
        res["chrf"] = round(_chrf.corpus_score(hyps, [refs]).score, 2)
        res["chrf_norm"] = round(_chrf.corpus_score([turkish_norm(h) for h in hyps],
                                                    [[turkish_norm(r) for r in refs]]).score, 2)
    res["n"] = len(ref_rows)
    res["missing_predictions"] = len(missing)
    res["by_length"] = length_buckets(ref_rows, items)
    if worst_path:
        write_worst(worst_path, ref_rows, refs, hyps, items)
    return res, items


def fmt_row(name, r):
    ci = r["ci95"]
    chrf = f"{r.get('chrf_norm', float('nan')):6.2f}" if "chrf_norm" in r else "   -  "
    return (f"{name:9s} {r['n']:>6} | CER {r['cer']:6.2f} | WER {r['wer']:6.2f} | "
            f"CERn {r['cer_norm']:6.2f} [{ci['cer_norm'][0]:.2f}-{ci['cer_norm'][1]:.2f}] | "
            f"WERn {r['wer_norm']:6.2f} | chrFn {chrf}"
            + (f" | EKSIK {r['missing_predictions']}" if r["missing_predictions"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--pred")
    ap.add_argument("--model_dir", help="preds/<model> klasoru; icinde <set>.jsonl dosyalari")
    ap.add_argument("--compare", help="karsilastirilacak ikinci model klasoru")
    ap.add_argument("--eval_dir", default="data/eval")
    args = ap.parse_args()

    if args.ref and args.pred:
        res, _ = eval_set(args.ref, args.pred)
        print(fmt_row(os.path.basename(args.ref).replace(".jsonl", ""), res))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if not args.model_dir:
        ap.error("--ref/--pred veya --model_dir ver")

    results, all_items = {}, {}
    print(f"\n=== {args.model_dir} ===")
    for s in SETS:
        pp = os.path.join(args.model_dir, f"{s}.jsonl")
        rp = os.path.join(args.eval_dir, f"{s}.jsonl")
        if not os.path.exists(pp):
            print(f"{s:9s}  (tahmin dosyasi yok, atlandi)")
            continue
        res, items = eval_set(rp, pp, os.path.join(args.model_dir, f"worst_{s}.txt"))
        results[s], all_items[s] = res, items
        print(fmt_row(s, res))

    if args.compare:
        print(f"\n=== Paired bootstrap: {args.model_dir} vs {args.compare} (norm-CER) ===")
        results["comparison"] = {"against": args.compare}
        for s in results:
            if s == "comparison":
                continue
            pp = os.path.join(args.compare, f"{s}.jsonl")
            if not os.path.exists(pp):
                continue
            refs, hyps, _ = align(load_jsonl(os.path.join(args.eval_dir, f"{s}.jsonl")), load_jsonl(pp))
            items_b = per_item(refs, hyps)
            p = paired_bootstrap(all_items[s], items_b)
            a, b = agg(all_items[s])["cer_norm"], agg(items_b)["cer_norm"]
            sig = "ANLAMLI" if p < 0.05 and abs(a - b) >= 1.0 else "anlamli degil"
            results["comparison"][s] = {"cer_norm_a": round(a, 2), "cer_norm_b": round(b, 2),
                                        "diff": round(a - b, 2), "p": p, "verdict": sig}
            print(f"{s:9s} A={a:6.2f}  B={b:6.2f}  fark={a - b:+6.2f}  p={p:.3f}  -> {sig}")

    with open(os.path.join(args.model_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[i] {os.path.join(args.model_dir, 'results.json')}")


if __name__ == "__main__":
    main()
