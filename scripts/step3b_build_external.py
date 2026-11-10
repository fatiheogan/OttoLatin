"""
OttoLatin - Adim 3b: Dis test setlerini ortak formata cevir
Kullanim: python scripts/step3b_build_external.py
Girdi : data/external/extracted/  (step3a'nin actigi klasorler)
Cikti : data/external/naskh.jsonl, riqa.jsonl, ot2tr.jsonl
        her satir: {"id", "source", "ottoman", "turkish", "turkish_norm", "ottoman_bytes"}
        out/step3/external_report.json
Osmanlica tarafa egitim verisiyle AYNI normalizasyon uygulanir (step2'deki norm_ot).
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2_clean_split import norm_ot, MAX_BYTES  # noqa: E402
from text_norm import turkish_norm  # noqa: E402

EXT = "data/external/extracted"
OUT_DIR = "data/external"
REP = "out/step3/external_report.json"
EXTRA = str.maketrans({"ۀ": "ه", "ہ": "ه", "ۂ": "ه"})  # ۀ ہ ۂ -> ه
RE_WS = re.compile(r"\s+")


PUNCT_MAP = {"\u06D4": "."}  # ۔ -> .
TRAIN_CHARS = set()          # main() icinde doldurulur
REMOVED = Counter()


def n_ot(s):
    # Arapca sunum bicimlerini (U+FB50-FEFF, or. ﺎ) temel harflere cevir
    s = "".join(unicodedata.normalize("NFKC", c) if "\uFB50" <= c <= "\uFEFF" else c for c in s)
    s = norm_ot(s.translate(EXTRA))
    if TRAIN_CHARS:
        out = []
        for c in s:
            c = PUNCT_MAP.get(c, c)
            if c in TRAIN_CHARS or c.isspace():
                out.append(c)
            else:
                REMOVED[c] += 1
        s = RE_WS.sub(" ", "".join(out)).strip()
    return s


def n_tr(s):
    return RE_WS.sub(" ", s).strip()


def read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def find_dir(name):
    for root, dirs, _ in os.walk(EXT):
        if os.path.basename(root) == name:
            return root
    return None


def build_line_set(name):
    """Riqa / Naskh: <name>/ottoman/X.gt.txt  <->  <name>/turkish/X.gt.txt"""
    base = find_dir(name)
    rows, miss = [], 0
    if not base:
        print(f"[!] {name} klasoru bulunamadi")
        return rows, {"missing_dir": True}
    od, td = os.path.join(base, "ottoman"), os.path.join(base, "turkish")
    for fn in sorted(os.listdir(od)):
        tp = os.path.join(td, fn)
        if not os.path.exists(tp):
            miss += 1
            continue
        o, t = n_ot(read(os.path.join(od, fn))), n_tr(read(tp))
        if o and t:
            rows.append({"id": f"{name}/{fn.replace('.gt.txt', '')}", "source": name, "ottoman": o, "turkish": t})
    return rows, {"pairs_missing_turkish": miss}


def build_ot2tr():
    """Sayfa dosyalari: X-osm-N.txt <-> X-tur-N.txt, satir satir hizali"""
    base = None
    for root, _, files in os.walk(EXT):
        if any("-osm-" in f for f in files):
            base = root
            break
    rows, info = [], {"pages": 0, "pages_line_mismatch": [], "pages_missing_tur": 0}
    if not base:
        print("[!] Ot2Tr dosyalari bulunamadi")
        return rows, info
    for fn in sorted(os.listdir(base)):
        if "-osm-" not in fn:
            continue
        tp = os.path.join(base, fn.replace("-osm-", "-tur-"))
        if not os.path.exists(tp):
            info["pages_missing_tur"] += 1
            continue
        ol = [l for l in read(os.path.join(base, fn)).splitlines() if l.strip()]
        tl = [l for l in read(tp).splitlines() if l.strip()]
        info["pages"] += 1
        if len(ol) != len(tl):
            info["pages_line_mismatch"].append(f"{fn} ({len(ol)} vs {len(tl)})")
            continue
        page = fn.replace(".txt", "").replace("-osm-", "-")
        for i, (o, t) in enumerate(zip(ol, tl)):
            o, t = n_ot(o), n_tr(t)
            if o and t:
                rows.append({"id": f"ot2tr/{page}/{i}", "source": "ot2tr", "ottoman": o, "turkish": t})
    return rows, info


def main():
    os.makedirs(os.path.dirname(REP), exist_ok=True)
    report = {}

    # egitim verisindeki karakter envanteri (kapsam kontrolu icin)
    train_chars = Counter()
    tp = "data/processed/train_small.jsonl"
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            for line in f:
                train_chars.update(json.loads(line)["ottoman"])
        TRAIN_CHARS.update(train_chars)
        print(f"[i] Egitim karakter envanteri: {len(TRAIN_CHARS)} farkli karakter")
    else:
        print("[!] train_small.jsonl yok -> egitimde gorulmemis karakterler silinmeyecek")

    sets = {"naskh": build_line_set("naskh"), "riqa": build_line_set("riqa"), "ot2tr": build_ot2tr()}
    for name, (rows, info) in sets.items():
        for r in rows:
            r["turkish_norm"] = turkish_norm(r["turkish"])
            r["ottoman_bytes"] = len(r["ottoman"].encode("utf-8"))
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        unseen = Counter()
        for r in rows:
            for ch in r["ottoman"]:
                if train_chars and ch not in train_chars and not ch.isspace():
                    unseen[ch] += 1
        info.update({
            "rows": len(rows),
            "over_max_bytes": sum(r["ottoman_bytes"] > MAX_BYTES for r in rows),
            "chars_unseen_in_train": {f"{c} U+{ord(c):04X}": n for c, n in unseen.most_common(20)},
        })
        report[name] = info
        print(f"[i] {name:6s} {len(rows):>5,} satir -> {path}   (>{MAX_BYTES}B: {info['over_max_bytes']})")
        if rows:
            r = rows[min(3, len(rows) - 1)]
            print(f"      OT : {r['ottoman'][:120]}\n      TR : {r['turkish'][:120]}\n      NRM: {r['turkish_norm'][:120]}")

    report["removed_chars_all_sets"] = {f"{c} U+{ord(c):04X}": n for c, n in REMOVED.most_common()}
    print(f"[i] Osmanlica girdiden silinen karakterler: {dict(REMOVED.most_common(10))}")
    with open(REP, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[i] Rapor: {REP}")


if __name__ == "__main__":
    main()
