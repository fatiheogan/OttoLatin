"""
OttoLatin - B0: Kural tabanli baseline (harf eslemesi + basit unlu sezgileri)
Ogrenme yok. Amac: "en kotu makul sistem" referansi. Unlu tahmini yapamadigi icin yuksek CER bekleniyor.

Kullanim: python scripts/baseline_rule.py
Cikti   : preds/rule/<set>.jsonl   ({"id", "prediction"})
"""
import json
import os
import re

EVAL = "data/eval"
OUT = "preds/rule"
SETS = ["test_2k", "riqa", "naskh", "ot2tr"]

CONS = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "c", "چ": "ç", "ح": "h", "خ": "h",
    "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "j", "س": "s", "ش": "ş", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "غ": "ğ", "ف": "f", "ق": "k", "ك": "k", "ک": "k",
    "گ": "g", "ڭ": "n", "ل": "l", "م": "m", "ن": "n",
}
SILENT = {"ع": "", "ء": "", "ئ": "", "ؤ": "u"}
ALEF = {"ا", "آ", "أ", "إ", "ٱ"}
PUNCT = {"،": ",", "؛": ";", "؟": "?", "٪": "%"}
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
RE_TOKEN = re.compile(r"(\s+)")


def word(w):
    # kelime basi elif + vav/ye -> tek unlu (او -> u, ای -> i)
    if len(w) > 1 and w[0] in ALEF and w[1] == "و":
        w = "\x01" + w[2:]
    elif len(w) > 1 and w[0] in ALEF and w[1] in ("ی", "ي"):
        w = "\x02" + w[2:]
    out = []
    n = len(w)
    for i, ch in enumerate(w):
        first, last = i == 0, i == n - 1
        if ch == "\x01":
            out.append("u")
        elif ch == "\x02":
            out.append("i")
        elif ch in ALEF:
            out.append("a")
        elif ch == "و":
            out.append("v" if first else "u")
        elif ch in ("ی", "ي", "ى"):
            out.append("y" if first and n > 1 else "i")
        elif ch in ("ه", "ة"):
            out.append("e" if last and n > 1 else "h")
        elif ch in CONS:
            c = CONS[ch]
            out.append(c)
            # iki unsuz arasina 'e' sok (Osmanlicada yazilmayan unlu icin kaba tahmin)
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt in CONS:
                out.append("e")
        elif ch in SILENT:
            out.append(SILENT[ch])
        elif ch in PUNCT:
            out.append(PUNCT[ch])
        else:
            out.append(ch.translate(DIGITS))
    s = "".join(out)
    return s.replace("ae", "a").replace("aa", "a")


def translit(text):
    return "".join(t if t.isspace() else word(t) for t in RE_TOKEN.split(text))


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in SETS:
        src = os.path.join(EVAL, f"{s}.jsonl")
        if not os.path.exists(src):
            print(f"[!] {src} yok, atlandi")
            continue
        n = 0
        with open(src, encoding="utf-8") as f, open(os.path.join(OUT, f"{s}.jsonl"), "w", encoding="utf-8") as g:
            for line in f:
                r = json.loads(line)
                g.write(json.dumps({"id": r["id"], "prediction": translit(r["ottoman"])}, ensure_ascii=False) + "\n")
                n += 1
        print(f"[i] {s:8s} {n:>6,} tahmin -> {OUT}/{s}.jsonl")


if __name__ == "__main__":
    for x in ["انسانلر هر زمان پرسونللره قارشو قبادرلر.", "یارین قبول ایدییورم.", "حضور سامی حضرت صدارت پناهییه"]:
        print(f"{x}  ->  {translit(x)}")
    main()
