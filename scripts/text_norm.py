"""
OttoLatin - ortak metin normalizasyonu (Enes: degerlendirmede bunu kullan)

turkish_norm(s):
    Bilimsel transliterasyon ("Huzûr-ı Sâmî", "şer‘an", "re’s") ile modern yazimi
    ("huzur ı sami", "şeran", "res") karsilastirilabilir hale getirir:
      - Turkce kurallarina gore kucuk harf (I -> ı, İ -> i)
      - sapka/isaretler silinir (â->a, î->i, û->u, ā->a, ū->u ...); ç ğ ı ö ş ü KORUNUR
      - ayn/hemze isaretleri ve kesme isaretleri silinir (‘ ʻ ’ ' ʼ ` ´)
      - tire -> bosluk, diger noktalama silinir, bosluklar tek bosluga indirilir
Model ciktisi ve referans AYNI fonksiyondan gecirilip CER/WER hesaplanmali.
"""
import re
import unicodedata

KEEP = {"ç", "ğ", "ı", "ö", "ş", "ü"}
APOS = "‘’ʻʼʿʾ'`´"
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_WS = re.compile(r"\s+")


def tr_lower(s):
    return s.replace("I", "ı").replace("İ", "i").lower()


def strip_marks(s):
    out = []
    for ch in s:
        if ch in KEEP:
            out.append(ch)
            continue
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        out.append(unicodedata.normalize("NFC", base) if base else "")
    return "".join(out)


def turkish_norm(s):
    s = unicodedata.normalize("NFC", s)
    s = tr_lower(s)
    s = unicodedata.normalize("NFC", s)       # "i̇" gibi birlesik karakterleri toparla
    s = s.replace("i̇", "i")
    s = strip_marks(s)
    for a in APOS:
        s = s.replace(a, "")
    s = s.replace("-", " ")
    s = RE_PUNCT.sub(" ", s).replace("_", " ")
    return RE_WS.sub(" ", s).strip()


if __name__ == "__main__":
    for x in ["Huzur-ı Sâmî-i Hazret-i Sadâret-penâhîye", "şer‘an bir nesne", "re’s öküzüm",
              "Bağdâd Kalʻası muhâfazasında", "İnsanlar her zaman personellere karşı kabadırlar.",
              "mütekāʻidlerinden Bağdâd'da"]:
        print(f"{x!r:50s} -> {turkish_norm(x)!r}")
