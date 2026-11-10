"""
OttoLatin - shared text normalization (use this everywhere in evaluation)

turkish_norm(s):
    Makes scholarly transliteration ("Huzûr-ı Sâmî", "şer‘an", "re’s") comparable with
    modern spelling ("huzur ı sami", "şeran", "res"):
      - Turkish-aware lowercasing (I -> ı, İ -> i)
      - circumflex/diacritics removed (â->a, î->i, û->u, ā->a, ū->u ...); ç ğ ı ö ş ü are KEPT
      - ayn/hamza marks and apostrophes removed (‘ ʻ ’ ' ʼ ` ´)
      - hyphen -> space, other punctuation removed, whitespace collapsed
Model output and reference must go through the SAME function before computing CER/WER.
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
    s = unicodedata.normalize("NFC", s)       # fold combined characters such as "i̇"
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
