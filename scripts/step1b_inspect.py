"""
OttoLatin - Adim 1b: Rakamlar ve belirsiz eslesmeler
Kullanim: python scripts/step1b_inspect.py --input out/step1/raw.parquet
Cikti: out/step1/inspect.txt  (bunu Claude'a yukle)
"""
import argparse
import re
from collections import Counter

import pandas as pd

RE_WEST = re.compile(r"[0-9]")
RE_ARABIC_INDIC = re.compile(r"[٠-٩]")
RE_PERSIAN = re.compile(r"[۰-۹]")
RE_ANY = re.compile(r"[0-9٠-٩۰-۹]+")

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="out/step1/raw.parquet")
ap.add_argument("--out", default="out/step1/inspect.txt")
args = ap.parse_args()

df = pd.read_parquet(args.input)[["ottoman", "turkish"]]
o, t = df["ottoman"], df["turkish"]
lines = []
p = lines.append

has = o.str.contains(RE_ANY)
p(f"Rakamli osmanlica satir: {has.sum():,} / {len(df):,}")
p(f"  Bati rakami (0-9):          {o.str.contains(RE_WEST).sum():,}")
p(f"  Arap-Hint rakami (٠-٩):     {o.str.contains(RE_ARABIC_INDIC).sum():,}")
p(f"  Farsca rakam (۰-۹):         {o.str.contains(RE_PERSIAN).sum():,}")
p(f"Rakamli turkce satir:          {t.str.contains(RE_WEST).sum():,}")
p(f"Osmanlica rakamli ama turkce rakamsiz: {(has & ~t.str.contains(RE_WEST)).sum():,}")

starts = o.str.match(r"^\s*[0-9٠-٩۰-۹]+[\s\.\-\)]")
p(f"Osmanlica rakamla BASLAYAN satir: {starts.sum():,}")
ends = o.str.contains(r"[0-9٠-٩۰-۹]+\s*$")
p(f"Osmanlica rakamla BITEN satir:    {ends.sum():,}")

tokens = Counter()
for s in o[has].sample(min(200_000, has.sum()), random_state=0):
    tokens.update(RE_ANY.findall(s))
p("\nEn sik 30 rakam dizisi (osmanlica tarafi):")
for tok, n in tokens.most_common(30):
    p(f"  {tok!r:15s} {n:,}")

p("\n--- 25 rakamli ornek ---")
for _, r in df[has].sample(25, random_state=1).iterrows():
    p(f"OT: {r.ottoman}\nTR: {r.turkish}\n")

p("\n--- Ayni osmanlica, farkli turkce: 15 ornek grup ---")
key = o.str.strip()
dup = df[key.duplicated(keep=False)].assign(k=key)
for k, g in list(dup.groupby("k"))[:15]:
    p(f"OT: {k}")
    for tr in g["turkish"].unique()[:5]:
        p(f"   -> {tr}")
    p("")

p("\n--- Ayni turkce, farkli osmanlica: 10 ornek grup ---")
tk = t.str.strip()
dup2 = df[tk.duplicated(keep=False)].assign(k=tk)
for k, g in list(dup2.groupby("k"))[:10]:
    p(f"TR: {k}")
    for ot in g["ottoman"].unique()[:5]:
        p(f"   -> {ot}")
    p("")

with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines[:12]))
print(f"\n[i] Tam rapor: {args.out}")
