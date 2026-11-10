"""
OttoLatin - Adim 2: Normalizasyon + filtreleme + train/dev/test split
Kullanim:
    python scripts/step2_clean_split.py
    (varsayilan girdi: out/step1/raw.parquet)

Ciktilar:
    data/processed/train.jsonl, dev.jsonl, test.jsonl, train_small.jsonl
        her satir: {"id": ..., "ottoman": ..., "turkish": ...}
    out/step2/clean_report.json   -> her kuralin kac satir attigi, split boyutlari, sizinti kontrolu
    out/step2/dropped_examples.txt -> her kuraldan atilan ornekler (kontrol icin)

Kurallar (osmanlica taraf):
    - Unicode NFC
    - Arapca yeh (ي U+064A) ve alef maksura (ى U+0649) -> Farsca yeh (ی U+06CC)
    - keheh (ک U+06A9) -> Arapca kaf (ك U+0643)   [veride zaten ك kullaniliyor]
    - ae (ە U+06D5) -> he (ه U+0647)
    - harekeler, tatweel ve gorunmez karakterler (ZWNJ dahil) silinir
    - bosluklar tek bosluga indirilir
    Turkce taraf: NFC + bosluk duzeltme. Buyuk/kucuk harf KORUNUR.

Filtreler (sirayla):
    too_long      : herhangi bir taraf > MAX_BYTES (UTF-8 byte; ByT5 icin)
    few_letters   : osmanlicada Arap harfi orani < 0.3 (neredeyse sadece sayi/noktalama)
    digit_mismatch: osmanlicadaki sayilar != turkcedeki sayilar (hizalama bozuk)
    len_ratio     : turkce/osmanlica karakter orani < 0.5 veya > 3
    duplicate     : normalizasyon sonrasi ayni (ottoman, turkish) cifti

Split: normalize osmanlica metnin hash'ine gore (deterministik).
    Ayni osmanlica metnin tum Turkce varyantlari (or. "Ates" / "ates") ayni split'e duser.
    Veri ortak bir cumle havuzundan birlestirildigi icin ek olarak: test/dev'deki herhangi bir
    cumleyi (>= MIN_SENT karakter) iceren train satirlari atilir -> cumle seviyesinde sizinti yok.
"""
import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd

MAX_BYTES = 512
SEED = 42
DEV_PER_MILLE = 10    # %1
TEST_PER_MILLE = 10   # %1
SMALL_N = 300_000
MIN_SENT = 15         # bu uzunluktan kisa cumle parcalari sizinti sayilmaz

CHAR_MAP = str.maketrans({
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ک": "ك",  # ک -> ك
    "ە": "ه",  # ە -> ه
})
RE_REMOVE = re.compile(r"[ً-ٰٟـ​-‏‪-‮⁦-⁩﻿]")
RE_WS = re.compile(r"\s+")
RE_ARABIC_LETTER = re.compile(r"[ء-غف-يٱ-ۓە]")
RE_AR_NUM = re.compile(r"[٠-٩]+")
RE_LAT_NUM = re.compile(r"[0-9]+")
AR2LAT = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
RE_SENT = re.compile(r"(?<=[\.\!\?;؛،,])\s+")


def norm_ot(s):
    s = unicodedata.normalize("NFC", s).translate(CHAR_MAP)
    s = RE_REMOVE.sub("", s)
    return RE_WS.sub(" ", s).strip()


def norm_tr(s):
    s = unicodedata.normalize("NFC", s)
    return RE_WS.sub(" ", s).strip()


def letter_ratio(s):
    nonspace = len(s.replace(" ", ""))
    return len(RE_ARABIC_LETTER.findall(s)) / nonspace if nonspace else 0.0


def bucket(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/step1/raw.parquet")
    ap.add_argument("--data_out", default="data/processed")
    ap.add_argument("--report_out", default="out/step2")
    ap.add_argument("--no_overlap_check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.data_out, exist_ok=True)
    os.makedirs(args.report_out, exist_ok=True)

    df = pd.read_parquet(args.input)[["ottoman", "turkish"]].reset_index(drop=True)
    df["id"] = df.index
    rep = {"input_rows": len(df)}
    print(f"[i] Girdi: {len(df):,} satir")

    # --- Normalizasyon (once kac satiri degistirdigini say)
    before = df["ottoman"].copy()
    df["ottoman"] = df["ottoman"].map(norm_ot)
    df["turkish"] = df["turkish"].map(norm_tr)
    rep["ottoman_rows_changed_by_norm"] = int((before != df["ottoman"]).sum())
    del before
    print(f"[i] Normalizasyon {rep['ottoman_rows_changed_by_norm']:,} osmanlica satiri degistirdi")

    dropped = {}
    examples = defaultdict(list)

    def apply(name, mask):
        nonlocal df
        bad = df[mask]
        dropped[name] = int(len(bad))
        examples[name] = bad.head(15)[["ottoman", "turkish"]].values.tolist()
        df = df[~mask]
        print(f"    - {name:15s} {len(bad):>9,} atildi  -> kalan {len(df):,}")

    print("[i] Filtreler:")
    apply("empty", (df["ottoman"] == "") | (df["turkish"] == ""))
    ob = df["ottoman"].map(lambda x: len(x.encode("utf-8")))
    tb = df["turkish"].map(lambda x: len(x.encode("utf-8")))
    apply("too_long", (ob > MAX_BYTES) | (tb > MAX_BYTES))
    apply("few_letters", df["ottoman"].map(letter_ratio) < 0.3)
    on = df["ottoman"].map(lambda x: [n.translate(AR2LAT) for n in RE_AR_NUM.findall(x)])
    tn = df["turkish"].map(RE_LAT_NUM.findall)
    apply("digit_mismatch", on != tn)
    ratio = df["turkish"].str.len() / df["ottoman"].str.len()
    apply("len_ratio", (ratio < 0.5) | (ratio > 3))
    apply("duplicate", df.duplicated(subset=["ottoman", "turkish"]))
    rep["dropped"] = dropped
    rep["kept_rows"] = len(df)

    # --- Split
    b = df["ottoman"].map(bucket)
    split = pd.Series("train", index=df.index)
    split[b < TEST_PER_MILLE] = "test"
    split[(b >= TEST_PER_MILLE) & (b < TEST_PER_MILLE + DEV_PER_MILLE)] = "dev"
    df["split"] = split

    parts = {k: df[df["split"] == k][["id", "ottoman", "turkish"]] for k in ["train", "dev", "test"]}
    rep["splits_before_decontam"] = {k: len(v) for k, v in parts.items()}

    # --- Cumle seviyesi temizleme: test/dev cumlelerini iceren train satirlarini at,
    #     test cumlelerini iceren dev satirlarini at.
    def sents(s):
        return [c for c in RE_SENT.split(s.lower()) if len(c) >= MIN_SENT]

    def collect(frame):
        out = set()
        for s in frame["turkish"]:
            out.update(sents(s))
        return out

    print("[i] Cumle seviyesi temizleme (train'den test/dev cumleleri cikariliyor)...")
    test_s = collect(parts["test"])
    m_dev = parts["dev"]["turkish"].map(lambda s: any(c in test_s for c in sents(s))).astype(bool)
    parts["dev"] = parts["dev"].loc[~m_dev]
    held_s = test_s | collect(parts["dev"])
    m_tr = parts["train"]["turkish"].map(lambda s: any(c in held_s for c in sents(s))).astype(bool)
    parts["train"] = parts["train"].loc[~m_tr]
    rep["decontam_removed"] = {"dev": int(m_dev.sum()), "train": int(m_tr.sum())}
    print(f"    dev'den {int(m_dev.sum()):,}, train'den {int(m_tr.sum()):,} satir cikarildi")

    parts["train_small"] = parts["train"].sample(min(SMALL_N, len(parts["train"])), random_state=SEED)
    rep["splits"] = {k: len(v) for k, v in parts.items()}

    # sanity: ayni osmanlica metin birden fazla split'te olmamali
    rep["exact_ottoman_overlap_train_test"] = int(parts["test"]["ottoman"].isin(set(parts["train"]["ottoman"])).sum())

    for k, v in parts.items():
        path = os.path.join(args.data_out, f"{k}.jsonl")
        v.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"[i] {k:12s} {len(v):>9,} satir -> {path}")

    # --- Cumle seviyesinde sizinti kontrolu (veri birlestirilmis cumlelerden olusuyor)
    if not args.no_overlap_check:
        print("[i] Cumle seviyesi sizinti kontrolu (birkac dakika surebilir)...")
        train_h = set()
        for s in parts["train"]["turkish"]:
            for c in RE_SENT.split(s.lower()):
                if len(c) >= MIN_SENT:
                    train_h.add(hash(c))
        tot = hit = 0
        for s in parts["test"]["turkish"]:
            for c in RE_SENT.split(s.lower()):
                if len(c) >= MIN_SENT:
                    tot += 1
                    hit += hash(c) in train_h
        rep["test_sentences_seen_in_train_pct"] = round(100 * hit / tot, 2) if tot else None
        print(f"    test cumlelerinin %{rep['test_sentences_seen_in_train_pct']}'i train'de de geciyor")

    with open(os.path.join(args.report_out, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.report_out, "dropped_examples.txt"), "w", encoding="utf-8") as f:
        for k, rows in examples.items():
            f.write(f"===== {k} ({dropped[k]:,}) =====\n")
            for o, t in rows:
                f.write(f"OT: {o}\nTR: {t}\n\n")
    print(f"\n[i] Rapor: {os.path.join(args.report_out, 'clean_report.json')}")


if __name__ == "__main__":
    main()
