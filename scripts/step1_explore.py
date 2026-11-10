"""
OttoLatin - Adim 1: Veriyi indir + ilk bakis
Kullanim:
    pip install datasets pandas pyarrow matplotlib
    python step1_explore.py                    # HF'den indirir
    python step1_explore.py --input raw.parquet   # yerel dosyadan okur

Ciktilar (out/step1/ klasorune):
    raw.parquet        -> ham veri (sonraki adimlar bunu kullanacak, tekrar indirmeye gerek yok)
    stats.json         -> tum istatistikler
    sample_300.csv     -> elle inceleme icin 300 rastgele satir (Excel'de acilir)
    length_hist.png    -> uzunluk histogrami
"""
import argparse
import json
import os
import re
import unicodedata

import pandas as pd

HF_NAME = "Kotek37/ottoman-turkish-4m"
SEED = 42

# Karakter siniflari
RE_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
RE_LATIN = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû]")
RE_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
ZERO_WIDTH = ["​", "‌", "‍", "‎", "‏", "﻿"]
TATWEEL = "ـ"
HARAKAT = re.compile(r"[ً-ٰٟ]")  # hareke / vowel isaretleri


def load(path):
    if path:
        print(f"[i] Yerel dosyadan okunuyor: {path}")
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True)
    from datasets import load_dataset
    print(f"[i] Hugging Face'den indiriliyor: {HF_NAME}")
    ds = load_dataset(HF_NAME, split="train")
    return ds.to_pandas()


def pct(n, total):
    return round(100 * n / total, 3) if total else 0.0


def describe_len(s):
    d = s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(1)
    return {k: float(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default="out/step1")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.input)
    print(f"[i] Sutunlar: {list(df.columns)}  |  Satir: {len(df):,}")
    assert {"ottoman", "turkish"} <= set(df.columns), "Beklenen sutunlar: ottoman, turkish"
    df = df[["ottoman", "turkish"]]

    raw_path = os.path.join(args.out, "raw.parquet")
    if not args.input:
        df.to_parquet(raw_path, index=False)
        print(f"[i] Ham veri kaydedildi: {raw_path}")

    N = len(df)
    o = df["ottoman"].fillna("").astype(str)
    t = df["turkish"].fillna("").astype(str)
    st = {"total_rows": N}

    # --- Bos / eksik
    st["null_ottoman"] = int(df["ottoman"].isna().sum())
    st["null_turkish"] = int(df["turkish"].isna().sum())
    st["empty_ottoman"] = int((o.str.strip() == "").sum())
    st["empty_turkish"] = int((t.str.strip() == "").sum())

    # --- Tekrarlar
    pair = o.str.strip() + "\t" + t.str.strip()
    st["dup_exact_pairs"] = int(pair.duplicated().sum())
    st["dup_exact_pairs_pct"] = pct(st["dup_exact_pairs"], N)
    st["dup_ottoman_only"] = int(o.str.strip().duplicated().sum())
    st["dup_turkish_only"] = int(t.str.strip().duplicated().sum())
    # ayni turkce cumle, farkli osmanlica yazim -> imla tutarsizligi gostergesi
    g = pd.DataFrame({"o": o.str.strip(), "t": t.str.strip()}).groupby("t")["o"].nunique()
    st["turkish_with_multiple_ottoman_spellings"] = int((g > 1).sum())

    # --- Uzunluklar (karakter)
    lo, lt = o.str.len(), t.str.len()
    st["len_ottoman_chars"] = describe_len(lo)
    st["len_turkish_chars"] = describe_len(lt)
    st["len_ottoman_utf8_bytes"] = describe_len(o.map(lambda x: len(x.encode("utf-8"))))
    st["len_turkish_utf8_bytes"] = describe_len(t.map(lambda x: len(x.encode("utf-8"))))
    ratio = (lt / lo.where(lo > 0)).dropna()
    st["len_ratio_tr_over_ot"] = describe_len(ratio)
    st["ratio_outliers_lt0.5_or_gt3"] = int(((ratio < 0.5) | (ratio > 3)).sum())

    # --- Script karisikligi
    st["ottoman_rows_with_latin"] = int(o.str.contains(RE_LATIN).sum())
    st["ottoman_rows_without_arabic"] = int((~o.str.contains(RE_ARABIC)).sum())
    st["turkish_rows_with_arabic"] = int(t.str.contains(RE_ARABIC).sum())
    st["ottoman_rows_with_digits"] = int(o.str.contains(RE_DIGIT).sum())

    # --- Normalizasyon ihtiyaci (Arapca vs Farsca harf varyantlari vb.)
    joined = "".join(o.sample(min(N, 200_000), random_state=SEED))
    st["char_counts_sample200k"] = {
        "arabic_yeh_U+064A (ي)": joined.count("ي"),
        "farsi_yeh_U+06CC (ی)": joined.count("ی"),
        "alef_maksura_U+0649 (ى)": joined.count("ى"),
        "arabic_kaf_U+0643 (ك)": joined.count("ك"),
        "keheh_U+06A9 (ک)": joined.count("ک"),
        "tatweel_U+0640": joined.count(TATWEEL),
        "zero_width_total": sum(joined.count(z) for z in ZERO_WIDTH),
        "harakat_total": len(HARAKAT.findall(joined)),
    }
    st["ottoman_not_nfc"] = int((o != o.map(lambda x: unicodedata.normalize("NFC", x))).sum())
    st["turkish_not_nfc"] = int((t != t.map(lambda x: unicodedata.normalize("NFC", x))).sum())

    # --- Karakter envanteri (en sik 80 karakter, osmanlica tarafi)
    vc = pd.Series(list(joined)).value_counts()
    st["ottoman_unique_chars_sample200k"] = int(len(vc))
    st["ottoman_top_chars"] = {f"{c!r} U+{ord(c):04X} {unicodedata.name(c, '?')}": int(n)
                               for c, n in vc.head(80).items()}

    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

    # --- Elle inceleme ornegi
    sm = df.sample(min(args.sample, N), random_state=SEED).reset_index().rename(columns={"index": "row_id"})
    sm["modern_mi (1/0)"] = ""
    sm["imla_sorunu (1/0)"] = ""
    sm["hizalama_bozuk (1/0)"] = ""
    sm["not"] = ""
    sm.to_csv(os.path.join(args.out, "sample_300.csv"), index=False, encoding="utf-8-sig")

    # --- Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        clip = int(max(lo.quantile(0.99), lt.quantile(0.99)))
        ax[0].hist(lo.clip(upper=clip), bins=60, color="#4C72B0")
        ax[0].set_title("Osmanlica (Arap harfli) - karakter uzunlugu")
        ax[1].hist(lt.clip(upper=clip), bins=60, color="#DD8452")
        ax[1].set_title("Turkce (Latin) - karakter uzunlugu")
        for a in ax:
            a.set_xlabel(f"karakter (p99={clip} ile kirpildi)")
            a.set_ylabel("satir sayisi")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "length_hist.png"), dpi=150)
    except Exception as e:
        print(f"[!] Histogram cizilemedi: {e}")

    # --- Ozet
    print("\n========== OZET ==========")
    for k in ["total_rows", "empty_ottoman", "empty_turkish", "dup_exact_pairs", "dup_exact_pairs_pct",
              "dup_ottoman_only", "dup_turkish_only", "turkish_with_multiple_ottoman_spellings",
              "ratio_outliers_lt0.5_or_gt3", "ottoman_rows_with_latin", "ottoman_rows_without_arabic",
              "turkish_rows_with_arabic", "ottoman_rows_with_digits", "ottoman_not_nfc", "turkish_not_nfc"]:
        print(f"{k:42s} {st[k]}")
    print("len_ottoman_chars  ", st["len_ottoman_chars"])
    print("len_turkish_chars  ", st["len_turkish_chars"])
    print("len_ottoman_bytes  ", st["len_ottoman_utf8_bytes"])
    print("char_counts        ", json.dumps(st["char_counts_sample200k"], ensure_ascii=False))
    print(f"\n[i] Ciktilar: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
