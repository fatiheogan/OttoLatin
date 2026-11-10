"""
OttoLatin - Step 2: normalization + filtering + train/dev/test split
Usage:
    python scripts/step2_clean_split.py
    (default input: out/step1/raw.parquet)

Outputs:
    data/processed/train.jsonl, dev.jsonl, test.jsonl, train_small.jsonl
        one row: {"id": ..., "ottoman": ..., "turkish": ...}
    out/step2/clean_report.json    -> rows dropped per rule, split sizes, leakage check
    out/step2/dropped_examples.txt -> examples dropped by each rule (for inspection)

Rules (Ottoman side):
    - Unicode NFC
    - Arabic yeh (ي U+064A) and alef maksura (ى U+0649) -> Farsi yeh (ی U+06CC)
    - keheh (ک U+06A9) -> Arabic kaf (ك U+0643)   [the data already uses ك]
    - ae (ە U+06D5) -> he (ه U+0647)
    - diacritics (harakat), tatweel and invisible characters (incl. ZWNJ) are removed
    - whitespace collapsed to single spaces
    Turkish side: NFC + whitespace only. Casing is KEPT.

Filters (in order):
    too_long      : either side > MAX_BYTES (UTF-8 bytes; ByT5 limit)
    few_letters   : share of Arabic letters on the Ottoman side < 0.3 (almost only digits/punctuation)
    digit_mismatch: numbers on the Ottoman side != numbers on the Turkish side (misaligned pair)
    len_ratio     : Turkish/Ottoman character ratio < 0.5 or > 3
    duplicate     : identical (ottoman, turkish) pair after normalization

Split: by hash of the normalized Ottoman text (deterministic).
    All Turkish variants of the same Ottoman text (e.g. "Ates" / "ates") land in the same split.
    Because rows are concatenations from a shared sentence pool, additionally: train rows containing
    any test/dev sentence (>= MIN_SENT chars) are dropped -> no sentence-level leakage.
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
MIN_SENT = 15         # sentence fragments shorter than this are not counted as leakage

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
    print(f"[i] Input: {len(df):,} rows")

    # --- Normalization (count how many rows change)
    before = df["ottoman"].copy()
    df["ottoman"] = df["ottoman"].map(norm_ot)
    df["turkish"] = df["turkish"].map(norm_tr)
    rep["ottoman_rows_changed_by_norm"] = int((before != df["ottoman"]).sum())
    del before
    print(f"[i] {rep['ottoman_rows_changed_by_norm']:,} Ottoman rows changed by normalization")

    dropped = {}
    examples = defaultdict(list)

    def apply(name, mask):
        nonlocal df
        bad = df[mask]
        dropped[name] = int(len(bad))
        examples[name] = bad.head(15)[["ottoman", "turkish"]].values.tolist()
        df = df[~mask]
        print(f"    - {name:15s} {len(bad):>9,} dropped -> remaining {len(df):,}")

    print("[i] Filters:")
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

    # --- Sentence-level decontamination: drop train rows containing test/dev sentences,
    #     drop dev rows containing test sentences.
    def sents(s):
        return [c for c in RE_SENT.split(s.lower()) if len(c) >= MIN_SENT]

    def collect(frame):
        out = set()
        for s in frame["turkish"]:
            out.update(sents(s))
        return out

    print("[i] Sentence-level decontamination (removing test/dev sentences from train)...")
    test_s = collect(parts["test"])
    m_dev = parts["dev"]["turkish"].map(lambda s: any(c in test_s for c in sents(s))).astype(bool)
    parts["dev"] = parts["dev"].loc[~m_dev]
    held_s = test_s | collect(parts["dev"])
    m_tr = parts["train"]["turkish"].map(lambda s: any(c in held_s for c in sents(s))).astype(bool)
    parts["train"] = parts["train"].loc[~m_tr]
    rep["decontam_removed"] = {"dev": int(m_dev.sum()), "train": int(m_tr.sum())}
    print(f"    removed {int(m_dev.sum()):,} rows from dev, {int(m_tr.sum()):,} rows from train")

    parts["train_small"] = parts["train"].sample(min(SMALL_N, len(parts["train"])), random_state=SEED)
    rep["splits"] = {k: len(v) for k, v in parts.items()}

    # sanity: the same Ottoman text must not appear in more than one split
    rep["exact_ottoman_overlap_train_test"] = int(parts["test"]["ottoman"].isin(set(parts["train"]["ottoman"])).sum())

    for k, v in parts.items():
        path = os.path.join(args.data_out, f"{k}.jsonl")
        v.to_json(path, orient="records", lines=True, force_ascii=False)
        print(f"[i] {k:12s} {len(v):>9,} rows -> {path}")

    # --- Sentence-level leakage check (rows are concatenated sentences)
    if not args.no_overlap_check:
        print("[i] Sentence-level leakage check (may take a few minutes)...")
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
        print(f"    {rep['test_sentences_seen_in_train_pct']}% of test sentences also appear in train")

    with open(os.path.join(args.report_out, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.report_out, "dropped_examples.txt"), "w", encoding="utf-8") as f:
        for k, rows in examples.items():
            f.write(f"===== {k} ({dropped[k]:,}) =====\n")
            for o, t in rows:
                f.write(f"OT: {o}\nTR: {t}\n\n")
    print(f"\n[i] Report: {os.path.join(args.report_out, 'clean_report.json')}")


if __name__ == "__main__":
    main()
