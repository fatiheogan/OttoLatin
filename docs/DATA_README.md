# OttoLatin – Data & Evaluation (Team 11)

Prepared by Fatih Emre Oğan (data + evaluation) · November 2026

## Files

| File | Rows | Purpose |
|---|---|---|
| `data/processed/train.jsonl` | 2,417,982 | Full training set (not in the repo – regenerate, see below) |
| `data/processed/train_small.jsonl` | 300,000 | Quick baselines / mid-project training (random subset of train, seed 42) |
| `data/processed/dev.jsonl` | 25,357 | Model selection, early stopping |
| `data/processed/test.jsonl` | 26,534 | Internal test (same distribution as training) |
| `data/external/riqa.jsonl` | 1,575 | External test – real handwritten (Riqa) documents |
| `data/external/naskh.jsonl` | 376 | External test – real printed (Naskh) book |
| `data/external/ot2tr.jsonl` | 2,358 | External test – 100 pages of archival documents, firmans, novels |

**Format (jsonl, UTF-8):**
- processed: `{"id", "ottoman", "turkish"}`
- external: `{"id", "source", "ottoman", "turkish", "turkish_norm", "ottoman_bytes"}`

## Sources

- **Training:** [Kotek37/ottoman-turkish-4m](https://huggingface.co/datasets/Kotek37/ottoman-turkish-4m), 2,717,146 rows, single split. Licence and creation method are not stated on the dataset page.
- **External test:** [osmanlica.com/test](https://osmanlica.com/test) (Mina ARGE). No licence stated; "research and academic use"; asks to cite the dataset version and date. → keep the repo private.
- **Not used:** Zenodo Osmanlica-Bench-v1. The page claims 6,500 samples, but the archive (3.6 kB) only contains a sample CSV with a handful of rows.

## What you need to know about the training data

1. **It is synthetic.** The Turkish side is modern Turkish (Wikipedia- and subtitle-like: "Orhan Pamuk Masumiyet Müzesi", "Prostat kanseri"). The Ottoman side was very likely produced automatically: the same Turkish sentence has a different Ottoman spelling only 134 times in 2.7M rows. Tool errors are visible ("kes" → "كثع", "km" → "قم").
2. **Rows are concatenations from a shared sentence pool.** With a random split, **77.5%** of test sentences also appeared in train. Train rows containing any test/dev sentence were removed → leakage **0%**.
3. **Casing cannot be inferred from the input.** The same Ottoman text maps to "Ateş" and "ateş" (8,190 cases). Report both raw and normalized scores.
4. **49% of rows contain digits.** Written as Arabic-Indic digits (٣٨) and matching the Turkish side 1:1 (38) – real content, not an artifact.
5. **Label noise.** At least ~0.7% of `test_2k` rows have an Ottoman infinitive but a Turkish imperative (طولدیرمق → "Doldur.", خاطرلامق → "Hatırla.") – another trace of the generation tool.

## Processing (`scripts/step2_clean_split.py`)

**Ottoman normalization:** NFC; ي, ى → ی; ک → ك; ە → ه; harakat, tatweel and invisible characters (incl. ZWNJ) removed; whitespace collapsed. Turkish side: NFC + whitespace only, **casing kept**.

| Filter | Dropped |
|---|---|
| > 512 UTF-8 bytes (ByT5 limit) | 677 |
| Almost only digits ("٣١ ٣٨٥.") | 67,148 |
| Numbers differ between the two sides (misalignment) | 65 |
| Length ratio < 0.5 or > 3 | 1 |
| Duplicates | 35 |
| Sentence-level decontamination (train / dev) | 178,413 / 934 |

**Split:** by MD5 hash of the normalized Ottoman text, 98% / 1% / 1%. Deterministic – rerunning the script reproduces the same split.

**Length:** Ottoman median 48 characters (87 bytes), p99 375 bytes, max 512 bytes.

## External test sets (`scripts/step3b_build_external.py`)

- The Ottoman side gets the **same** normalization as training, plus: presentation forms (ﺎ → ا) mapped to base letters, ۔ → ".", punctuation never seen in training (؟ ( ) « » " …) removed. No unseen characters remain.
- Ot2Tr: 3 pages with mismatched line counts were dropped (100 of 103 pages used). 22 rows exceed 512 bytes; skip them using `ottoman_bytes` if needed.
- **The Turkish references use scholarly transliteration** ("Huzûr-ı Sâmî", "şer‘an"). `turkish_norm` comes from `scripts/text_norm.py`: Turkish-aware lowercasing, â/î/û and ayn/hamza marks removed, hyphen → space, punctuation removed.
- **Reference point** (osmanlica.com's own system, from their `metrics.json`): Riqa transliteration CER 0.113, norm-CER 0.012; Naskh CER 0.113, norm-CER 0.043. Their normalization may differ from ours – treat as a rough reference only.

## Evaluation protocol

**Sets** (`scripts/step4_make_eval_sets.py` → `data/eval/`): `test_2k` (2,000 rows from the internal test, seed 42), `riqa` (1,575), `naskh` (376), `ot2tr` (2,358). **Every model is scored on these files.** `fewshot.jsonl` holds the 8 LLM demonstration pairs (from train).

**Prediction format:** `preds/<model_name>/<set>.jsonl`, one line `{"id": "...", "prediction": "..."}` per example. ids must match `data/eval`. A missing prediction counts as an empty string and is penalized.

**Metrics** (`scripts/evaluate.py`, corpus level, percent):
- **Main metric: normalized CER** – character error rate after passing prediction and reference through `text_norm.turkish_norm()`.
- Also raw CER, raw and normalized WER, chrF (sacrebleu).
- 95% bootstrap confidence interval for every score (1,000 resamples).

**What counts as a real difference:** normalized CER gap **≥ 1 point** and **paired bootstrap p < 0.05**. `--compare` reports this automatically.

**Expected gap on external sets:** CER will be higher than on the internal test. `text_norm` removes diacritics but not archaic forms: references write "etdirilmeyüp", "içün" while models trained on modern Turkish produce "ettirilmeyip", "için". This is part of the domain shift we want to measure.

**Baselines:**
| | System | Description |
|---|---|---|
| B0 | `scripts/baseline_rule.py` | Letter mapping + simple vowel rules, no learning (lower bound) |
| B1 | `docs/baseline_llm_fewshot_colab.ipynb` | Qwen2.5-1.5B-Instruct, 8-shot, no training |
| B2 | osmanlica.com `metrics.json` | Published reference scores (Riqa, Naskh) |

**B0 results** (normalized CER %, 95% CI):
| test_2k | Riqa | Naskh | Ot2Tr |
|---|---|---|---|
| 28.82 [28.52–29.13] | 33.81 [33.59–34.01] | 34.85 [34.19–35.49] | 31.92 [31.66–32.26] |

**Error analysis** (`scripts/error_analysis.py`): for B0, **80–90% of character errors are vowel errors** (vowels are unwritten or ambiguous in Ottoman script); ambiguous consonants (ك → k/g/ğ/n, ط → t/d) account for 3–7%; casing/diacritics explain 5–15% of raw CER.

**Commands:**
```bash
pip install rapidfuzz sacrebleu
python scripts/step4_make_eval_sets.py
python scripts/baseline_rule.py
python scripts/evaluate.py --model_dir preds/rule
python scripts/evaluate.py --model_dir preds/qwen_fewshot --compare preds/rule
python scripts/error_analysis.py --model_dir preds/rule preds/qwen_fewshot
```
Each model folder gets `results.json`, `error_analysis.json/.png` and `worst_<set>.txt` (30 worst examples per set).

## Reproducing the data

```bash
pip install datasets pandas pyarrow matplotlib
python scripts/step1_explore.py --out out/step1            # download + statistics
python scripts/step2_clean_split.py                         # normalize, filter, split
python scripts/step3a_peek_external.py                      # unpack data/external/raw/*.zip
python scripts/step3b_build_external.py                     # convert external sets
```
