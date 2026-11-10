# OttoLatin – Ottoman Turkish → Latin Transliteration

CSE 484 / CSE 654 Natural Language Processing · GTÜ · Fall 2026 · **Team 11**

**Research question:** Does a byte-level seq2seq model (ByT5) transliterate Ottoman Turkish (Arabic script) into Modern Turkish (Latin script) better than LLM-based approaches (Qwen2)?

| Member | Role |
|---|---|
| Sena Gizem Temuçin | Research lead, experimental design & ablations |
| Fatih Emre Oğan | Data + evaluation pipeline |
| Enes | Evaluation |
| Sedef | Model training |

## Repo structure

```
scripts/        data preparation, evaluation, baselines
  step1_explore.py .. step3b_build_external.py   data pipeline (see docs/DATA_README.md)
  step4_make_eval_sets.py                        fixed evaluation sets
  evaluate.py                                    CER/WER/chrF + bootstrap CI + paired test
  text_norm.py                                   shared normalization (use it everywhere)
  baseline_rule.py                               B0: rule-based baseline
data/eval/      FIXED evaluation sets – every model is scored on these
data/external/  real Ottoman test documents (osmanlica.com) – research use only
docs/           DATA_README.md (data + eval protocol), Colab notebooks
```

Training data (`data/processed/`, ~500 MB) is **not** in the repo. Recreate it (deterministic split):
```bash
pip install datasets pandas pyarrow matplotlib
python scripts/step1_explore.py --out out/step1
python scripts/step2_clean_split.py
```

## Submitting model outputs

Write predictions to `preds/<model_name>/<set>.jsonl` for each set in `data/eval/` (`test_2k`, `riqa`, `naskh`, `ot2tr`), one line per example:
```json
{"id": "...", "prediction": "..."}
```
Then:
```bash
pip install rapidfuzz sacrebleu
python scripts/evaluate.py --model_dir preds/<model_name> --compare preds/rule
```
Main metric: **normalized CER** (`text_norm.turkish_norm`). A difference counts as real if ≥ 1 point and paired bootstrap p < 0.05.

## Results so far

| System | test_2k | Riqa | Naskh | Ot2Tr |
|---|---|---|---|---|
| B0 rule-based | 28.82 | 33.81 | 34.85 | 31.92 |
| B1 Qwen2.5-1.5B few-shot | – | – | – | – |
| ByT5 (fine-tuned) | – | – | – | – |
| Qwen2 (fine-tuned) | – | – | – | – |

*norm-CER %, lower is better.* Reference (osmanlica.com own system, their normalization): Riqa 1.19, Naskh 4.28.

> ⚠️ This repo is private: the external test sets have no stated license (research/academic use only).
