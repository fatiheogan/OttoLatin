"""
OttoLatin - Step 4: freeze the evaluation sets
Usage : python scripts/step4_make_eval_sets.py
Output (data/eval/):
    test_2k.jsonl  -> 2,000 rows from the internal test set, fixed seed (affordable for LLMs)
    riqa.jsonl, naskh.jsonl, ot2tr.jsonl -> external sets
    fewshot.jsonl  -> LLM few-shot examples (from train, disjoint from test)
One row: {"id", "ottoman", "turkish"}
ALL models (baselines + trained models) are evaluated on these files.
"""
import json
import os
import random

SEED = 42
N_TEST = 2000
N_FEWSHOT = 8
OUT = "data/eval"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": str(r["id"]), "ottoman": r["ottoman"], "turkish": r["turkish"]},
                               ensure_ascii=False) + "\n")
    print(f"[i] {os.path.basename(path):14s} {len(rows):>6,} rows")


os.makedirs(OUT, exist_ok=True)
rng = random.Random(SEED)

test = load("data/processed/test.jsonl")
dump(rng.sample(test, min(N_TEST, len(test))), os.path.join(OUT, "test_2k.jsonl"))

for name in ["riqa", "naskh", "ot2tr"]:
    dump(load(f"data/external/{name}.jsonl"), os.path.join(OUT, f"{name}.jsonl"))

# few-shot: medium-length, varied examples from train_small
small = load("data/processed/train_small.jsonl")
cand = [r for r in small if 40 <= len(r["turkish"]) <= 120]
dump(random.Random(SEED + 1).sample(cand, N_FEWSHOT), os.path.join(OUT, "fewshot.jsonl"))
