"""
OttoLatin - seeded word-level error sample (for the manual-style error table)

Usage:
  python scripts/sample_word_errors.py --model_dir preds/rule
  python scripts/sample_word_errors.py --model_dir preds/byt5 --per_set 15 --seed 42

What it does:
  For each evaluation set, compares normalized reference and prediction word by word.
  - If both have the same number of words, every differing word is one candidate error.
  - If the word counts differ, the whole line is one candidate ("word boundary (word count differs)").
  It then shuffles the candidates with a fixed seed, keeps --per_set of them per set,
  and groups them with fixed rules:
    vowel only                      : consonant letters identical, only vowels differ
    ambiguous consonant (+vowel)    : same consonant skeleton once k/g/ğ/n, t/d, p/b, c/ç, s/z are merged
    و/ی/ه read as consonant vs vowel : identical after also ignoring v, y, h
    و/ی/ه + ambiguous consonant     : both of the above together
    other / different word          : anything else
Output: category counts with one example each.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_norm import turkish_norm  # noqa: E402

SETS = ["test_2k", "riqa", "naskh", "ot2tr"]
VOWELS = set("aeıioöuü")
AMBIGUOUS = {"k": "K", "g": "K", "ğ": "K", "n": "K", "t": "T", "d": "T", "p": "P", "b": "P",
             "c": "C", "ç": "C", "s": "S", "z": "S", "h": "H"}


def consonants(w):
    return "".join(c for c in w if c not in VOWELS)


def skeleton(w):
    return "".join(AMBIGUOUS.get(c, c) for c in w if c not in VOWELS)


def drop(s, chars):
    return s.translate(str.maketrans("", "", chars))


def category(ref, pred):
    if pred == "":
        return "missing word"
    if consonants(ref) == consonants(pred):
        return "vowel only"
    if skeleton(ref) == skeleton(pred):
        return "ambiguous consonant (+vowel)"
    if drop(consonants(ref), "vyh") == drop(consonants(pred), "vyh"):
        return "و/ی/ه read as consonant vs vowel"
    if drop(skeleton(ref), "vyH") == drop(skeleton(pred), "vyH"):
        return "و/ی/ه + ambiguous consonant"
    return "other / different word"


def load(path):
    with open(path, encoding="utf-8") as f:
        return {str(r["id"]): r for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--eval_dir", default="data/eval")
    ap.add_argument("--per_set", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    sample = []
    for s in SETS:
        refs = load(os.path.join(args.eval_dir, f"{s}.jsonl"))
        preds = load(os.path.join(args.model_dir, f"{s}.jsonl"))
        cands = []
        for i, r in refs.items():
            rw = turkish_norm(r["turkish"]).split()
            pw = turkish_norm(preds[i]["prediction"]).split() if i in preds else []
            if len(rw) != len(pw):
                cands.append((s, "word boundary (word count differs)", " ".join(rw), " ".join(pw)))
                continue
            for a, b in zip(rw, pw):
                if a != b:
                    cands.append((s, None, a, b))
        random.shuffle(cands)
        sample.extend(cands[:args.per_set])

    counts, examples = collections.Counter(), {}
    for s, cat, a, b in sample:
        cat = cat or category(a, b)
        counts[cat] += 1
        examples.setdefault(cat, (s, a, b))

    print(f"{args.model_dir}: {len(sample)} sampled errors ({args.per_set} per set, seed {args.seed})")
    for cat, n in counts.most_common():
        s, a, b = examples[cat]
        print(f"{n:4d}  {cat:36s} e.g. [{s}] \"{a[:60]}\" -> \"{b[:60]}\"")


if __name__ == "__main__":
    main()
