"""
show_results.py - scores the demo run (possible only because the FAKE test set keeps its answers)
and prints a few matches side by side so you can see what the pipeline did.

Usage:  python demo/show_results.py --data demo_data --out demo_output
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import metrics  # noqa: E402


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def pairs(df, col):
    return {(a, x) for a, b in zip(df.source1_entity_id, df[col]) for x in b.split(",") if x}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--out", default="demo_output")
    a = ap.parse_args()
    pred = read(os.path.join(a.out, "matching_results.tsv"))
    cand = read(os.path.join(a.out, "candidate_pairs.tsv"))
    truth = read(os.path.join(a.data, "test_answers_DEMO_ONLY.tsv"))
    P, T, C = pairs(pred, "matched_entity_ids"), pairs(truth, "matched_entity_ids"), pairs(cand, "candidate_entity_ids")
    ids = list(truth.source1_entity_id)
    print("=" * 70)
    print(f"Macro F0.5 on the demo test set : {metrics.macro_f05(ids, P, T):.4f}")
    print(f"Pair precision / recall        : {len(P & T) / max(len(P), 1):.3f} / {len(P & T) / max(len(T), 1):.3f}")
    print(f"Candidates per business        : {len(C) / len(ids):.2f}   (recall ceiling {len(C & T) / max(len(T), 1):.3f})")
    print("=" * 70)
    recs = pd.concat([read(os.path.join(a.data, "test", f"test_source{s}.tsv")) for s in (1, 2, 3)])
    recs = recs.set_index("entity_id")
    shown = 0
    for s1, m in zip(pred.source1_entity_id, pred.matched_entity_ids):
        if not m or shown >= 5:
            continue
        r = recs.loc[s1]
        print(f"\n{s1} | {r.business_name} | {r.business_address} | {r.country}")
        for x in m.split(","):
            q = recs.loc[x]
            ok = "correct" if (s1, x) in T else "WRONG"
            print(f"   -> {x} | {q.business_name} | {q.business_address}   [{ok}]")
        shown += 1
