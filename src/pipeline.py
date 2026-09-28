"""
pipeline.py - the whole solution, run as separate stages from the command line.

    python src/pipeline.py prep     # read + clean the 7 TSV files            -> cache/*.parquet
    python src/pipeline.py block    # blocking for train and test              -> cache/*_cand.npz
    python src/pipeline.py train    # features + LightGBM + threshold tuning   -> model/
    python src/pipeline.py predict  # score test pairs, write the 2 TSV files  -> output/
    python src/pipeline.py all      # everything above, in order

Every stage saves its result to disk, so if Colab disconnects you restart from the last stage.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking  # noqa: E402
import features  # noqa: E402
import metrics  # noqa: E402
import normalize  # noqa: E402

CFG = dict(
    # blocking
    max_name_tokens=6, max_addr_tokens=8, max_num_tokens=3, max_prefix_tokens=3, max_addr_prefix_tokens=6,
    max_s1_per_key=15, max_s23_per_key=150, top_k=30, min_score=0.0, s1_chunk=50_000, s23_chunk=500_000,
    # training
    n_train_s1=300_000, n_val_s1=100_000, n_prune_s1=150_000, feature_chunk=1_500_000, seed=42,
    # stage-1 pruning (learned blocking): keep pairs until this share of true pairs is kept
    prune_keep_recall=0.995, max_cand_per_s1=10,
    lgb1_params=dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=0,
                     verbose=-1, seed=42),
    lgb1_rounds=300,
    lgb_params=dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    num_threads=0, verbose=-1, seed=42),
    lgb_rounds=1500, lgb_early_stop=50,
    # predict: also write matching_results_t<thr>.tsv for these stricter thresholds (comma-separated)
    extra_thresholds="0.80,0.85,0.90,0.95",
)

T0 = time.time()
LOG_FILE = None


def log(*msg):
    line = f"[{time.time() - T0:7.0f}s] " + " ".join(str(m) for m in msg)
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ----------------------------------------------------------------------------------- stage: prep
def stage_prep(args):
    for split in ("train", "test"):
        d = os.path.join(args.data, split)
        if not os.path.isdir(d):
            log(f"skip {split}: {d} not found"); continue
        t1 = normalize.prepare(os.path.join(d, f"{split}_source1.tsv"))
        t2 = normalize.prepare(os.path.join(d, f"{split}_source2.tsv"))
        t3 = normalize.prepare(os.path.join(d, f"{split}_source3.tsv"))
        t23 = pa.concat_tables([t2, t3]).combine_chunks()
        t23 = t23.append_column("is_s3", pc.starts_with(t23.column("entity_id"), "S3"))
        pq.write_table(t1, os.path.join(args.work, f"{split}_s1.parquet"))
        pq.write_table(t23, os.path.join(args.work, f"{split}_s23.parquet"))
        cnt = pd.Series(t1.column("country").to_pylist()).value_counts().to_dict()
        log(f"prep {split}: S1={t1.num_rows:,} S2+S3={t23.num_rows:,} countries(S1)={cnt}")
        nl = pc.sum(t23.column("non_latin")).as_py()
        log(f"   non-Latin names in S2+S3: {nl:,}  (transliterated)")
        del t1, t2, t3, t23


NEEDED = ["entity_id", "country", "nm", "nm_ns", "non_latin", "ad", "ad_alpha", "nums", "hn"]


def load(args, split, extra=()):
    s1 = pq.read_table(os.path.join(args.work, f"{split}_s1.parquet"), columns=NEEDED + list(extra))
    s23 = pq.read_table(os.path.join(args.work, f"{split}_s23.parquet"), columns=NEEDED + ["is_s3"] + list(extra))
    s1 = features.add_record_columns(s1.combine_chunks())
    s23 = features.add_record_columns(s23.combine_chunks())
    return features.add_frequency_columns(s1, s23)


# ----------------------------------------------------------------------------------- stage: block
def stage_block(args, splits=("train", "test")):
    for split in splits:
        if not os.path.exists(os.path.join(args.work, f"{split}_s1.parquet")):
            continue
        s1, s23 = load(args, split)
        t = time.time()
        cand = blocking.generate_candidates(s1, s23, CFG, log=log)
        np.savez(os.path.join(args.work, f"{split}_cand.npz"), **cand)
        log(f"block {split}: {len(cand['s1']):,} pairs, {len(cand['s1']) / s1.num_rows:.1f} per S1 record, "
            f"{time.time() - t:.0f}s")


def load_cand(args, split):
    z = np.load(os.path.join(args.work, f"{split}_cand.npz"))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------- helpers
def truth_arrays(args, s1, s23):
    """Ground truth -> (pair codes int64, number of true matches per S1 row)."""
    gt = normalize.read_tsv(os.path.join(args.data, "train", "train_ground_truth.tsv"))
    lists = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists)
    ok = pc.greater(pc.utf8_length(flat), 0)
    flat, parent = pc.filter(flat, ok), pc.filter(parent, ok)
    a = pc.index_in(pc.take(gt.column("source1_entity_id"), parent), value_set=s1.column("entity_id"))
    b = pc.index_in(flat, value_set=s23.column("entity_id"))
    good = pc.and_(pc.is_valid(a), pc.is_valid(b))
    a = pc.filter(a, good).to_numpy().astype(np.int64)
    b = pc.filter(b, good).to_numpy().astype(np.int64)
    bad = len(good) - len(a)
    if bad:
        log(f"   warning: {bad} ground-truth ids not found in the source files (ignored)")
    n_true = np.bincount(a, minlength=s1.num_rows)
    ub, cb = np.unique(b, return_counts=True)
    log(f"   ground truth: {len(a):,} true pairs; S2/S3 records matched to >1 S1 entity: {(cb > 1).sum():,}")
    return a * np.int64(s23.num_rows) + b, n_true


def featurize(fn, s1, s23, cand, ctx, idx):
    parts = []
    for a in range(0, len(idx), CFG["feature_chunk"]):
        parts.append(fn(s1, s23, cand, ctx, idx[a:a + CFG["feature_chunk"]]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def predict_all_stage1(model, cols, s1, s23, cand, ctx):
    """Stage-1 probability for EVERY blocking pair (chunked to save memory)."""
    n = len(cand["s1"])
    p1 = np.zeros(n, np.float32)
    t = time.time()
    step = CFG["feature_chunk"]
    for a in range(0, n, step):
        idx = np.arange(a, min(a + step, n))
        p1[idx] = model.predict(features.compute_cheap_features(s1, s23, cand, ctx, idx)[cols])
        if (a // step) % 10 == 0:
            log(f"   stage-1 scored {min(a + step, n):,}/{n:,} pairs ({time.time() - t:.0f}s)")
    return p1


def prune_mask(p1, ctx1, t1):
    """Learned blocking: keep a pair if its stage-1 probability is high enough and it is among the
    best `max_cand_per_s1` partners of its S1 record."""
    return (p1 >= t1) & (ctx1["p1_rank_s1"] < CFG["max_cand_per_s1"])


def select(s1_idx, prob, s23_idx, thr, rel=0.0):
    """Turn probabilities into final matches.
    1) one-to-one: an S2/S3 record belongs to at most ONE S1 entity (S1 is deduplicated), so it is
       kept only for the S1 entity where its probability is highest;
    2) keep it only if probability >= thr (and >= rel * best probability of that S1 entity)."""
    keep = features.group_rank(s23_idx, prob) == 0
    keep &= prob >= thr
    if rel > 0:
        best = features.group_stat(s1_idx, prob, np.maximum)
        keep &= prob >= rel * best
    return keep


# ----------------------------------------------------------------------------------- stage: train
def lgb_fit(params, X, y, Xes, yes, rounds, early):
    import lightgbm as lgb
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    t = time.time()
    m = lgb.train(params, dtr, rounds, valid_sets=[des],
                  callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(100)])
    log(f"   LightGBM: {m.best_iteration} trees, {time.time() - t:.0f}s")
    return m


def stage_train(args):
    s1, s23 = load(args, "train")
    cand = load_cand(args, "train")
    n1, n23 = s1.num_rows, s23.num_rows
    true_codes, n_true = truth_arrays(args, s1, s23)
    code = cand["s1"].astype(np.int64) * np.int64(n23) + cand["s23"].astype(np.int64)
    label = np.isin(code, true_codes)
    del code
    ctx = features.pair_context(cand)
    tot = len(true_codes)
    ctry = np.array(s1.column("country").to_pylist())

    log(f"BLOCKING: {len(label):,} candidate pairs, {len(label) / n1:.1f} per S1, true pairs={tot:,}")
    for k in (5, 10, CFG["top_k"]):
        log(f"   recall@{k:<3d} = {label[ctx['brank_s1'] < k].sum() / tot:.4f}")

    # ---- split S1 entities: P = stage-1 training, 0 = fit, 1 = early stop, 2 = validation
    rng = np.random.default_rng(CFG["seed"])
    perm = rng.permutation(n1)
    npr = min(CFG["n_prune_s1"], int(n1 * 0.2))
    ntr = min(CFG["n_train_s1"], int(n1 * 0.5))
    nva = min(CFG["n_val_s1"], int(n1 * 0.2))
    grp = np.full(n1, -1, np.int8)
    grp[perm[:npr]] = 3
    grp[perm[npr:npr + int(ntr * 0.9)]] = 0
    grp[perm[npr + int(ntr * 0.9):npr + ntr]] = 1
    grp[perm[npr + ntr:npr + ntr + nva]] = 2
    g = grp[cand["s1"]]

    # ================= STAGE 1: learned pruning of the blocking candidates =================
    log("STAGE 1 (pruning model)")
    iP, iE = np.flatnonzero(g == 3), np.flatnonzero(g == 1)
    XP = featurize(features.compute_cheap_features, s1, s23, cand, ctx, iP)
    XE = featurize(features.compute_cheap_features, s1, s23, cand, ctx, iE)
    cols1 = list(XP.columns)
    m1 = lgb_fit(CFG["lgb1_params"], XP, label[iP], XE, label[iE], CFG["lgb1_rounds"], 30)
    del XP, XE
    p1 = predict_all_stage1(m1, cols1, s1, s23, cand, ctx)
    ctx1 = features.stage1_context(cand, p1)
    # choose the pruning threshold on the validation S1 entities
    iV = np.flatnonzero(g == 2)
    lv = label[iV]
    t1 = 0.0
    for t in np.r_[np.arange(0.001, 0.02, 0.001), np.arange(0.02, 0.2, 0.005), np.arange(0.2, 0.61, 0.02)]:
        kept = prune_mask(p1[iV], {"p1_rank_s1": ctx1["p1_rank_s1"][iV]}, t)
        if (kept & lv).sum() >= CFG["prune_keep_recall"] * lv.sum():
            t1 = float(t)
        else:
            break
    keep1 = prune_mask(p1, ctx1, t1)
    log(f"   pruning threshold t1={t1:.3f}: candidates {len(p1):,} -> {keep1.sum():,} "
        f"({keep1.sum() / n1:.2f} per S1, was {len(p1) / n1:.2f})")
    log(f"   recall ceiling after pruning = {(label & keep1).sum() / tot:.4f} (before {label.sum() / tot:.4f})")
    for c in np.unique(ctry):
        m = ctry[cand["s1"]] == c
        log(f"   [{c}] candidates/S1={keep1[m].sum() / (ctry == c).sum():.2f}  "
            f"recall ceiling={(label & keep1)[m].sum() / max(n_true[ctry == c].sum(), 1):.4f}")
    ctx_all = dict(ctx); ctx_all.update(ctx1)

    # ================= STAGE 2: the matching model on the pruned candidates =================
    log("STAGE 2 (matching model)")
    idx = {k: np.flatnonzero((g == k) & keep1) for k in (0, 1, 2)}
    X = {}
    for k in (0, 1, 2):
        t = time.time()
        X[k] = featurize(features.compute_features, s1, s23, cand, ctx_all, idx[k])
        log(f"   features group {k}: {len(idx[k]):,} pairs, {time.time() - t:.0f}s")
    cols = list(X[0].columns)
    model = lgb_fit(CFG["lgb_params"], X[0], label[idx[0]], X[1], label[idx[1]],
                    CFG["lgb_rounds"], CFG["lgb_early_stop"])
    del X[0], X[1]

    pv = model.predict(X[2], num_iteration=model.best_iteration).astype(np.float32)
    vs1, vs23, vlab = cand["s1"][idx[2]], cand["s23"][idx[2]], label[idx[2]]
    val_rows = np.flatnonzero(grp == 2)
    remap = np.full(n1, -1, np.int64); remap[val_rows] = np.arange(len(val_rows))
    vloc = remap[vs1]
    best = (0, 0.5, 0.0)
    for rel in (0.0, 0.3, 0.5):
        for thr in np.arange(0.20, 0.96, 0.025):
            keep = select(vs1, pv, vs23, thr, rel)
            f = metrics.macro_f05_arrays(len(val_rows), vloc, keep, vlab, n_true[val_rows])
            if f > best[0]:
                best = (f, float(thr), rel)
    f, thr, rel = best
    log(f"VALIDATION macro F0.5 = {f:.4f}  (threshold={thr:.3f}, relative={rel})  on {len(val_rows):,} S1 entities")
    keep = select(vs1, pv, vs23, thr, rel)
    for c in np.unique(ctry):
        mrow = ctry[val_rows] == c
        mp = ctry[vs1] == c
        rr = np.full(n1, -1, np.int64); rr[val_rows[mrow]] = np.arange(mrow.sum())
        fc = metrics.macro_f05_arrays(int(mrow.sum()), rr[vs1[mp]], keep[mp], vlab[mp], n_true[val_rows[mrow]])
        log(f"   F0.5 [{c}] = {fc:.4f}")
    sing = n_true[val_rows] == 0
    pred_n = np.bincount(vloc, weights=keep, minlength=len(val_rows))
    log(f"   singletons: {sing.mean():.3f} of val; predicted empty for {(pred_n[sing] == 0).mean():.3f} of them")
    log(f"   precision(pairs)={(keep & vlab).sum() / max(keep.sum(), 1):.4f}  recall(pairs)="
        f"{(keep & vlab).sum() / max(n_true[val_rows].sum(), 1):.4f}")
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    log("feature importance (gain, top 15):\n" + (imp.head(15) / imp.sum()).round(3).to_string())

    os.makedirs(args.model_dir, exist_ok=True)
    m1.save_model(os.path.join(args.model_dir, "lgb_stage1.txt"), num_iteration=m1.best_iteration)
    model.save_model(os.path.join(args.model_dir, "lgb_model.txt"), num_iteration=model.best_iteration)
    json.dump(dict(threshold=thr, relative=rel, val_f05=f, features=cols, features_stage1=cols1,
                   prune_threshold=t1, cfg=CFG),
              open(os.path.join(args.model_dir, "model_meta.json"), "w"), indent=1, default=str)
    log(f"saved models to {args.model_dir}")
    write_error_report(args, s1, s23, vs1, vs23, pv, keep, vlab, true_codes, val_rows)


def write_error_report(args, s1, s23, vs1, vs23, pv, keep, vlab, true_codes, val_rows, n=40):
    """Human-readable examples of mistakes, to understand what to improve next."""
    raw1 = pq.read_table(os.path.join(args.work, "train_s1.parquet"), columns=["raw_name", "raw_addr"])
    raw23 = pq.read_table(os.path.join(args.work, "train_s23.parquet"), columns=["entity_id", "raw_name", "raw_addr"])
    rng = np.random.default_rng(0)

    def fmt1(i):
        return f"{s1.column('entity_id')[i].as_py()} | {raw1.column('raw_name')[i].as_py()} | {raw1.column('raw_addr')[i].as_py()}"

    def fmt2(j):
        return f"{raw23.column('entity_id')[j].as_py()} | {raw23.column('raw_name')[j].as_py()} | {raw23.column('raw_addr')[j].as_py()}"

    lines = []
    fp = np.flatnonzero(keep & ~vlab)
    lines.append(f"===== FALSE POSITIVES (predicted match, actually different) : {len(fp):,} =====")
    for i in rng.choice(fp, min(n, len(fp)), replace=False):
        lines += [fmt1(vs1[i]), f"   -> {fmt2(vs23[i])}   p={pv[i]:.3f}", ""]
    fn = np.flatnonzero(~keep & vlab)
    lines.append(f"===== FALSE NEGATIVES in candidates (missed by the model) : {len(fn):,} =====")
    for i in rng.choice(fn, min(n, len(fn)), replace=False):
        lines += [fmt1(vs1[i]), f"   -> {fmt2(vs23[i])}   p={pv[i]:.3f}", ""]
    # true pairs that blocking never produced
    n23 = np.int64(s23.num_rows)
    tc_s1 = true_codes // n23
    inval = np.isin(tc_s1, val_rows)
    have = set((vs1[vlab].astype(np.int64) * n23 + vs23[vlab]).tolist())
    miss = [c for c in true_codes[inval][:200000] if c not in have]
    lines.append(f"===== TRUE PAIRS LOST BY BLOCKING (sample; val set) =====")
    for c in rng.choice(miss, min(n, len(miss)), replace=False) if miss else []:
        lines += [fmt1(int(c // n23)), f"   -> {fmt2(int(c % n23))}", ""]
    path = os.path.join(args.model_dir, "val_errors.txt")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    log(f"wrote error examples to {path}")


# ----------------------------------------------------------------------------------- stage: predict
def write_id_lists(path, header, s1_ids, s1_idx, other_ids_arr):
    """Write one row per S1 entity: id <TAB> comma-separated list (possibly empty)."""
    n1 = len(s1_ids)
    order = np.argsort(s1_idx, kind="stable")
    counts = np.bincount(s1_idx, minlength=n1)
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    if isinstance(other_ids_arr, pa.ChunkedArray):
        other_ids_arr = other_ids_arr.combine_chunks()
    vals = pc.take(other_ids_arr, pa.array(order)) if len(order) else pa.array([], pa.string())
    lists = pa.LargeListArray.from_arrays(pa.array(offsets, pa.int64()), vals)
    joined = pc.binary_join(lists, ",")
    if isinstance(s1_ids, pa.ChunkedArray):
        s1_ids = s1_ids.combine_chunks()
    tbl = pa.table({header[0]: s1_ids, header[1]: joined})
    with open(path, "wb") as fh:
        fh.write(("\t".join(header) + "\n").encode("utf-8"))
        pacsv.write_csv(tbl, fh, write_options=pacsv.WriteOptions(
            include_header=False, delimiter="\t", quoting_style="none"))


def stage_predict(args):
    import lightgbm as lgb
    meta = json.load(open(os.path.join(args.model_dir, "model_meta.json")))
    m1 = lgb.Booster(model_file=os.path.join(args.model_dir, "lgb_stage1.txt"))
    model = lgb.Booster(model_file=os.path.join(args.model_dir, "lgb_model.txt"))
    s1, s23 = load(args, "test")
    cand = load_cand(args, "test")
    ctx = features.pair_context(cand)
    log("STAGE 1 (pruning) on all test blocking pairs")
    p1 = predict_all_stage1(m1, meta["features_stage1"], s1, s23, cand, ctx)
    ctx.update(features.stage1_context(cand, p1))
    keep1 = prune_mask(p1, ctx, meta["prune_threshold"])
    log(f"   {len(p1):,} -> {keep1.sum():,} candidates ({keep1.sum() / s1.num_rows:.2f} per S1)")
    # keep only the pruned candidates from here on
    idx1 = np.flatnonzero(keep1)
    cand = {k: v[idx1] for k, v in cand.items()}
    ctx = {k: v[idx1] for k, v in ctx.items()}
    del p1, keep1, idx1
    n = len(cand["s1"])
    prob = np.zeros(n, np.float32)
    t = time.time()
    step = CFG["feature_chunk"]
    log("STAGE 2 (matching) on the pruned candidates")
    for a in range(0, n, step):
        idx = np.arange(a, min(a + step, n))
        prob[idx] = model.predict(features.compute_features(s1, s23, cand, ctx, idx)[meta["features"]])
        if (a // step) % 5 == 0:
            log(f"   scored {min(a + step, n):,}/{n:,} pairs ({time.time() - t:.0f}s)")
    keep = select(cand["s1"], prob, cand["s23"], meta["threshold"], meta["relative"])
    os.makedirs(args.out, exist_ok=True)
    ids1 = s1.column("entity_id")
    ids23 = s23.column("entity_id")
    write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"],
                   ids1, cand["s1"], pc.take(ids23, pa.array(cand["s23"])))
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
                   ids1, cand["s1"][keep], pc.take(ids23, pa.array(cand["s23"][keep])))
    np.savez(os.path.join(args.work, "test_scored.npz"), s1=cand["s1"], s23=cand["s23"], prob=prob)
    for t in [float(x) for x in str(CFG["extra_thresholds"]).split(",") if x.strip()]:
        k2 = select(cand["s1"], prob, cand["s23"], t, meta["relative"])
        write_id_lists(os.path.join(args.out, f"matching_results_t{t:.2f}.tsv"),
                       ["source1_entity_id", "matched_entity_ids"],
                       ids1, cand["s1"][k2], pc.take(ids23, pa.array(cand["s23"][k2])))
        log(f"   variant threshold {t:.2f}: {k2.sum():,} matches, "
            f"empty={np.mean(np.bincount(cand['s1'][k2], minlength=s1.num_rows) == 0):.3f}")
    per = np.bincount(cand["s1"][keep], minlength=s1.num_rows)
    pc_ = np.bincount(cand["s1"], minlength=s1.num_rows)
    ctry = np.array(s1.column("country").to_pylist())
    log(f"PREDICT: {keep.sum():,} matches for {s1.num_rows:,} S1 entities; "
        f"empty={np.mean(per == 0):.3f}; mean matches={per.mean():.2f}; mean candidates={pc_.mean():.2f}")
    for c in np.unique(ctry):
        m = ctry == c
        log(f"   [{c}] empty={np.mean(per[m] == 0):.3f} mean matches={per[m].mean():.2f} "
            f"mean candidates={pc_[m].mean():.2f}")
    log(f"wrote {args.out}/matching_results.tsv and candidate_pairs.tsv")


# ----------------------------------------------------------------------------------- main
def main():
    global LOG_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "block", "train", "predict", "all"])
    ap.add_argument("--data", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--work", default="cache", help="folder for intermediate files")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--out", default="output")
    ap.add_argument("--only", choices=["train", "test"], help="block: run for one split only")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. --set top_k=30 --set n_train_s1=150000")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in CFG:
            sys.exit(f"unknown setting {k}; choose from {sorted(CFG)}")
        CFG[k] = type(CFG[k])(float(v)) if isinstance(CFG[k], (int, float)) else v
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    LOG_FILE = os.path.join(args.model_dir, "run_log.txt")
    log(f"===== stage: {args.stage} =====  settings overridden: {args.set or 'none'}")
    if args.stage in ("prep", "all"):
        stage_prep(args)
    if args.stage in ("block", "all"):
        stage_block(args, (args.only,) if args.only else ("train", "test"))
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("predict", "all"):
        stage_predict(args)
    log("done")


if __name__ == "__main__":
    main()
