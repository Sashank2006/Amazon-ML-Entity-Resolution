"""
features.py - Step 3: describe every candidate pair with numbers ("features").

A machine-learning model cannot read text. Instead, for each pair (S1 record, S2/S3 record) we compute
~30 numbers such as "how similar are the two names (0-100)?" or "is the house number identical?".
The model (LightGBM) then learns from the training labels which combinations of numbers mean "same
business".

String similarity functions come from the `rapidfuzz` library (MIT licence), which runs in C++ and
compares millions of pairs per second.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler



def _pairwise(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _take(tbl, col, idx):
    return pc.take(tbl.column(col), pa.array(idx)).to_numpy(zero_copy_only=False)


def group_rank(keys, values, descending=True):
    """Rank of each value within its group (0 = best). Works on plain numpy arrays."""
    order = np.lexsort((-values if descending else values, keys))
    ks = keys[order]
    first = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    r = np.arange(len(ks)) - np.repeat(first, np.diff(np.r_[first, len(ks)]))
    out = np.empty(len(keys), np.int32)
    out[order] = r
    return out


def group_stat(keys, values, fn):
    """Per-row value of fn (np.maximum / np.add) aggregated over the row's group."""
    u, inv = np.unique(keys, return_inverse=True)
    agg = np.full(len(u), -np.inf if fn is np.maximum else 0, np.float64)
    fn.at(agg, inv, values)
    return agg[inv].astype(np.float32)


def pair_context(cand):
    """Features that look at a pair in the context of the other candidates (computed once, globally)."""
    s1, s23, bs = cand["s1"], cand["s23"], cand["bscore"].astype(np.float32)
    ctx = {}
    ctx["brank_s1"] = group_rank(s1, bs)                       # rank of this partner for the S1 record
    ctx["brank_s23"] = group_rank(s23, bs)                     # rank of this S1 for the S2/S3 record
    ctx["n_cand_s1"] = np.bincount(s1)[s1].astype(np.int32)
    ctx["n_cand_s23"] = np.bincount(s23)[s23].astype(np.int32)
    ctx["bscore_rel_s1"] = bs / np.maximum(group_stat(s1, bs, np.maximum), 1e-6)
    ctx["bscore_rel_s23"] = bs / np.maximum(group_stat(s23, bs, np.maximum), 1e-6)
    return ctx


def add_record_columns(tbl):
    """Per-record numbers computed once (much faster than per pair)."""
    from normalize import phonetic
    tbl = tbl.append_column("nm_ph", phonetic(tbl.column("nm")))
    tbl = tbl.append_column("nm_len", pc.utf8_length(tbl.column("nm")))
    tbl = tbl.append_column("ad_len", pc.utf8_length(tbl.column("ad")))
    nums = tbl.column("nums")
    n = pc.if_else(pc.equal(nums, ""), 0, pc.add(pc.count_substring(nums, " "), 1))
    return tbl.append_column("num_n", n)


def add_frequency_columns(s1, s23):
    """How common is each record's name / address, counted in Source 1 and in Source 2+3 (per country)?
    A unique name that matches exactly is strong evidence; a name or address shared by many
    businesses ("Modern Infra", a busy office building) is weak evidence."""
    for col, tag in (("nm_ns", "nm"), ("ad", "ad")):
        k1 = pc.binary_join_element_wise(s1.column("country"), s1.column(col), "|")
        k2 = pc.binary_join_element_wise(s23.column("country"), s23.column(col), "|")
        enc = pc.dictionary_encode(pa.chunked_array([k1, k2])).combine_chunks().indices.to_numpy()
        i1, i2 = enc[:len(k1)], enc[len(k1):]
        V = int(enc.max()) + 1 if len(enc) else 1
        c1, c2 = np.bincount(i1, minlength=V), np.bincount(i2, minlength=V)
        e1 = pc.equal(s1.column(col), "").to_numpy(zero_copy_only=False)
        e2 = pc.equal(s23.column(col), "").to_numpy(zero_copy_only=False)
        s1 = s1.append_column(f"{tag}_in_s1", pa.array(np.where(e1, -1, c1[i1]).astype(np.int32)))
        s1 = s1.append_column(f"{tag}_in_s23", pa.array(np.where(e1, -1, c2[i1]).astype(np.int32)))
        s23 = s23.append_column(f"{tag}_in_s1", pa.array(np.where(e2, -1, c1[i2]).astype(np.int32)))
        s23 = s23.append_column(f"{tag}_in_s23", pa.array(np.where(e2, -1, c2[i2]).astype(np.int32)))
    return s1, s23


FREQ_COLS = ("nm_in_s1", "nm_in_s23", "ad_in_s1", "ad_in_s23")


def top2_context(keys, values):
    """For each row: rank inside its group, the group's best value and the best value of the OTHER rows."""
    order = np.lexsort((-values, keys))
    ks, vs = keys[order], values[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, len(ks)])
    gmax = np.repeat(vs[start], size)
    second = np.where(size > 1, vs[np.minimum(start + 1, len(vs) - 1)], -1.0)
    gsecond = np.repeat(second, size)
    rank = np.arange(len(ks)) - np.repeat(start, size)
    other = np.where(rank == 0, gsecond, gmax)
    out_rank, out_max, out_other = (np.empty(len(keys), np.int32), np.empty(len(keys), np.float32),
                                    np.empty(len(keys), np.float32))
    out_rank[order], out_max[order], out_other[order] = rank, gmax, other
    return out_rank, out_max, out_other


def stage1_context(cand, p1):
    """Context built from the stage-1 (pruning) model probabilities: is this S2/S3 record better
    explained by another S1 entity? These are strong signals for the final model."""
    ctx = {}
    r, mx, oth = top2_context(cand["s23"], p1)
    ctx["p1"] = p1
    ctx["p1_rank_s23"] = r
    ctx["p1_rel_s23"] = p1 / np.maximum(mx, 1e-6)
    ctx["p1_other_s23"] = oth
    r, mx, oth = top2_context(cand["s1"], p1)
    ctx["p1_rank_s1"] = r
    ctx["p1_rel_s1"] = p1 / np.maximum(mx, 1e-6)
    return ctx


def _base(cand, ctx, idx, s1_tbl, s23_tbl, i1, i2):
    f = {}
    for k in ("bscore", "nkeys", "name_keys", "addr_keys"):
        f[k] = cand[k][idx].astype(np.float32)
    for k, v in ctx.items():
        f[k] = v[idx].astype(np.float32)
    for c in ("nm_len", "ad_len", "num_n") + FREQ_COLS:
        f[c + "1"] = _take(s1_tbl, c, i1).astype(np.float32)
        f[c + "2"] = _take(s23_tbl, c, i2).astype(np.float32)
    return f


def compute_cheap_features(s1_tbl, s23_tbl, cand, ctx, idx, workers=-1):
    """Small, fast feature set for the stage-1 pruning model (run on ALL blocking pairs)."""
    i1, i2 = cand["s1"][idx], cand["s23"][idx]
    f = _base(cand, ctx, idx, s1_tbl, s23_tbl, i1, i2)
    n1, n2 = _take(s1_tbl, "nm", i1), _take(s23_tbl, "nm", i2)
    f["nm_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    del n1, n2
    n1, n2 = _take(s1_tbl, "nm_ns", i1), _take(s23_tbl, "nm_ns", i2)
    f["ns_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    del n1, n2
    n1, n2 = _take(s1_tbl, "nm_ph", i1), _take(s23_tbl, "nm_ph", i2)
    f["ph_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    del n1, n2
    a1, a2 = _take(s1_tbl, "ad", i1), _take(s23_tbl, "ad", i2)
    f["ad_tset"] = _pairwise(a1, a2, fuzz.token_set_ratio, workers)
    del a1, a2
    a1, a2 = _take(s1_tbl, "nums", i1), _take(s23_tbl, "nums", i2)
    f["num_tset"] = _pairwise(a1, a2, fuzz.token_set_ratio, workers)
    return pd.DataFrame(f)


def compute_features(s1_tbl, s23_tbl, cand, ctx, idx, workers=-1):
    """Full feature set for the final matching model."""
    i1, i2 = cand["s1"][idx], cand["s23"][idx]
    f = _base(cand, ctx, idx, s1_tbl, s23_tbl, i1, i2)
    n1, n2 = _take(s1_tbl, "nm", i1), _take(s23_tbl, "nm", i2)
    f["nm_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    f["nm_tsort"] = _pairwise(n1, n2, fuzz.token_sort_ratio, workers)
    f["nm_tset"] = _pairwise(n1, n2, fuzz.token_set_ratio, workers)
    f["nm_partial"] = _pairwise(n1, n2, fuzz.partial_ratio, workers)
    f["nm_jw"] = _pairwise(n1, n2, JaroWinkler.normalized_similarity, workers)
    del n1, n2
    n1, n2 = _take(s1_tbl, "nm_ns", i1), _take(s23_tbl, "nm_ns", i2)
    f["ns_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    f["ns_partial"] = _pairwise(n1, n2, fuzz.partial_ratio, workers)
    f["ns_equal"] = (n1 == n2).astype(np.float32)
    del n1, n2
    n1, n2 = _take(s1_tbl, "nm_ph", i1), _take(s23_tbl, "nm_ph", i2)
    f["ph_ratio"] = _pairwise(n1, n2, fuzz.ratio, workers)
    f["ph_tset"] = _pairwise(n1, n2, fuzz.token_set_ratio, workers)
    del n1, n2
    f["nonlatin2"] = _take(s23_tbl, "non_latin", i2).astype(np.float32)
    a1, a2 = _take(s1_tbl, "ad", i1), _take(s23_tbl, "ad", i2)
    f["ad_ratio"] = _pairwise(a1, a2, fuzz.ratio, workers)
    f["ad_tset"] = _pairwise(a1, a2, fuzz.token_set_ratio, workers)
    f["ad_tsort"] = _pairwise(a1, a2, fuzz.token_sort_ratio, workers)
    f["ad_partial"] = _pairwise(a1, a2, fuzz.partial_ratio, workers)
    del a1, a2
    a1, a2 = _take(s1_tbl, "ad_alpha", i1), _take(s23_tbl, "ad_alpha", i2)
    f["st_tset"] = _pairwise(a1, a2, fuzz.token_set_ratio, workers)
    f["st_partial"] = _pairwise(a1, a2, fuzz.partial_ratio, workers)
    del a1, a2
    a1, a2 = _take(s1_tbl, "nums", i1), _take(s23_tbl, "nums", i2)
    f["num_tset"] = _pairwise(a1, a2, fuzz.token_set_ratio, workers)
    f["num_tsort"] = _pairwise(a1, a2, fuzz.token_sort_ratio, workers)
    del a1, a2
    h1, h2 = _take(s1_tbl, "hn", i1), _take(s23_tbl, "hn", i2)
    has = (h1 != "") & (h2 != "")
    f["hn_equal"] = np.where(has, (h1 == h2).astype(np.float32), np.float32(-1))
    f["hn_ratio"] = np.where(has, _pairwise(h1, h2, fuzz.ratio, workers), np.float32(-1))
    del h1, h2
    f["is_s3"] = _take(s23_tbl, "is_s3", i2).astype(np.float32)
    return pd.DataFrame(f)
