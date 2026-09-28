"""
blocking.py - Step 2: candidate generation ("blocking").

Problem: comparing every Source-1 record with every Source-2/3 record is ~1.7M x 10M = 17 trillion
pairs - impossible. Blocking cheaply finds a short list of plausible partners for each S1 record.

Method: "token blocking + block purging + meta-blocking" (standard entity-resolution techniques)
  1. For each record we create BLOCKING KEYS (short codes). Examples for
     "Davis Family Office | 88 Olive Circle, Lebanon":
       UNI  one name word                 davis
       BI   two name words                davis|family
       NS   name without spaces           davisfamilyoffice   (matches "davisfamilyoffice.com")
       PB   two name-word prefixes (4)    davi|fami           (survives typos: "Famliy")
       NN   name word + house number      davis|88
       X    house number + address word   88|olive            (works when the name is garbage)
       Y    two numbers of the address    12|190
       PA   name prefix + address prefix  davi|oliv           (for addresses with no number)
       PHU/PHB/PHN  the same ideas on the SOUND of the name ("consultancy" and the Hindi-script
            transliteration "kansaltensi" both sound like "knsltns")
  2. Two records become candidates if they share at least one key (same country only).
  3. Keys that are too common are thrown away ("block purging").
  4. Each shared key adds a weight (rarer key = more weight). For every S1 record we keep only the
     top-K partners by total weight ("meta-blocking").

Memory trick: keys are built from Source 1 first. Source 2/3 is then processed in chunks and we keep
only keys that also exist in Source 1 - everything else can never create a pair.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from normalize import phonetic

TYPE_BITS = 56
T_UNI, T_BI, T_NS, T_PB, T_NN, T_X, T_Y, T_PA, T_PHU, T_PHB, T_PHN = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
NAME_TYPES = np.array([T_UNI, T_BI, T_NS, T_PB, T_PHU, T_PHB])
ADDR_TYPES = np.array([T_X, T_Y])
TYPE_WEIGHT = np.array([0, 1.0, 1.4, 1.5, 0.8, 1.2, 1.3, 0.8, 0.6, 0.7, 1.0, 1.0], np.float32)
VALUE_MASK = (np.int64(1) << np.int64(TYPE_BITS)) - 1


def _tokens(arr, pattern, cap):
    """Split strings on spaces, keep tokens matching `pattern`, at most `cap` per row.
    Returns (flat_tokens pyarrow array, row_index numpy array)."""
    lists = pc.split_pattern(arr, " ")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = pc.match_substring_regex(flat, pattern).to_numpy(zero_copy_only=False)
    flat, parent = pc.filter(flat, pa.array(keep)), parent[keep]
    if len(parent):
        starts = np.r_[0, np.flatnonzero(np.diff(parent)) + 1]
        run_start = np.repeat(starts, np.diff(np.r_[starts, len(parent)]))
        m = (np.arange(len(parent)) - run_start) < cap
        flat, parent = pc.filter(flat, pa.array(m)), parent[m]
    return flat, parent


def _row_product(ra, ca, rb, cb, n_rows, upper_only=False):
    """For every row: all combinations (a, b) of its A-codes and B-codes. Rows must be sorted."""
    na = np.bincount(ra, minlength=n_rows)
    nb = np.bincount(rb, minlength=n_rows)
    oa = np.r_[0, np.cumsum(na)[:-1]]
    ob = np.r_[0, np.cumsum(nb)[:-1]]
    npair = na * nb
    total = int(npair.sum())
    if total == 0:
        z = np.zeros(0, np.int64)
        return z.astype(np.int32), z, z
    rows = np.repeat(np.arange(n_rows, dtype=np.int32), npair)
    k = np.arange(total, dtype=np.int64) - np.repeat(np.r_[0, np.cumsum(npair)[:-1]], npair)
    nbr = nb[rows]
    i = k // nbr
    j = k % nbr
    a = ca[oa[rows] + i]
    b = cb[ob[rows] + j]
    if upper_only:
        m = i < j
        rows, a, b = rows[m], a[m], b[m]
    return rows, a.astype(np.int64), b.astype(np.int64)


def extract_tokens(tbl, cfg):
    """All token lists we need for the keys of one table."""
    nm, ad = tbl.column("nm"), tbl.column("ad")
    t = {}
    t["name"] = _tokens(nm, r"^[a-z0-9]{2,}$", cfg["max_name_tokens"])
    t["addr"] = _tokens(ad, r"^[a-z][a-z0-9]{2,}$", cfg["max_addr_tokens"])
    t["num"] = _tokens(ad, r"^[0-9][0-9a-z]*$", cfg["max_num_tokens"])
    ns = tbl.column("nm_ns")
    ok = pc.greater_equal(pc.utf8_length(ns), 4).to_numpy(zero_copy_only=False)
    t["ns"] = (pc.filter(ns, pa.array(ok)), np.flatnonzero(ok))
    # 4-letter prefixes (robust to typos at the end of words)
    pn_tok, pn_row = _tokens(nm, r"^[a-z][a-z0-9]{2,}$", cfg["max_prefix_tokens"])
    t["pname"] = (pc.utf8_slice_codeunits(pn_tok, 0, 4), pn_row)
    pa_tok, pa_row = _tokens(ad, r"^[a-z][a-z0-9]{3,}$", cfg["max_addr_prefix_tokens"])
    t["paddr"] = (pc.utf8_slice_codeunits(pa_tok, 0, 4), pa_row)
    t["phon"] = _tokens(phonetic(nm), r"^[a-z]{3,}$", cfg["max_name_tokens"])
    t["n"] = tbl.num_rows
    return t


def build_vocab(tok):
    """Vocabulary (sorted unique strings) of every token appearing in Source 1."""
    parts = [tok[k][0] for k in ("name", "addr", "num", "ns", "pname", "paddr", "phon")]
    return pc.unique(pa.chunked_array(parts, type=pa.string()).combine_chunks())


def keys_from_tokens(tok, vocab):
    """Turn tokens into int64 keys: [type (8 bits) | value (56 bits)]. Tokens not in the vocabulary
    are dropped (they cannot be shared with Source 1). Returns (keys, rows) sorted by row."""
    V = np.int64(len(vocab) + 1)
    codes, rows = {}, {}
    for k in ("name", "addr", "num", "ns", "pname", "paddr", "phon"):
        idx = pc.index_in(tok[k][0], value_set=vocab)
        valid = pc.is_valid(idx).to_numpy(zero_copy_only=False)
        codes[k] = pc.filter(idx, pa.array(valid)).to_numpy().astype(np.int64) if valid.any() else np.zeros(0, np.int64)
        rows[k] = tok[k][1][valid]
    n = tok["n"]
    ks, rs = [], []

    def add(t, r, v):
        ks.append((np.int64(t) << np.int64(TYPE_BITS)) | v); rs.append(np.asarray(r, np.int32))

    add(T_UNI, rows["name"], codes["name"])
    r, a, b = _row_product(rows["name"], codes["name"], rows["name"], codes["name"], n, True)
    add(T_BI, r, np.minimum(a, b) * V + np.maximum(a, b))
    add(T_NS, rows["ns"], codes["ns"])
    r, a, b = _row_product(rows["pname"], codes["pname"], rows["pname"], codes["pname"], n, True)
    add(T_PB, r, np.minimum(a, b) * V + np.maximum(a, b))
    r, a, b = _row_product(rows["name"], codes["name"], rows["num"], codes["num"], n)
    add(T_NN, r, a * V + b)
    r, a, b = _row_product(rows["num"], codes["num"], rows["addr"], codes["addr"], n)
    add(T_X, r, a * V + b)
    r, a, b = _row_product(rows["num"], codes["num"], rows["num"], codes["num"], n, True)
    add(T_Y, r, np.minimum(a, b) * V + np.maximum(a, b))
    r, a, b = _row_product(rows["pname"], codes["pname"], rows["paddr"], codes["paddr"], n)
    add(T_PA, r, a * V + b)
    add(T_PHU, rows["phon"], codes["phon"])
    r, a, b = _row_product(rows["phon"], codes["phon"], rows["phon"], codes["phon"], n, True)
    add(T_PHB, r, np.minimum(a, b) * V + np.maximum(a, b))
    r, a, b = _row_product(rows["phon"], codes["phon"], rows["num"], codes["num"], n)
    add(T_PHN, r, a * V + b)
    keys = np.concatenate(ks)
    rws = np.concatenate(rs)
    order = np.lexsort((keys, rws))
    keys, rws = keys[order], rws[order]
    dup = np.r_[False, (keys[1:] == keys[:-1]) & (rws[1:] == rws[:-1])]
    return keys[~dup], rws[~dup]


def generate_candidates(s1_tbl, s23_tbl, cfg, s1_mask=None, log=print):
    """Candidate pairs as numpy arrays of row indices into s1_tbl / s23_tbl plus blocking features.
    s1_mask: optional boolean array - only produce candidates for these S1 rows (training subset)."""
    K = cfg["top_k"]
    res = {k: [] for k in ("s1", "s23", "bscore", "nkeys", "name_keys", "addr_keys")}
    countries = sorted(set(pc.unique(s1_tbl.column("country")).to_pylist()))
    c23_all = s23_tbl.column("country")
    for ctry in countries:
        i1 = np.flatnonzero(pc.equal(s1_tbl.column("country"), ctry).to_numpy(zero_copy_only=False))
        i23 = np.flatnonzero(pc.equal(c23_all, ctry).to_numpy(zero_copy_only=False))
        if len(i1) == 0 or len(i23) == 0:
            continue
        # ---- Source 1 keys
        tok1 = extract_tokens(s1_tbl.take(pa.array(i1)), cfg)
        vocab = build_vocab(tok1)
        k1, r1 = keys_from_tokens(tok1, vocab)
        del tok1
        u1, c1 = np.unique(k1, return_counts=True)
        useful = u1[c1 <= cfg["max_s1_per_key"]]           # purge keys too common in S1
        # ---- Source 2+3 keys, chunk by chunk, keeping only keys that exist in `useful`
        k23l, r23l = [], []
        ch = cfg["s23_chunk"]
        for a in range(0, len(i23), ch):
            sub = i23[a:a + ch]
            k, r = keys_from_tokens(extract_tokens(s23_tbl.take(pa.array(sub)), cfg), vocab)
            pos = np.clip(np.searchsorted(useful, k), 0, len(useful) - 1)
            m = useful[pos] == k
            k23l.append(k[m]); r23l.append((r[m] + a).astype(np.int32))
        k23 = np.concatenate(k23l); r23 = np.concatenate(r23l)
        del k23l, r23l
        o = np.argsort(k23, kind="stable")
        k23, r23 = k23[o], r23[o]
        del o
        u23, st23, c23 = np.unique(k23, return_index=True, return_counts=True)
        del k23
        # ---- attach frequencies to every S1 key row
        cnt1 = c1[np.searchsorted(u1, k1)]
        p = np.clip(np.searchsorted(u23, k1), 0, max(len(u23) - 1, 0))
        present = (u23[p] == k1) if len(u23) else np.zeros(len(k1), bool)
        cnt23 = np.where(present, c23[p] if len(u23) else 0, 0)
        start23 = np.where(present, st23[p] if len(u23) else 0, 0)
        ok = present & (cnt1 <= cfg["max_s1_per_key"]) & (cnt23 <= cfg["max_s23_per_key"])
        if s1_mask is not None:
            ok &= s1_mask[i1[r1]]
        ktype = (k1 >> np.int64(TYPE_BITS)).astype(np.int8)
        w = TYPE_WEIGHT[ktype] / np.log2(2.0 + cnt1 * cnt23).astype(np.float32)
        rows1, st, cn, w, ktype = r1[ok], start23[ok], cnt23[ok], w[ok].astype(np.float32), ktype[ok]
        del k1, r1, cnt1, p, present, cnt23, start23, ok, u1, c1, u23, st23, c23
        # ---- expand to pairs, S1-chunk by S1-chunk, and keep the top-K per S1 record
        n23 = np.int64(len(i23))
        bounds = np.searchsorted(rows1, np.arange(0, len(i1) + cfg["s1_chunk"], cfg["s1_chunk"]))
        before = sum(len(x) for x in res["s1"])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            cnts = cn[a:b]
            tot = int(cnts.sum())
            rep = np.repeat(np.arange(b - a), cnts)
            off = np.arange(tot) - np.repeat(np.r_[0, np.cumsum(cnts)[:-1]], cnts)
            s23_loc = r23[st[a:b][rep] + off].astype(np.int64)
            code = rows1[a:b][rep].astype(np.int64) * n23 + s23_loc
            wt = w[a:b][rep]
            ty = ktype[a:b][rep]
            uc, inv = np.unique(code, return_inverse=True)
            score = np.bincount(inv, weights=wt).astype(np.float32)
            nk = np.bincount(inv).astype(np.int16)
            nn = np.bincount(inv, weights=np.isin(ty, NAME_TYPES)).astype(np.int16)
            na = np.bincount(inv, weights=np.isin(ty, ADDR_TYPES)).astype(np.int16)
            ps1, ps23 = uc // n23, uc % n23
            order = np.lexsort((-score, ps1))
            ps1, ps23, score, nk, nn, na = ps1[order], ps23[order], score[order], nk[order], nn[order], na[order]
            first = np.r_[0, np.flatnonzero(np.diff(ps1)) + 1]
            rank = np.arange(len(ps1)) - np.repeat(first, np.diff(np.r_[first, len(ps1)]))
            keep = (rank < K) & (score >= cfg["min_score"])
            res["s1"].append(i1[ps1[keep]].astype(np.int32))
            res["s23"].append(i23[ps23[keep]].astype(np.int32))
            res["bscore"].append(score[keep]); res["nkeys"].append(nk[keep])
            res["name_keys"].append(nn[keep]); res["addr_keys"].append(na[keep])
        after = sum(len(x) for x in res["s1"])
        log(f"  [{ctry}] S1={len(i1):,}  S2+S3={len(i23):,}  ->  {after - before:,} candidate pairs")
        del r23, rows1, st, cn, w, ktype
    return {k: (np.concatenate(v) if v else np.zeros(0, np.int32)) for k, v in res.items()}
