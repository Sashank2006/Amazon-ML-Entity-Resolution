"""
demo_model.py - a small, fully transparent version of the matcher that runs in a web browser.

The browser page (web/index.html) re-implements these exact functions in JavaScript, so this file
is the reference. It trains a logistic-regression model on the FAKE demo data (never on the
competition data) and exports the weights plus the word lists to web/build/model.json.

Run:  python web/build/demo_model.py --data demo_data --cache demo_cache
"""
import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import normalize as N  # noqa: E402  (only for the shared word lists)

# ---------------------------------------------------------------- Devanagari -> Latin (simple)
CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
        "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f"}
VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
       "औ": "au", "ऑ": "o"}
MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
         "ॉ": "o"}
VIRAMA, NUKTA = "्", "़"
NASAL = {"ं": "n", "ँ": "n", "ः": "h"}


def devanagari_to_latin(s):
    out, i = [], 0
    chars = list(s)
    while i < len(chars):
        ch = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == NUKTA and (ch + NUKTA) in CONS:
            ch = ch + NUKTA
            i += 1
        if ch in CONS:
            out.append(CONS[ch])
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in MATRA:
                out.append(MATRA[nxt]); i += 1
            elif nxt == VIRAMA:
                i += 1
            else:
                out.append("a")
        elif ch in VOW:
            out.append(VOW[ch])
        elif ch in NASAL:
            out.append(NASAL[ch])
        elif ch in MATRA or ch == VIRAMA or ch == NUKTA:
            pass
        else:
            out.append(ch)
        i += 1
    # drop the silent final 'a' of each word (schwa deletion), e.g. 'kansaltensee'
    words = "".join(out).split(" ")
    words = [w[:-1] if len(w) > 3 and w.endswith("a") and not w.endswith("aa") else w for w in words]
    return " ".join(words)


HAS_DEVA = re.compile("[ऀ-ॿ]")


def fold(s):
    if HAS_DEVA.search(s):
        s = devanagari_to_latin(s)
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def squash(s):
    return " ".join(s.split())


NAME_STOP = set(N.NAME_STOP) | {"praivet", "praaivet", "limited", "limitad", "sarvisej", "sarvises", "sarvis",
                                "kampani", "kampanee", "enterprij", "enterprijej"}
ADDR_STOP = set(N.ADDR_STOP)
STATES = set(N.STATE_SET)
CANON = {v: k for k, vs in N.ADDR_CANON.items() for v in vs}


def norm_name(raw):
    s = fold(raw)
    s = re.sub(r"\bdba\b\s*:?", " ", s)
    s = re.sub(r"\.(com|net|org|in|co|fr|us)\b", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"(\w)'s\b", r"\1s", s)
    s = re.sub(r"\b([a-z])\.(?:[a-z]\.)+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in NAME_STOP)


def norm_addr(raw):
    s = fold(raw)
    s = re.sub(r"<null>|\bn/a\b", " ", s)
    segs = []
    for seg in s.split(","):
        c = squash(re.sub(r"[^a-z0-9]", " ", seg))
        if c and c not in STATES:
            segs.append(seg)
    s = " , ".join(segs)
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"(\d)([a-z]{3,})", r"\1 \2", s)
    s = re.sub(r"([a-z]{3,})(\d)", r"\1 \2", s)
    toks = [CANON.get(w, w) for w in s.split()]
    toks = [re.sub(r"^0+(\d)", r"\1", w) for w in toks]
    return " ".join(w for w in toks if w not in ADDR_STOP)


PHON_RULES = [(r"igh", "i"), (r"ph", "f"), (r"ck", "k"), (r"x", "ks"), (r"q", "k"), (r"tion", "shon"),
              (r"c([eiy])", r"s\1"), (r"g([ei])", r"j\1"), (r"c", "k"), (r"([sdtbkgj])h", r"\1"),
              (r"w", "v"), (r"z", "j"), (r"j\b", "s"), (r"\B[aeiouy]", ""), (r"\Bh", "")]


def phonetic(s):
    for p, r in PHON_RULES:
        s = re.sub(p, r, s)
    return re.sub(r"([bdfgjklmnprstv])\1+", r"\1", s)


# ---------------------------------------------------------------- similarities (0..1)
def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def ratio(a, b):
    if not a and not b:
        return 0.0
    return 2.0 * lcs(a, b) / (len(a) + len(b))


def token_set(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    inter = " ".join(sorted(A & B))
    da = " ".join(sorted(A - B))
    db = " ".join(sorted(B - A))
    if inter and (not da or not db):
        return 1.0
    s1 = (inter + " " + da).strip()
    s2 = (inter + " " + db).strip()
    best = ratio(s1, s2)
    if inter:
        best = max(best, ratio(inter, s1), ratio(inter, s2))
    return best


def split_addr(ad):
    toks = ad.split()
    nums = [t for t in toks if t[0].isdigit()]
    alpha = [t for t in toks if not t[0].isdigit()]
    return nums, alpha


FEATURES = ["name_ratio", "name_tokens", "name_nospace", "name_sound", "addr_tokens", "street_tokens",
            "number_overlap", "house_no_differs"]
MISSING_ADDR = 0.5   # neutral value used when one of the addresses is empty


def features(n1raw, a1raw, n2raw, a2raw):
    n1, n2 = norm_name(n1raw), norm_name(n2raw)
    a1, a2 = norm_addr(a1raw), norm_addr(a2raw)
    num1, al1 = split_addr(a1)
    num2, al2 = split_addr(a2)
    hn1 = max(num1, key=len) if num1 else ""
    hn2 = max(num2, key=len) if num2 else ""
    name = [ratio(n1, n2), token_set(n1, n2), ratio(n1.replace(" ", ""), n2.replace(" ", "")),
            ratio(phonetic(n1), phonetic(n2))]
    if not a1 or not a2:
        addr = [MISSING_ADDR, MISSING_ADDR, MISSING_ADDR, 0.0]
    else:
        nums = token_set(" ".join(num1), " ".join(num2)) if (num1 and num2) else MISSING_ADDR
        addr = [token_set(a1, a2), token_set(" ".join(al1), " ".join(al2)), nums,
                float(bool(hn1 and hn2) and hn1 != hn2)]
    return name + addr


if __name__ == "__main__":
    import pyarrow.parquet as pq
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--cache", default="demo_cache")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json"))
    a = ap.parse_args()
    s1 = pq.read_table(f"{a.cache}/train_s1.parquet", columns=["entity_id", "raw_name", "raw_addr"]).to_pandas()
    s23 = pq.read_table(f"{a.cache}/train_s23.parquet", columns=["entity_id", "raw_name", "raw_addr"]).to_pandas()
    z = np.load(f"{a.cache}/train_cand.npz")
    import pandas as pd
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    truth = {(x, m) for x, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(z["s1"]), size=min(40000, len(z["s1"])), replace=False)
    X, y = [], []
    for i in idx:
        r1, r2 = s1.iloc[z["s1"][i]], s23.iloc[z["s23"][i]]
        X.append(features(r1.raw_name, r1.raw_addr, r2.raw_name, r2.raw_addr))
        y.append((r1.entity_id, r2.entity_id) in truth)
    X, y = np.array(X), np.array(y)
    n = int(len(y) * 0.8)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[:n], y[:n])
    p = m.predict_proba(X[n:])[:, 1]
    for t in (0.5, 0.7, 0.8, 0.9):
        k = p >= t
        print(f"thr {t}: precision {(k & y[n:]).sum() / max(k.sum(), 1):.3f}  recall {(k & y[n:]).sum() / y[n:].sum():.3f}")
    json.dump(dict(features=FEATURES, weights=m.coef_[0].round(4).tolist(), bias=round(float(m.intercept_[0]), 4),
                   name_stop=sorted(NAME_STOP), addr_stop=sorted(ADDR_STOP), states=sorted(STATES), canon=CANON,
                   n_train=int(n), positives=int(y.sum())),
              open(a.out, "w"), indent=1)
    print("weights:", dict(zip(FEATURES, m.coef_[0].round(2))), "bias", round(float(m.intercept_[0]), 2))
