"""
normalize.py - Step 1 of the pipeline: load the raw TSV files and clean the text.

Why: the same business is written in many different ways ("Pvt Ltd" vs "Private Limited",
"RD" vs "Road", "é" vs "e", Hindi script vs English). Before we can compare two records we
turn every name and address into one standard ("canonical") form.

Everything here is vectorised with pyarrow.compute (runs in C++, many rows at once),
because a Python for-loop over ~12 million rows would take far too long.
Only non-Latin names (Hindi, Telugu, ...) go through a small Python loop for transliteration.
"""
import re
import unicodedata

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# --------------------------------------------------------------------------------------
# Word lists (general language knowledge only - no business identity lookup of any kind)
# --------------------------------------------------------------------------------------
# Legal / generic words that say nothing about WHICH business it is.
NAME_STOP = """
pvt private ltd limited llc l l c inc incorporated corp corporation co company plc pllc llp lp
the and of sri shri mr ms mrs m s dba services service center centre group india france us usa
sarl sas sasu eurl sa sci ei snc cie et de du des la le les d l s a r
www com net org in co in
""".split()

# Address words -> one canonical spelling (English + French abbreviations).
ADDR_CANON = {
    "st": ["street", "str", "st"], "rd": ["road", "rd"], "ave": ["avenue", "av", "ave", "avn"],
    "dr": ["drive", "dr"], "ln": ["lane", "ln"], "blvd": ["boulevard", "blvd", "bd", "bld", "boul"],
    "ct": ["court", "ct"], "cir": ["circle", "cir"], "pl": ["place", "pl"], "pkwy": ["parkway", "pkwy"],
    "hwy": ["highway", "hwy"], "ter": ["terrace", "ter"], "sq": ["square", "sq"], "trl": ["trail", "trl"],
    "rue": ["rue", "r"], "all": ["allee", "all"], "rte": ["route", "rte"], "ch": ["chemin", "ch"],
    "imp": ["impasse", "imp"], "fl": ["floor", "flr", "fl"], "apt": ["apartment", "apt"],
    "ste": ["suite", "ste"], "n": ["north", "n"], "s": ["south", "s"], "e": ["east", "e"], "w": ["west", "w"],
    "ngr": ["nagar", "ngr"], "opp": ["opposite", "opp"], "nr": ["near", "nr"],
}
# Address filler words that carry no location information.
ADDR_STOP = """
no h hno house door flat plot shop office unit n a na null none pmb po box number at near nr opp
behind c o s o d o w o of the and bis ter
""".split()

# State / region names and codes. Whole comma-separated segments equal to one of these are removed,
# because they are written inconsistently (MH / Maharashtra / महाराष्ट्र) and are too broad to help.
US_STATES = """alabama al alaska ak arizona az arkansas ar california ca colorado co connecticut ct delaware de
florida fl georgia ga hawaii hi idaho id illinois il indiana in iowa ia kansas ks kentucky ky louisiana la
maine me maryland md massachusetts ma michigan mi minnesota mn mississippi ms missouri mo montana mt
nebraska ne nevada nv new_hampshire nh new_jersey nj new_mexico nm new_york ny north_carolina nc
north_dakota nd ohio oh oklahoma ok oregon or pennsylvania pa rhode_island ri south_carolina sc
south_dakota sd tennessee tn texas tx utah ut vermont vt virginia va washington wa west_virginia wv
wisconsin wi wyoming wy district_of_columbia dc puerto_rico pr"""
IN_STATES = """andhra_pradesh ap arunachal_pradesh ar assam as bihar br chhattisgarh cg ct goa ga gujarat gj
haryana hr himachal_pradesh hp jharkhand jh karnataka ka kerala kl keralam madhya_pradesh mp maharashtra mh
manipur mn meghalaya ml mizoram mz nagaland nl odisha orissa od or punjab pb rajasthan rj sikkim sk
tamil_nadu tn telangana tg ts tripura tr uttar_pradesh up uttarakhand uttaranchal uk ut west_bengal wb
delhi dl new_delhi chandigarh ch puducherry pondicherry py jammu_and_kashmir jammu_kashmir jk ladakh la
andaman_and_nicobar_islands an dadra_and_nagar_haveli dn daman_and_diu dd lakshadweep ld"""
FR_REGIONS = """hauts_de_france nouvelle_aquitaine pays_de_la_loire ile_de_france auvergne_rhone_alpes
bourgogne_franche_comte bretagne centre_val_de_loire corse grand_est normandie occitanie
provence_alpes_cote_d_azur nord pas_de_calais gironde loire_atlantique somme aisne oise vendee
maine_et_loire sarthe mayenne landes dordogne charente charente_maritime pyrenees_atlantiques
lot_et_garonne correze creuse haute_vienne vienne deux_sevres paris bouches_du_rhone rhone"""
STATE_SET = set(w.replace("_", " ") for w in (US_STATES + " " + IN_STATES + " " + FR_REGIONS).split())

# Words used when an Indian-language name is transliterated to Latin letters.
TRANSLIT_FIX = {"praiveta": "private", "praivet": "private", "praibhet": "private", "bhiraivedh": "private",
                "limiteda": "limited", "limitedda": "limited", "limidhedh": "limited", "limatida": "limited",
                "limited": "limited"}


# --------------------------------------------------------------------------------------
def read_tsv(path):
    """Read a challenge TSV as all-string columns. quote_char=False: quotes are normal text."""
    tbl = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(strings_can_be_null=False, column_types={
            "entity_id": pa.string(), "business_name": pa.string(), "business_address": pa.string(),
            "country": pa.string(), "source1_entity_id": pa.string(), "matched_entity_ids": pa.string()}),
        read_options=pacsv.ReadOptions(block_size=1 << 26),
    )
    return tbl


def _ascii_fold(arr):
    """lower-case + remove accents: 'Élan Café' -> 'elan cafe'."""
    arr = pc.utf8_lower(arr)
    arr = pc.utf8_normalize(arr, "NFKD")
    arr = pc.replace_substring_regex(arr, r"\p{Mn}", "")  # drop combining accent marks
    return arr


_INDIC = None


def _transliterate_list(strings):
    """Convert Hindi/Telugu/Tamil/... script to Latin letters, e.g. 'सुप्रीम' -> 'suprima'."""
    global _INDIC
    if _INDIC is None:
        from indic_transliteration import sanscript
        from indic_transliteration.detect import detect
        _INDIC = (sanscript, detect)
    sanscript, detect = _INDIC
    out = []
    for s in strings:
        try:
            t = sanscript.transliterate(s, detect(s), sanscript.IAST)
        except Exception:
            t = s
        t = unicodedata.normalize("NFKD", t.replace("‌", "").replace("‍", ""))
        t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
        t = " ".join(_fix_translit_word(w) for w in re.split(r"\s+", t))
        out.append(t)
    return out


def _fix_translit_word(w):
    """Make phonetic spellings closer to English: 'tredimga' -> 'treding', 'dijitala' -> 'dijital'."""
    w = TRANSLIT_FIX.get(w, w)
    w = re.sub(r"m(?=[tdgkcj])", "n", w)          # nasal sign before these letters sounds like 'n'
    if len(w) > 4 and w.endswith("a") and not w.endswith("aa"):
        w = w[:-1]                                  # silent final 'a' of Indian scripts
    return TRANSLIT_FIX.get(w, w)


def _remove_words(arr, words):
    pat = r"\b(" + "|".join(sorted(set(re.escape(w) for w in words), key=len, reverse=True)) + r")\b"
    return pc.replace_substring_regex(arr, pat, " ")


def _squash(arr):
    arr = pc.replace_substring_regex(arr, r"\s+", " ")
    return pc.utf8_trim_whitespace(arr)


def normalize_names(names):
    """Return (clean_name, name_is_non_latin flag)."""
    names = pc.cast(names, pa.string())
    non_latin = pc.match_substring_regex(names, r"[^\x00-\x7F\p{Latin}\p{P}\p{S}\p{N}\s]")
    s = _ascii_fold(names)
    # transliterate only the rows that contain non-Latin script (a small minority)
    n_nl = pc.sum(non_latin).as_py() or 0
    if n_nl:
        vals = pc.filter(names, non_latin).to_pylist()   # original text: folding would break Indic scripts
        s = pc.replace_with_mask(s, non_latin, pa.array(_transliterate_list(vals), pa.string()))
    # common character-level tricks used in the noisy data
    s = pc.replace_substring_regex(s, r"\bdba\b\s*:?", " ")         # "X DBA: Y" -> keep both parts
    s = pc.replace_substring_regex(s, r"\.(com|net|org|in|co|fr|us)\b", " ")  # teamair.com -> teamair
    s = pc.replace_substring(s, "&", " and ")
    s = pc.replace_substring_regex(s, r"(\w)'s\b", "\\1s")          # mendez's -> mendezs
    s = pc.replace_substring_regex(s, r"\b([a-z])\.(?:([a-z])\.)+", "")  # l.l.c. / s.a.r.l.
    s = pc.replace_substring_regex(s, r"[^a-z0-9 ]", " ")           # punctuation -> space
    s = _remove_words(s, NAME_STOP)
    return _squash(s), non_latin


def normalize_addresses(addr):
    """Clean an address: lower-case, drop state/region segments, canonical abbreviations."""
    a = _ascii_fold(pc.cast(addr, pa.string()))
    a = pc.replace_substring_regex(a, r"<null>|\bn/a\b", " ")
    # drop comma-separated segments that are only a state name/code or have no Latin text
    segs = pc.split_pattern(a, ",")
    flat = pc.utf8_trim_whitespace(pc.list_flatten(segs))
    flat_clean = _squash(pc.replace_substring_regex(flat, r"[^a-z0-9]", " "))
    keep = pc.and_(pc.invert(pc.is_in(flat_clean, value_set=pa.array(sorted(STATE_SET)))),
                   pc.match_substring_regex(flat, r"[a-z0-9]"))
    flat_kept = pc.if_else(keep, flat, pa.scalar("", pa.string()))
    rebuilt = pa.ListArray.from_arrays(segs.offsets, flat_kept)
    a = pc.binary_join(rebuilt, " , ")
    a = pc.replace_substring_regex(a, r"(\d)(st|nd|rd|th)\b", "\\1")  # 1st -> 1
    a = pc.replace_substring_regex(a, r"[^a-z0-9 ]", " ")
    a = pc.replace_substring_regex(a, r"(\d)([a-z]{3,})", "\\1 \\2")   # 339nehru -> 339 nehru
    a = pc.replace_substring_regex(a, r"([a-z]{3,})(\d)", "\\1 \\2")   # street10 -> street 10
    for canon, variants in ADDR_CANON.items():
        pat = r"\b(" + "|".join(variants) + r")\b"
        a = pc.replace_substring_regex(a, pat, canon)
    a = pc.replace_substring_regex(a, r"\b0+(\d)", "\\1")  # 0044023 -> 44023
    return _squash(a)


def prepare(path):
    """Read one source TSV and add the normalised columns. Returns a pyarrow Table."""
    t = read_tsv(path)
    t = t.combine_chunks()
    col = lambda c: t.column(c).combine_chunks()
    nm, non_latin = normalize_names(col("business_name"))
    ad = normalize_addresses(col("business_address"))
    ad_words = _squash(_remove_words(ad, ADDR_STOP))
    nums = _squash(pc.replace_substring_regex(ad_words, r"\b[a-z][a-z0-9]*\b", " "))
    ad_alpha = _squash(pc.replace_substring_regex(ad_words, r"\b[0-9][a-z0-9]*\b", " "))
    hn = pc.struct_field(pc.extract_regex(pc.binary_join_element_wise(nums, pa.scalar(" "), " "),
                                          r"(?P<hn>\b[0-9]{2,}[a-z]?\b|^[0-9][a-z]?\b)"), [0])
    hn = pc.fill_null(hn, "")
    out = pa.table({
        "entity_id": col("entity_id"),
        "country": pc.utf8_lower(pc.utf8_trim_whitespace(col("country"))),
        "nm": nm,
        "nm_ns": pc.replace_substring(nm, " ", ""),
        "non_latin": non_latin,
        "ad": ad_words,
        "ad_alpha": ad_alpha,
        "nums": nums,
        "hn": hn,
        "raw_name": col("business_name"),
        "raw_addr": col("business_address"),
    })
    return out


def phonetic(arr):
    """Sound-based spelling of a (clean) name, so that different spellings of the same sounds match:
       'consultancy' and 'kansaltensi' (from Hindi script) both become 'knsltns'.
    Steps: unify letters that sound alike, then drop vowels (except at word start) and repeated letters."""
    s = arr
    for pat, rep in [(r"igh", "i"), (r"ph", "f"), (r"ck", "k"), (r"x", "ks"), (r"q", "k"), (r"tion", "shon"),
                     (r"c([eiy])", "s\\1"), (r"g([ei])", "j\\1"), (r"c", "k"),
                     (r"([sdtbkgj])h", "\\1"), (r"w", "v"), (r"z", "j"), (r"j\b", "s"),
                     (r"\B[aeiouy]", ""), (r"\Bh", "")]:
        s = pc.replace_substring_regex(s, pat, rep)
    for ch in "bdfgjklmnprstv":                      # collapse doubled letters: 'll' -> 'l'
        s = pc.replace_substring_regex(s, ch + ch + "+", ch)
    return s
