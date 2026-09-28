# Amazon ML Challenge 2026: Business Entity Resolution (Methodology)

[← Back to README](../README.md)

**Participant:** Sashank S A (solo), St. Joseph's College of Engineering and Technology  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We built a three-step pipeline: **token blocking → learned pruning → matching model**. Blocking uses 11 types of rare-token keys, including phonetic keys for transliterated Indian-script names. A small LightGBM "pruning" model then cuts the candidate set from 29 to **4.5 records per Source-1 entity** while keeping 99.5% of the reachable true matches. A second LightGBM model makes the final decision, with a one-to-one constraint (each Source-2/3 record belongs to at most one Source-1 entity). The best validation macro F0.5 is **0.9496**, measured on 100,000 held-out training entities. Our final leaderboard submission keeps only the pairs on which two pipeline versions agree, and it scores **0.935** on the public leaderboard. The approach uses only the provided data, runs on a 2-core, 12 GB CPU machine, and uses only MIT-licensed libraries.

---

## 2. Methodology

### 2.1 Problem Analysis
Scale: train has 2.2M S1 and 10.3M S2+S3 records; test has 1.73M S1 and 9.97M S2+S3 records. Train covers US and India; test adds France (15% of test S1).

Findings from exploring the data:
- **Most entities have matches.** Only 5.6% of S1 entities are singletons. The mean is 3.46 matches per entity and the maximum is 11. Recall therefore matters as much as precision.
- **Strict one-to-one structure.** In the training labels, no S2/S3 record is matched to more than one S1 entity (0 of 7.64M pairs). This is consistent with S1 being deduplicated.
- **Names are the weakest field.** They show typos, character swaps (0/o, 1/l, 5/s), accents, reordered words, legal-suffix changes (Pvt Ltd / Private Limited, LLC / L.L.C.) and junk tokens ("Services", "Center", "#41961", "Mr"). Some names are written as domains (`teamair.com`). Some are replaced entirely by a trade or DBA name. About 7–9% of S2/S3 names are in Indian scripts (Devanagari, Telugu, Tamil, Bengali, Kannada, …).
- **Addresses are the most reliable field, but noisy too:** Rd/Road-style abbreviations, state names vs codes vs native-script state names, reordered components, dropped house numbers, noise prefixes (`#`, `##`, `No`, `H.no`, `1-`), leading zeros (`0044023`), placeholder strings (`N/A`, `<NULL>`, `null`), PMB/Unit additions, and some records with an empty address.
- **Deliberate hard negatives.** Different businesses sometimes have near-identical names at the same address (e.g. "Brightola Staffing" vs "Brightova Staffing"). House numbers differ slightly in both true and false pairs.

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned pruning (meta-blocking) + gradient-boosted classifier + one-to-one assignment  
**Core Innovation:** A two-stage LightGBM cascade. Stage 1 is a cheap model that works as *learned blocking*, shrinking candidates to 4.5 per entity. Its probabilities are then turned into **competition features** for stage 2, such as "does another S1 entity explain this S2/S3 record better?" and "how does this pair rank among this entity's candidates?". These were the strongest features in the final model.

---

## 3. Candidate Generation (Blocking)

**Text normalisation** (vectorised with Apache Arrow):
- lower-casing, accent removal (NFKD), transliteration of Indian scripts to Latin (`indic-transliteration`, MIT) plus phonetic clean-up
- removing legal suffixes and noise words
- canonical address abbreviations (English and French: rue, bd, av, allée, …)
- dropping state/region segments, splitting digits from glued words, stripping leading zeros

**Blocking keys** (within the same country only):

| Key | Example | Purpose |
|---|---|---|
| name word | `davis` | basic name match |
| name word pair | `davis\|family` | makes common words rarer |
| name without spaces | `davisfamilyoffice` | domain-style names |
| name-prefix pair | `davi\|fami` | robust to typos |
| name word + number | `davis\|88` | name plus house number |
| number + address word | `88\|olive` | works when the name is unusable |
| number pair | `12\|190` | Indian plot/door numbers |
| name prefix + address prefix | `davi\|oliv` | addresses without numbers |
| phonetic word / pair / word+number | `knsltns` | "Consultancy" ≈ "kansaltensi" (Hindi script) |

**How the candidate set is kept small:**
1. **Block purging.** Keys shared by more than 15 S1 or 150 S2/S3 records are dropped.
2. **Weighted meta-blocking.** Each shared key adds weight 1/log2(2 + block size), multiplied by a type weight. The top 30 partners per S1 entity are kept.
3. **Learned pruning (stage 1).** A LightGBM model uses blocking scores, rank features and 4 fast similarities. Its threshold is chosen on validation so that 99.5% of the true pairs available after blocking are kept, with at most 10 candidates per entity.

Memory stays bounded because S1 keys are built first, S2/S3 are processed in chunks, and only keys that also occur in S1 are kept. All work uses integer arrays.

- **Candidate pairs generated (test):** 7,723,118 in `candidate_pairs.tsv`, i.e. **4.46 per S1 entity** (France 4.79, India 4.14, US 4.71). The raw blocking output before pruning was 50.4M (29.1 per entity), out of a possible 1.73M × 9.97M ≈ 1.7 × 10¹³ pairs. That is a reduction ratio above 99.9999%.
- **Recall ceiling (train, share of all true pairs):** 0.9358 after keyed blocking, **0.9314 after pruning** (US 0.956, India 0.895).
- **How we avoided losing true matches:** we used key types that do not depend on the name (number × address word) and ones that survive name corruption (prefix and phonetic keys). The pruning threshold is set by a recall constraint rather than a fixed K.

---

## 4. Matching Model

**Features used (~50 per pair):**
- **Name features:** rapidfuzz ratio, token-sort, token-set, partial ratio, Jaro-Winkler; ratio and partial ratio on the name without spaces; exact-equality flag; phonetic-name ratio and token-set; name lengths; non-Latin-script flag.
- **Address features:** ratio, token-set, token-sort and partial ratio on the normalised address; token-set and partial ratio on the street-only part (numbers removed); token-set and token-sort on the number set; house-number equality and similarity; address lengths; number counts.
- **Frequency features:** how many S1 and S2/S3 records share this exact name or this exact address. A unique exact name is strong evidence; a busy building is weak evidence.
- **Blocking and context features:** blocking score, number of shared keys (name vs address keys), rank of the pair within its S1 entity and within its S2/S3 record, and score relative to the best candidate.
- **Stage-1 competition features:** stage-1 probability, its rank and relative value within the S1 entity, the best competing S1 probability for the same S2/S3 record, and its rank within that record.
- **Other:** source flag (S2 vs S3). Country is deliberately **not** used as a feature, so that France, which is absent from training, is handled like any other country.

**Model type:** LightGBM binary classifier (MIT licence; well under 8B parameters, a few MB). Stage 1 has 171 trees and 63 leaves. Stage 2 has 812 trees, 127 leaves and learning rate 0.08, with early stopping on a separate set of entities.

**Training split (by S1 entity, seed 42):** 150k entities for stage 1; 270k for stage-2 fitting; 30k for early stopping; 100k for validation (never used in training).

**Post-processing:**
1. **One-to-one assignment.** Each S2/S3 record is kept only for the S1 entity where its probability is highest.
2. **Threshold.** A pair is accepted if its probability is at least 0.675.

**Threshold selection method:** a grid search of the probability threshold (and an optional relative-to-best-candidate rule) that maximises **macro F0.5 on the validation entities**, which selected 0.675. The metric is implemented exactly as specified: singletons score 1.0 or 0.0, and true pairs lost in blocking count as missed.

**What the leaderboard taught us about thresholds:** the test set has more Source-2/3 records per entity than training (5.8 vs 4.7), and therefore more look-alike distractors. The validation-optimal threshold of 0.675 over-predicted on test: 3.44 matches per entity against about 3.16 expected. Precision mattered more on test than validation suggested. We tested stricter settings on the leaderboard (a single model at threshold 0.95, and the agreement of two models). **Agreement between two differently built models** worked best: it removes pairs that only one model believes in. The final submission keeps a pair only if both v1 (single-stage) and v4 (two-stage, this code) predict it.

---

## 5. Results & Error Analysis

| Version | Change | Candidates / S1 | Validation F0.5 | Leaderboard |
|---|---|---|---|---|
| v1 | Blocking (8 key types) + one LightGBM | 24.3 | 0.9398 | 0.931 |
| v4 | + learned pruning, competition features, frequency features, phonetic keys | **4.46** | **0.9496** | 0.903 |
| v5 | v4 model, stricter threshold 0.95 | 4.46 | — | 0.922 |
| v1 ∩ v4 at 0.80 | agreement with v4 at a stricter threshold | 4.46 | — | 0.934 |
| **v1 ∩ v4 (final)** | **keep only pairs predicted by both v1 and v4** | **4.46** | — | **0.935** |

- **F0.5 score (macro, validation):** **0.9496** (US 0.9650, India 0.9262). Pair-level precision is 0.991 and recall 0.900. Singletons are correctly left empty 96.4% of the time.
- **Common false positives (wrong merges):**
  - deliberate near-duplicate businesses at the same address with a one-letter name change ("Brightola" vs "Brightova Staffing Inc.", 359 Kurzon Rd)
  - the same name with a slightly different house number ("Sterling Fitness LLC", 1610 vs 161 Canal Dr)
  - a record with a garbage or replaced name at an identical address
- **Common false negatives (missed matches):**
  - S2/S3 records with an **empty address** and a common name
  - trade/DBA names unrelated to the legal name ("Evowex" for "Noida Spinners Pvt Ltd")
  - Indian-script names whose address was shortened to "number, city"
  - house-number noise in true matches (7132 vs 7141)
- **Validation vs leaderboard:** validation tracked the leaderboard well for v1 (0.940 vs 0.931), but overestimated v4 (0.950 vs 0.903). The pairs v4 added on top of v1 were mostly correct on training data but mostly wrong on test. This distribution shift was the main lesson of the challenge.
- **Main remaining gap:** blocking recall for India (0.895 vs 0.956 for the US). Short addresses combined with transliterated names give few rare keys.

---

## 6. Conclusion
A scalable cascade of rare-token blocking, learned pruning and a LightGBM matcher resolves 1.7M test entities against 10M records on a free 2-core CPU machine in about 1.5 hours. The final candidate set is only 4.5 records per entity, and the final submission (agreement of two model versions) scores 0.935 on the public leaderboard. The largest gains came from treating pruning as a learned model and from **competition features**, which exploit the one-to-one structure of the data. The largest remaining opportunity is recall for short-address and non-Latin Indian records.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/normalize.py`: reading TSVs (Apache Arrow), text normalisation, transliteration, phonetic encoding
- `src/blocking.py`: key generation, block purging, weighted top-K meta-blocking
- `src/features.py`: pair features, context and competition features, frequency features
- `src/metrics.py`: macro F0.5 exactly as defined by the challenge
- `src/pipeline.py`: command-line entry point with stages `prep → block → train → predict` (or `all`)
- `run_in_colab.ipynb`: the exact notebook used; `README.md`; `requirements.txt` (pinned versions)

To reproduce, run from `student_resource/`:
```
python code/business_entity_resolution/src/pipeline.py all --data dataset --work cache --model-dir model --out output
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. All randomness is seeded.

**Final submission file (as submitted):** the `output/matching_results.tsv` in this package keeps only the pairs predicted by *both* pipeline versions: v1 (a single-stage LightGBM over the top-25 blocking candidates, leaderboard 0.931) and v4 (this code, leaderboard 0.903). The intersection scored 0.935. `output/candidate_pairs.tsv` comes from this code's `predict` stage, and every final match is contained in it.

**Environment:** Google Colab (2 vCPU, 12.7 GB RAM, no GPU), Python 3.13, numpy 2.1.3, pandas 2.2.3, pyarrow 23.0.1, rapidfuzz 3.14.6, lightgbm 4.6.0, indic-transliteration 2.3.82. Approximate runtime: prep 21 min, blocking 35 min, train 36 min, predict 38 min.

**Data use:** only the provided files. No external databases, APIs, geocoding or lookups. The only hand-written lists are general language knowledge: address abbreviations, legal suffixes, and state/region names used to *remove* those segments.

### B. Additional Results
Stage-1 pruning on train: 63.8M → 8.2M pairs (28.9 → 3.72 per entity), with the recall ceiling going from 0.9358 to 0.9314.
