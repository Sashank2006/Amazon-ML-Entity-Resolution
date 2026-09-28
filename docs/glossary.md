# Glossary

[Back to README](../README.md)

| Term | Plain meaning |
|---|---|
| **Entity resolution** | Deciding which records describe the same real-world thing (here, a business). |
| **Blocking** | A cheap filter that decides which pairs are worth comparing at all, so you avoid comparing everything with everything. |
| **Blocking key** | A short code two records must share to be compared, e.g. `88 + olive` (house number + street word). |
| **Block purging** | Dropping keys that too many records share, because they create millions of useless pairs. |
| **Meta-blocking** | Scoring candidate pairs by how many (and how rare) keys they share, then keeping the best few per record. |
| **Recall ceiling** | The share of true matches that survive blocking. The final model can never do better than this. |
| **Learned pruning** | Using a small ML model, instead of a fixed rule, to shrink the candidate list further. |
| **Feature** | A number that describes a pair, e.g. "name similarity = 0.87". Models only see features, never raw text. |
| **LightGBM** | A fast gradient-boosted decision-tree library (MIT licence). It builds many small trees that each fix the previous trees' mistakes. |
| **Precision** | Of the matches you predicted, the share that were right. |
| **Recall** | Of the true matches, the share you found. |
| **F0.5** | A score combining precision and recall that weights precision twice as much. |
| **Macro average** | Compute the score per business, then average, so every business counts equally. |
| **Singleton** | A Source-1 business with no matches. Predicting nothing for it scores 1.0. |
| **Validation set** | Training data held back to test the model honestly. |
| **Distribution shift** | When test data differs from training data (here: more decoys), so validation scores mislead. |
| **Transliteration** | Writing one script in another's letters: `कंसल्टेंसी` → `kansaltensi`. |
| **Phonetic code** | A spelling of how a word sounds, so different spellings match: `consultancy` → `knsltns`. |
| **Token-set ratio** | A similarity score that ignores word order and repeated words (from the RapidFuzz library). |
| **One-to-one assignment** | Each Source-2/3 record may belong to at most one Source-1 business. |
