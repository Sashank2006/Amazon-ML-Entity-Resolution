# 1 · The problem: "Is this the same business?"

[← Back to README](../README.md) · Next: [2 · How the pipeline works →](2-how-it-works.md)

## In one sentence
Given business records from **three independent sources**, find every Source-2 and Source-3 record that describes the **same real-world business** as each Source-1 record.

This task is called **entity resolution**, also known as *record linkage* or *deduplication*.

## Why it matters
Large platforms receive data about the same seller from many places: registration forms, partner feeds, old databases. None of them share a common ID. If the platform can't tell that two records are one business, it pays the wrong party, shows duplicate listings, or misses fraud.

## The data at a glance
| | Train | Test |
|---|---|---|
| Source 1 (clean reference, no duplicates) | 2.2 M | 1.73 M |
| Source 2 + Source 3 (noisy) | 10.3 M | 9.97 M |
| Countries | US, India | US, India, **France** (never seen in training) |
| Matches per Source-1 business | 0–11 (average 3.5) | unknown |

Each record has just four fields: `entity_id`, `business_name`, `business_address`, `country`.

## What makes it hard
| Noise | Example |
|---|---|
| Spelling and legal suffixes | `Sri Balaji Traders Pvt Ltd` ↔ `SREE BALAJI TRADERS PRIVATE LIMITED` |
| Other scripts | `Consultancy Services` ↔ `कंसल्टेंसी सर्विसेज` |
| Website-style names | `Team Air Pvt Ltd` ↔ `teamair.com` |
| Character swaps | `o→0`, `l→1`, `s→5`, `é` |
| Reordered words and junk | `Society Desert Inc`, `*** Royal Packaging`, `#41961` |
| Address variations | `Rd`/`Road`, `MH`/`Maharashtra`/`महाराष्ट्र`, dropped house numbers, `N/A`, `<NULL>` |
| Empty addresses | a name and nothing else |
| **Decoys** | a *different* business at the same address, or one letter away (`Brightola` vs `Brightova`) |

## How it is scored
**Macro F0.5** is computed for each Source-1 business and then averaged.

- **Precision** is the share of predicted matches that are correct. **Recall** is the share of true matches that were found.
- **F0.5** weights precision twice as much as recall, because merging two different businesses is worse than missing a link.
- A business with no true matches scores **1.0** if you predict nothing, and **0.0** if you predict anything.

```
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

## Rules that shaped the design
- **No external lookups:** no geocoding, company registries or web data. Only the provided files.
- **Model limits:** an MIT or Apache-2.0 licence and at most 8B parameters. LightGBM is a few MB.
- **Required outputs:** two files, `matching_results.tsv` (scored) and `candidate_pairs.tsv` (audited). Smaller candidate sets were ranked higher.
- **Hardware:** everything had to run on **free Google Colab: 2 CPU cores, 12.7 GB RAM, no GPU**.
