# 4 · Run it yourself

[← 3 · Results and lessons](3-results-and-lessons.md) · [Back to README](../README.md)

## Option A: 2-minute demo in Colab (no setup)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sashank2006/Amazon-ML-Entity-Resolution/blob/main/notebooks/demo_colab.ipynb)

Click the badge, then choose **Runtime → Run all**. The notebook creates a fake dataset with realistic noise, runs the whole pipeline and prints the score plus example matches.

## Option B: demo on your own machine
```bash
git clone https://github.com/Sashank2006/Amazon-ML-Entity-Resolution.git
cd Amazon-ML-Entity-Resolution
pip install -r requirements.txt

python demo/make_demo_data.py --out demo_data            # 6,000 + 3,000 fake businesses
python src/pipeline.py all --data demo_data --work demo_cache --model-dir demo_model --out demo_output
python demo/show_results.py --data demo_data --out demo_output
```
Expected output (about 20 seconds):
```
Macro F0.5 on the demo test set : 0.97
Pair precision / recall        : 0.994 / 0.943
Candidates per business        : 3.34   (recall ceiling 0.981)
```

## Option C: the real challenge data
The competition data isn't public. If you have it, point `--data` at the folder that contains `train/` and `test/`:
```bash
python src/pipeline.py prep    --data dataset --work cache --model-dir model --out output   # ~20 min
python src/pipeline.py block   --data dataset --work cache --model-dir model --out output   # ~35 min
python src/pipeline.py train   --data dataset --work cache --model-dir model --out output   # ~35 min
python src/pipeline.py predict --data dataset --work cache --model-dir model --out output   # ~40 min
```
Times are for free Colab (2 vCPU, 12.7 GB RAM). [`notebooks/full_pipeline_colab.ipynb`](../notebooks/full_pipeline_colab.ipynb) is the exact notebook used during the challenge.

### Changing settings without editing code
```bash
python src/pipeline.py train ... --set top_k=30 --set n_train_s1=900000
```
| Setting | Default | Meaning |
|---|---|---|
| `top_k` | 30 | blocking candidates kept per business |
| `max_s1_per_key` / `max_s23_per_key` | 15 / 150 | purge keys shared by more records than this |
| `prune_keep_recall` | 0.995 | share of reachable true pairs the pruning stage must keep |
| `max_cand_per_s1` | 10 | hard cap after pruning |
| `n_train_s1` / `n_val_s1` | 300k / 100k | businesses used for training / validation |
| `extra_thresholds` | `0.80,0.85,0.90,0.95` | extra stricter output files written by `predict` |

### Outputs
| File | Content |
|---|---|
| `output/matching_results.tsv` | one row per Source-1 business: `source1_entity_id <TAB> matched ids` (comma-separated, may be empty) |
| `output/candidate_pairs.tsv` | the pruned candidate list the model scored |
| `model/run_log.txt` | every log line: blocking recall, validation F0.5, feature importance |
| `model/val_errors.txt` | examples of wrong merges, missed matches and pairs lost by blocking |

## The interactive page
[`web/index.html`](../web/index.html) is a single self-contained HTML file. Open it in a browser, or view it live on [GitHub Pages](https://sashank2006.github.io/Amazon-ML-Entity-Resolution/web/).

It uses a transparent 8-feature logistic model so that every step is visible. [`web/build/demo_model.py`](../web/build/demo_model.py) is its Python reference and trains it on the demo data:
```bash
python web/build/demo_model.py --data demo_data --cache demo_cache   # writes web/build/model.json
```
