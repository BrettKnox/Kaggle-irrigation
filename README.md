# Kaggle Playground Series S6E4 — Irrigation Need Prediction

A multiclass classification project predicting irrigation need (Low / Medium / High) for agricultural fields, using a stacked ensemble of gradient boosting and a binary specialist model.

**Final public LB score:** 0.96495 (balanced accuracy)
**OOF score:** 0.96817
**Competition:** [Kaggle Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4)

---

## What this project does

The goal: given soil, weather, crop, and field characteristics, predict whether a field needs Low, Medium, or High irrigation. Evaluated on balanced accuracy, which weights all three classes equally despite the dataset being heavily imbalanced (~59% Low, ~38% Medium, ~3% High).

I built a two-layer stacked ensemble:

**Layer 1 — Base models** (each trained with 5-fold stratified CV, class-weighted for balanced accuracy):
- **LightGBM** — fast leaf-wise gradient boosting
- **XGBoost** — depth-wise gradient boosting (different inductive bias from LGB)
- **CatBoost** — ordered boosting with native categorical handling
- **High-class binary specialist** — small LGB+CAT pair trained only on `is_High`, providing a focused probability signal for the hardest decision boundary

**Layer 2 — Combiner:**
- **Hill-climbing weighted average** on out-of-fold predictions, optimizing balanced accuracy directly
- A LightGBM stacker was tried as an alternative — interestingly, it *underperformed* hill-climbing (0.964 vs 0.968 OOF). With only ~13 stacking features and 5-fold CV, the level-2 model overfit. A reminder that more complex isn't always better.

---

## What I learned

The tactical and methodological notes that I'd want to remember for the next competition:

**On methodology:**

- **Out-of-fold predictions are everything.** Stacking, hill-climbing, ensembling, pseudo-labeling — every advanced ensembling technique relies on having honest predictions on data the model didn't train on. If you skip OOF and use train predictions, the level-2 model thinks every base model is omniscient and crashes on real test data.
- **CV-to-LB gap is diagnostic.** My OOF was 0.968, LB was 0.965 — a small, normal gap. That told me my CV was honest, no distribution shift, and I could trust local scores when deciding what to optimize. If the gap had been large in either direction, that itself would have been the signal to dig into.
- **Class weighting matters more than the metric suggests.** Balanced accuracy averages per-class recall, so the rarest class (High at 3%) influences the metric as much as the majority class. Every model gets `class_weight="balanced"` or its equivalent.

**On model selection:**

- **The specialist beat the foundation model.** I'd planned to use TabPFN v2 as a strong NN ensemble member. With ~600k training rows and TabPFN's ~10k context limit, even with subsampling and bagging it was projected to take ~9 hours per variant on a T4 — and likely contribute less than the binary specialist (which trained in 15 minutes). Tactical decision: skip TabPFN, ship the specialist. Sometimes the right call is "smaller, focused model, deployed."
- **RealMLP looked great on paper, didn't dispatch to GPU correctly.** pytabkit reported `device='cuda'` but `nvidia-smi` showed 0% GPU utilization. After 30+ minutes on fold 0 with no progress, I dropped it. **Lesson: trust `nvidia-smi`, not framework log messages.** If the GPU isn't busy, training isn't happening, regardless of what the wrapper claims.

**On engineering:**

- **OOF caching to Drive saved the project.** Colab disconnects mid-run. Without checkpointing, every disconnect would have meant restarting from scratch. With per-model cached `.npy` files on Drive, a disconnect cost at most one model. This pattern is non-negotiable for any long ML run on ephemeral compute.
- **GPU memory leaks after interrupting.** Killing TabPFN mid-run left ~6GB of CUDA memory allocated to nothing. `torch.cuda.empty_cache()` + `gc.collect()` reclaims it; otherwise the next model gets warnings about insufficient memory. Worth knowing before you reach for "Restart runtime."
- **Don't put API keys in notebook source.** I made this mistake once during this project, rotated the key immediately. Colab Secrets is the right pattern; the 60 seconds it takes to set up is worth it.

---

## What I'd do differently next time

In rough order of expected payoff if I had another week:

1. **Optuna-tune CatBoost and LightGBM.** My base models use sensible-but-untuned defaults. The leaderboard top has tuned base models at ~0.98 individually. Closing 80% of the CV gap is probably just hyperparameter optimization, not novel architecture.
2. **Get RealMLP working with explicit GPU verification.** Check `nvidia-smi` mid-run, not framework reports. If util > 50%, let it cook. If util = 0%, kill and use vanilla PyTorch instead. RealMLP is the architecture currently dominating tabular leaderboards; getting it in the ensemble is a real differentiator.
3. **Add a Medium-class specialist.** I built a High-vs-not specialist; the Medium↔High boundary cuts both directions. A `is_Medium` specialist would provide a complementary signal.
4. **Pseudo-labeling.** With a strong ensemble, pseudo-labeling the most confident test predictions and adding them to training typically gains 0.001-0.003 on Playground series.
5. **Threshold/bias optimization.** For balanced accuracy specifically, additive per-class biases on logits, optimized via Nelder-Mead on OOF, sometimes wins 0.0005 for free.

---

## Project structure

```
.
├── README.md                          # this file
├── requirements.txt                   # pinned dependencies
├── ps_s6e4_irrigation_v2.ipynb       # main notebook (annotated, runs end-to-end)
├── output/
│   ├── submission.csv                 # final LB-scored predictions
│   └── oof/                           # cached OOF + test predictions per model
│       ├── oof_lgb.npy
│       ├── oof_xgb.npy
│       ├── oof_cat.npy
│       ├── oof_specialist.npy
│       ├── test_lgb.npy
│       ├── test_xgb.npy
│       ├── test_cat.npy
│       └── test_specialist.npy
└── results/
    └── model_comparison.md            # per-model CV scores and analysis
```

---

## How to reproduce

**Requirements:**
- Python 3.10+
- A GPU helps (a T4 is sufficient; the project was developed on Colab Pro)
- A Kaggle account with API credentials

**Setup:**

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

**Get the data.** Either:

(a) Use `kagglehub` from inside the notebook (requires `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables). The notebook downloads and caches automatically.

(b) Download manually from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e4/data) (you must accept rules first) and place the CSVs in `data/`. Update the `COMP_DIR` path in the config cell.

**Run the notebook:**

Open `ps_s6e4_irrigation_v2.ipynb`. The config cell has feature flags (`RUN_LGB`, `RUN_XGB`, etc.) — turn off any models you want to skip. Cached OOFs are loaded automatically; only models without cache files are retrained.

Full pipeline runtime on a T4 with all models enabled: ~3-4 hours. Specialist + stacker only: ~30 minutes.

---

## Results summary

| Model | OOF Balanced Accuracy |
|---|---|
| LightGBM | ~0.962 |
| XGBoost | ~0.961 |
| CatBoost | ~0.964 |
| High Specialist (AUC) | 0.996 |
| **Hill-climb ensemble** | **0.96817** |
| Stacker (overfit) | 0.964 |
| **Public LB** | **0.96495** |

The specialist's AUC of 0.996 on the binary High-vs-not task is high enough that its probability signal genuinely helps the stacker on borderline cases — even though the stacker itself didn't beat hill-climbing.

---

## Honest framing

This project did not place at the top of the leaderboard. The winning ensembles are at ~0.981; I'm at ~0.965. That gap reflects (a) untuned base models, (b) no working neural net contribution, and (c) the Playground Series dynamic where top scores increasingly come from public-ensemble blending of community submissions.

What I aimed for instead: a clean, reproducible end-to-end pipeline that demonstrates real ML engineering — proper CV methodology, sensible ensembling, honest diagnostics, and tactical decisions under time and compute constraints. The notebook is annotated to be readable as a learning resource, not just executable.
