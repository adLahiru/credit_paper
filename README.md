# 3-Hour German Credit Conference Paper — Run Guide

## Timeline

| Time | Phase | Action |
|---|---|---|
| 0:00–0:10 | Setup | Install deps |
| 0:10–0:25 | Run experiment | `python run_experiment.py` (~10-15 min) |
| 0:25–0:30 | Auto-fill paper | `python fill_paper.py` |
| 0:30–0:50 | Overleaf upload + compile | 20 min |
| 0:50–2:30 | Review & polish | 90 min |
| 2:30–3:00 | Proofread + submit | 30 min |

## Files You Have

| File | What it does |
|---|---|
| `run_experiment.py` | 9 classifiers with grid-search tuning, fairness, interpretability, 5 figures |
| `fill_paper.py` | Auto-fills the paper template from results |
| `paper.tex` | IEEE 8-page conference template (German Credit study) |

## Phase 1 — Setup (10 min)

```bash
mkdir ~/credit_paper && cd ~/credit_paper
# Drop the 3 files above into this folder

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib lightgbm
```

Verify:

```bash
python -c "from sklearn.ensemble import RandomForestClassifier; from lightgbm import LGBMClassifier; print('OK')"
```

## Phase 2 — Run Experiment (10-15 min)

```bash
python run_experiment.py
```

The script will:
1. Download German Credit from UCI automatically
2. Train + tune 9 classifiers with 5-fold CV
3. Compute fairness metrics (DP, EO, DI)
4. Compute feature importances (RF) and coefficients (LogReg)
5. Generate 5 publication-quality figures
6. Produce `results/stats.json` with all numbers

If UCI URL fails (rare), the script prints clear error. Manual download backup:

```bash
# If automatic download fails:
mkdir -p data
curl -L -o data/german.data "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
# Then rerun python run_experiment.py
```

## Phase 3 — Auto-fill Paper (1 min)

```bash
python fill_paper.py
```

This creates `paper_filled.tex` with all numbers, tables, and prose populated from your results.

If it says "All markers replaced successfully" you're ready.

## Phase 4 — Overleaf (20 min)

1. **overleaf.com** → sign up (free)
2. New Project → Blank Project → name `credit-paper`
3. Delete default `main.tex`
4. Upload `paper_filled.tex`
5. Set `paper_filled.tex` as main document (Menu → Settings → Main document)
6. Create `figures` folder in Overleaf (New Folder button in file tree)
7. Upload all 5 PDFs from local `figures/` into the Overleaf `figures` folder:
   - `fig1_model_comparison.pdf`
   - `fig2_accuracy_fairness.pdf`
   - `fig3_cost_sensitive.pdf`
   - `fig4_feature_importance.pdf`
   - `fig5_gender_disparity.pdf`
8. Click **Recompile**

If you see compile errors, most likely a figure file is missing — check the `figures/` folder contents.

## Phase 5 — Review & Polish (90 min)

Read the paper end-to-end. Things to check:

**Page count.** Currently sized for ~6-7 pages. If you need exactly 8 pages per IEEE requirement:
- Add a paragraph under each Results subsection expanding on one observation
- Expand the Related Work subsections (cite 1-2 more references with 2-3 sentences each)
- Expand Limitations with more specific items
- Add one paragraph of per-feature interpretation in Section IV.D

**Numbers sanity check.** Open `results/main_results.csv` and spot-check 2-3 values against Table I in the paper. If the best AUC is around 0.75-0.80 you're seeing real UCI data (my smoke test showed 0.67 because it used synthetic data).

**Author info.** Currently says "Lahiru Dilshan, University of Moratuwa" — verify correct and add student ID if required.

**References.** 14 references currently. Check your instructor's style guide — some require 15-20 references. If you need more, cite 2-3 of: Hastie/Tibshirani textbook, Elkan's cost-sensitive learning paper, a recent ethics-in-ML survey.

**GitHub link.** Auto-filled as placeholder URL. If you push to a real repo, update footnote in Section I. Otherwise change to "code available upon request".

## Phase 6 — Submit (30 min)

- Spell check in Overleaf
- Check all `\ref{}` resolve (no `??` in PDF)
- Check figures render with captions
- Download PDF
- Submit

---

## Troubleshooting

**UCI download fails** → Check firewall / VPN. Alternative via Kaggle mirror:
```python
# Edit run_experiment.py, in load_german_credit() replace URL with:
# https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv
# (Note: different columns; would need schema fix)
```

**LightGBM install fails on Mac** → `brew install libomp` then `pip install lightgbm`. Or just skip — script auto-detects and runs with 8 classifiers.

**Grid search too slow** → Open `run_experiment.py`, find `get_classifiers_and_grids()`, reduce each `param_grid` to 2 values per hyperparameter. Cuts runtime in half.

**Paper is only 5 pages, not 8** → That's fine; the IEEE requirement often says "up to" rather than "exactly 8". If instructor insists on 8:
- Add subsection IV.E "Discussion of Individual Classifiers" with one paragraph per model family
- Add a full Appendix A with the complete hyperparameter grids as a table
- Expand Related Work from 4 subsections to 5-6 by splitting "Classical ML" into "pre-2020" and "post-2020"

**LightGBM in Table I but shouldn't be** → If you want to exclude it (for strict "classical only" reading), uncomment the `pip install` step and it won't be loaded. The script auto-removes it from all tables/figures in that case.
