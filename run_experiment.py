"""
German Credit Risk Prediction: Accuracy-Fairness-Interpretability Analysis
Conference-quality 3-hour version.

REQUIREMENTS:
    pip install numpy pandas scipy scikit-learn matplotlib lightgbm

Expected runtime: 5-15 minutes on any laptop (dataset is small: 1000 rows).
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import json
import warnings
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score,
                             brier_score_loss)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("NOTE: lightgbm not installed - skipping. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')
RNG = 42
np.random.seed(RNG)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ======================================================================
# 1. DATA LOADING  (UCI Statlog German Credit Data)
# ======================================================================

GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMNS = [
    'checking_status',       # A11..A14 qualitative
    'duration_months',       # numeric
    'credit_history',        # A30..A34 qualitative
    'purpose',               # A40..A410 qualitative
    'credit_amount',         # numeric
    'savings_status',        # A61..A65 qualitative
    'employment_since',      # A71..A75 qualitative
    'installment_rate_pct',  # numeric (1-4)
    'personal_status_sex',   # A91..A95 qualitative - contains gender
    'other_debtors',         # A101..A103 qualitative
    'residence_since',       # numeric
    'property',              # A121..A124 qualitative
    'age_years',             # numeric
    'other_installment_plans',  # A141..A143
    'housing',               # A151..A153
    'existing_credits',      # numeric
    'job',                   # A171..A174
    'dependents',            # numeric
    'telephone',             # A191..A192
    'foreign_worker',        # A201..A202
    'target',                # 1=good, 2=bad
]

NUMERIC_COLS = ['duration_months', 'credit_amount', 'installment_rate_pct',
                'residence_since', 'age_years', 'existing_credits', 'dependents']
CATEGORICAL_COLS = [c for c in COLUMNS if c not in NUMERIC_COLS and c != 'target']


def load_german_credit():
    """Download + parse German Credit dataset from UCI."""
    cache = Path("data/german.data")
    cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        print("Downloading German Credit dataset from UCI...")
        try:
            urllib.request.urlretrieve(GERMAN_URL, cache)
        except Exception as e:
            print(f"  Direct UCI fetch failed ({e}); trying mirror...")
            # Mirror on github
            mirror = "https://raw.githubusercontent.com/uci-ml-repo/statlog-german-credit/main/german.data"
            try:
                urllib.request.urlretrieve(mirror, cache)
            except Exception as e2:
                raise RuntimeError(f"Could not download dataset. Check internet. {e2}")
    df = pd.read_csv(cache, sep=r'\s+', header=None, names=COLUMNS)
    # Recode target: 1 (good) -> 0, 2 (bad) -> 1 (so "bad credit" is the positive class)
    df['target'] = (df['target'] == 2).astype(int)
    # Extract gender from personal_status_sex (A91,A93,A94 = male; A92,A95 = female)
    df['gender'] = df['personal_status_sex'].map({
        'A91': 'male', 'A92': 'female', 'A93': 'male',
        'A94': 'male', 'A95': 'female'
    })
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]-1} features")
    print(f"  Class balance: good={((df.target==0).sum())}, bad={((df.target==1).sum())}")
    print(f"  Gender: male={((df.gender=='male').sum())}, female={((df.gender=='female').sum())}")
    return df


# ======================================================================
# 2. PREPROCESSING PIPELINE
# ======================================================================

def make_preprocessor():
    """StandardScaler for numeric + OneHotEncoder for categorical."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
        ], remainder='drop'
    )


# ======================================================================
# 3. CLASSIFIERS  (with hyperparameter grids for CV tuning)
# ======================================================================

def get_classifiers_and_grids():
    """Return dict: name -> (classifier, param_grid). Grids are small for speed."""
    base = {
        "LogReg": (
            LogisticRegression(max_iter=2000, random_state=RNG),
            {'clf__C': [0.01, 0.1, 1.0, 10.0], 'clf__class_weight': [None, 'balanced']},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RNG),
            {'clf__max_depth': [3, 5, 7, 10, None], 'clf__class_weight': [None, 'balanced']},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {'clf__n_neighbors': [3, 5, 7, 11, 15], 'clf__weights': ['uniform', 'distance']},
        ),
        "LinearSVM": (
            SVC(kernel='linear', probability=True, random_state=RNG),
            {'clf__C': [0.1, 1.0, 10.0], 'clf__class_weight': [None, 'balanced']},
        ),
        "RBFSVM": (
            SVC(kernel='rbf', probability=True, random_state=RNG),
            {'clf__C': [0.1, 1.0, 10.0], 'clf__gamma': ['scale', 'auto']},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, random_state=RNG, n_jobs=-1),
            {'clf__max_depth': [5, 10, None], 'clf__class_weight': [None, 'balanced']},
        ),
        "GradBoost": (
            GradientBoostingClassifier(random_state=RNG),
            {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5], 'clf__learning_rate': [0.05, 0.1]},
        ),
        "MLP": (
            MLPClassifier(max_iter=500, random_state=RNG),
            {'clf__hidden_layer_sizes': [(32,), (64,), (32, 16)], 'clf__alpha': [0.001, 0.01]},
        ),
    }
    if HAS_LGBM:
        base["LightGBM"] = (
            LGBMClassifier(random_state=RNG, verbose=-1, n_jobs=-1),
            {'clf__n_estimators': [100, 200], 'clf__num_leaves': [15, 31], 'clf__learning_rate': [0.05, 0.1]},
        )
    return base


# ======================================================================
# 4. COST-SENSITIVE EVALUATION  (per UCI docs: FN costs 5x more than FP)
# ======================================================================

def cost_matrix_score(y_true, y_pred, cost_fn=5, cost_fp=1):
    """German Credit uses a cost matrix: misclassifying bad as good is 5x worse."""
    cm = confusion_matrix(y_true, y_pred)
    # cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP   (positive class = bad credit)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total_cost = cost_fn * fn + cost_fp * fp
    return int(total_cost)


# ======================================================================
# 5. FAIRNESS METRICS
# ======================================================================

def demographic_parity_difference(y_pred, group):
    """P(Y_hat=1 | group=A) - P(Y_hat=1 | group=B). Closer to 0 = more fair."""
    groups = sorted(set(group))
    if len(groups) != 2:
        return float('nan')
    rates = {g: (y_pred[np.array(group) == g]).mean() for g in groups}
    return float(rates[groups[0]] - rates[groups[1]])


def equal_opportunity_difference(y_true, y_pred, group):
    """TPR_A - TPR_B. Closer to 0 = more fair."""
    groups = sorted(set(group))
    if len(groups) != 2:
        return float('nan')
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    tprs = {}
    for g in groups:
        mask = (group == g) & (y_true == 1)
        if mask.sum() == 0:
            tprs[g] = float('nan')
        else:
            tprs[g] = (y_pred[mask] == 1).mean()
    return float(tprs[groups[0]] - tprs[groups[1]])


def disparate_impact_ratio(y_pred, group):
    """min/max of P(Y_hat=1 | group). 1.0 = perfectly fair. <0.8 = concerning per 4/5 rule."""
    groups = sorted(set(group))
    if len(groups) != 2:
        return float('nan')
    rates = {g: (np.asarray(y_pred)[np.array(group) == g]).mean() for g in groups}
    if max(rates.values()) == 0:
        return float('nan')
    return float(min(rates.values()) / max(rates.values()))


# ======================================================================
# 6. MAIN EXPERIMENT
# ======================================================================

def main():
    print("=" * 70)
    print("German Credit Risk: Accuracy-Fairness-Interpretability Analysis")
    print("=" * 70)

    df = load_german_credit()

    # Target and features
    y = df['target'].values
    gender = df['gender'].values
    # Drop target + the raw gender/personal_status column (we'll use gender as protected attribute)
    X = df.drop(columns=['target', 'gender'])

    # Train/test split with stratification
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, gender, test_size=0.3, stratify=y, random_state=RNG
    )
    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")

    # ------------------ Baseline (majority class) ------------------
    majority = int(pd.Series(y_train).mode()[0])
    base_pred = np.full(len(y_test), majority)
    baseline_acc = accuracy_score(y_test, base_pred)
    print(f"  Majority-class baseline accuracy: {baseline_acc:.4f}")

    # ------------------ Train & evaluate each classifier ------------------
    classifiers = get_classifiers_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    results = []
    all_preds = {}
    all_probs = {}

    print(f"\n=== Tuning & evaluating {len(classifiers)} classifiers ===")
    for name, (clf, grid) in classifiers.items():
        print(f"\n[{name}]")
        t0 = time.perf_counter()

        pipeline = Pipeline([
            ('prep', make_preprocessor()),
            ('clf', clf),
        ])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring='roc_auc',
                              n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        all_preds[name] = y_pred
        all_probs[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1)
        brier = brier_score_loss(y_test, y_prob)
        cost = cost_matrix_score(y_test, y_pred)

        # Fairness
        dp_diff = demographic_parity_difference(y_pred, g_test)
        eo_diff = equal_opportunity_difference(y_test, y_pred, g_test)
        di_ratio = disparate_impact_ratio(y_pred, g_test)

        print(f"  acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  cost={cost}  "
              f"DP_diff={dp_diff:+.3f}  EO_diff={eo_diff:+.3f}  DI={di_ratio:.3f}  "
              f"train={train_time:.1f}s")

        results.append({
            'classifier': name,
            'accuracy': acc, 'auc': auc, 'f1': f1, 'precision': prec,
            'recall': rec, 'brier': brier, 'cost_matrix_score': cost,
            'cv_best_auc': float(search.best_score_),
            'demographic_parity_diff': dp_diff,
            'equal_opportunity_diff': eo_diff,
            'disparate_impact_ratio': di_ratio,
            'train_time_s': train_time,
            'best_params': str(search.best_params_),
        })

    df_res = pd.DataFrame(results).sort_values('auc', ascending=False).reset_index(drop=True)
    df_res.to_csv(RESULTS_DIR / "main_results.csv", index=False)

    # ------------------ Interpretability: feature importance ------------------
    print("\n=== Feature Importance (Random Forest) ===")
    rf_pipe = Pipeline([('prep', make_preprocessor()),
                        ('clf', RandomForestClassifier(n_estimators=300, max_depth=10,
                                                       random_state=RNG, n_jobs=-1))])
    rf_pipe.fit(X_train, y_train)
    ohe = rf_pipe.named_steps['prep'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feat_names = NUMERIC_COLS + cat_names
    importances = rf_pipe.named_steps['clf'].feature_importances_
    fi = pd.DataFrame({'feature': feat_names, 'importance': importances})\
           .sort_values('importance', ascending=False).reset_index(drop=True)
    fi.to_csv(RESULTS_DIR / "feature_importance_rf.csv", index=False)
    print(fi.head(10).to_string(index=False))

    # ------------------ Interpretability: logistic regression coefficients ------------------
    print("\n=== Logistic Regression Coefficients (interpretable model) ===")
    lr_pipe = Pipeline([('prep', make_preprocessor()),
                        ('clf', LogisticRegression(max_iter=2000, C=1.0,
                                                   class_weight='balanced', random_state=RNG))])
    lr_pipe.fit(X_train, y_train)
    coefs = lr_pipe.named_steps['clf'].coef_.ravel()
    lr_df = pd.DataFrame({'feature': feat_names, 'coef': coefs,
                          'abs_coef': np.abs(coefs)})\
              .sort_values('abs_coef', ascending=False).reset_index(drop=True)
    lr_df.to_csv(RESULTS_DIR / "logreg_coefficients.csv", index=False)
    print(lr_df[['feature', 'coef']].head(10).to_string(index=False))

    # ------------------ Fairness deep-dive for best AUC model ------------------
    best_name = df_res.iloc[0]['classifier']
    best_pred = all_preds[best_name]
    print(f"\n=== Fairness deep-dive for best model: {best_name} ===")
    male_mask = g_test == 'male'
    female_mask = g_test == 'female'
    fairness_detail = {
        'best_model': best_name,
        'n_male': int(male_mask.sum()),
        'n_female': int(female_mask.sum()),
        'male_approval_rate': float(1 - best_pred[male_mask].mean()),
        'female_approval_rate': float(1 - best_pred[female_mask].mean()),
        'male_accuracy': float(accuracy_score(y_test[male_mask], best_pred[male_mask])),
        'female_accuracy': float(accuracy_score(y_test[female_mask], best_pred[female_mask])),
        'male_tpr': float(recall_score(y_test[male_mask], best_pred[male_mask], zero_division=0)),
        'female_tpr': float(recall_score(y_test[female_mask], best_pred[female_mask], zero_division=0)),
    }
    with open(RESULTS_DIR / "fairness_detail.json", "w") as f:
        json.dump(fairness_detail, f, indent=2)
    print(json.dumps(fairness_detail, indent=2))

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n=== Generating figures ===")
    plt.rcParams.update({'font.size': 10, 'savefig.bbox': 'tight', 'savefig.dpi': 300})

    # Figure 1: Model comparison bar chart (Accuracy, AUC, F1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df_res))
    w = 0.25
    ax.bar(x - w, df_res['accuracy'], w, label='Accuracy', color='#1f77b4')
    ax.bar(x, df_res['auc'], w, label='AUC', color='#2ca02c')
    ax.bar(x + w, df_res['f1'], w, label='F1 (bad class)', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['classifier'], rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.set_title('Classifier comparison on German Credit (held-out test set, 30% split)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(baseline_acc, ls='--', color='gray', alpha=0.7, label=f'Majority baseline ({baseline_acc:.2f})')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_model_comparison.pdf")
    plt.savefig(FIGURES_DIR / "fig1_model_comparison.png", dpi=150)
    plt.close()

    # Figure 2: Accuracy-fairness tradeoff scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 80 + 200 * df_res['auc']  # bubble size encodes AUC
    colors_list = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_res)))
    for i, row in df_res.iterrows():
        ax.scatter(abs(row['demographic_parity_diff']), row['accuracy'],
                   s=sizes.iloc[i], c=[colors_list[i]], alpha=0.75,
                   edgecolors='black', linewidth=0.6)
        ax.annotate(row['classifier'],
                    (abs(row['demographic_parity_diff']), row['accuracy']),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel('|Demographic Parity Difference| (lower = fairer)')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy vs Fairness Tradeoff\n(bubble size $\\propto$ AUC)')
    ax.grid(alpha=0.3)
    ax.axvline(0.1, ls='--', color='red', alpha=0.5, label='DP diff = 0.1 (concerning)')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_accuracy_fairness.pdf")
    plt.savefig(FIGURES_DIR / "fig2_accuracy_fairness.png", dpi=150)
    plt.close()

    # Figure 3: Cost-sensitive ranking
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_cost = df_res.sort_values('cost_matrix_score', ascending=True).reset_index(drop=True)
    bars = ax.bar(df_cost['classifier'], df_cost['cost_matrix_score'],
                  color=plt.cm.RdYlGn_r(df_cost['cost_matrix_score'] / df_cost['cost_matrix_score'].max()),
                  edgecolor='black')
    ax.set_xticklabels(df_cost['classifier'], rotation=30, ha='right')
    ax.set_ylabel('Total Cost (5$\\times$FN + 1$\\times$FP)')
    ax.set_title('Cost-sensitive ranking (lower is better)')
    for bar, val in zip(bars, df_cost['cost_matrix_score']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_cost_sensitive.pdf")
    plt.savefig(FIGURES_DIR / "fig3_cost_sensitive.png", dpi=150)
    plt.close()

    # Figure 4: Top-15 feature importances (RF)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = fi.head(15).iloc[::-1]  # reverse so largest at top
    ax.barh(top['feature'], top['importance'], color='#2ca02c', edgecolor='black')
    ax.set_xlabel('Feature importance (Random Forest)')
    ax.set_title('Top-15 features driving credit risk prediction')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_feature_importance.pdf")
    plt.savefig(FIGURES_DIR / "fig4_feature_importance.png", dpi=150)
    plt.close()

    # Figure 5: Gender approval rate comparison for all models
    fig, ax = plt.subplots(figsize=(9, 4.5))
    male_rates = []
    female_rates = []
    for name in df_res['classifier']:
        preds = all_preds[name]
        male_rates.append(1 - preds[male_mask].mean())  # approval = predicted good = target=0
        female_rates.append(1 - preds[female_mask].mean())
    x = np.arange(len(df_res))
    w = 0.35
    ax.bar(x - w/2, male_rates, w, label='Male', color='#1f77b4', edgecolor='black')
    ax.bar(x + w/2, female_rates, w, label='Female', color='#ff7f0e', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['classifier'], rotation=30, ha='right')
    ax.set_ylabel('Approval rate (= P(predicted good credit))')
    ax.set_title('Approval-rate disparity by gender')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_gender_disparity.pdf")
    plt.savefig(FIGURES_DIR / "fig5_gender_disparity.png", dpi=150)
    plt.close()

    # ==================================================================
    # STATS JSON
    # ==================================================================
    best_row = df_res.iloc[0]
    best_cost_row = df_res.sort_values('cost_matrix_score').iloc[0]
    fairest_row = df_res.loc[df_res['demographic_parity_diff'].abs().idxmin()]

    stats = {
        'dataset': {'n_total': 1000, 'n_train': int(len(X_train)),
                    'n_test': int(len(X_test)), 'positive_class': 'bad credit',
                    'class_balance': f"{int((y_train==0).sum())} good / {int((y_train==1).sum())} bad",
                    'protected_attribute': 'gender (derived from personal_status_sex)'},
        'baseline_majority_accuracy': float(baseline_acc),
        'best_by_auc': {
            'name': best_row['classifier'], 'auc': float(best_row['auc']),
            'accuracy': float(best_row['accuracy']), 'f1': float(best_row['f1']),
            'cost': int(best_row['cost_matrix_score']),
            'dp_diff': float(best_row['demographic_parity_diff']),
            'eo_diff': float(best_row['equal_opportunity_diff']),
        },
        'best_by_cost': {
            'name': best_cost_row['classifier'],
            'cost': int(best_cost_row['cost_matrix_score']),
            'accuracy': float(best_cost_row['accuracy']),
            'auc': float(best_cost_row['auc']),
        },
        'fairest_model': {
            'name': fairest_row['classifier'],
            'dp_diff': float(fairest_row['demographic_parity_diff']),
            'auc': float(fairest_row['auc']),
            'accuracy': float(fairest_row['accuracy']),
        },
        'fairness_detail': fairness_detail,
        'top10_features_rf': fi.head(10).to_dict('records'),
        'top10_features_logreg_signed': lr_df.head(10)[['feature', 'coef']].to_dict('records'),
    }
    with open(RESULTS_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)

    # ==================================================================
    # HEADLINE OUTPUT
    # ==================================================================
    print("\n" + "=" * 70)
    print("HEADLINE NUMBERS")
    print("=" * 70)
    print(f"Majority baseline:     {baseline_acc:.4f} accuracy")
    print(f"Best AUC model:        {best_row['classifier']:15s} AUC={best_row['auc']:.4f} acc={best_row['accuracy']:.4f}")
    print(f"Best cost-sensitive:   {best_cost_row['classifier']:15s} cost={best_cost_row['cost_matrix_score']}")
    print(f"Fairest model:         {fairest_row['classifier']:15s} |DP diff|={abs(fairest_row['demographic_parity_diff']):.3f}")
    print(f"\nResults in: results/       Figures in: figures/")


if __name__ == "__main__":
    main()
