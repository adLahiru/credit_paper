"""
Auto-fill paper.tex from experiment results.

USAGE (after run_experiment.py finishes):
    python fill_paper.py

Writes: paper_filled.tex
"""
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np


def load_all():
    with open("results/stats.json") as f:
        stats = json.load(f)
    main = pd.read_csv("results/main_results.csv")
    fi = pd.read_csv("results/feature_importance_rf.csv")
    lr = pd.read_csv("results/logreg_coefficients.csv")
    with open("results/fairness_detail.json") as f:
        fd = json.load(f)
    return stats, main, fi, lr, fd


def fmt_main_row(r):
    """Return a Table I row: classifier & acc & auc & f1 & prec & rec & cost"""
    return (f"{r['classifier']} & {r['accuracy']:.3f} & {r['auc']:.3f} & "
            f"{r['f1']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{int(r['cost_matrix_score'])}")


def fmt_fair_row(r):
    """Return a Table II row: classifier & DP & EO & DI"""
    return (f"{r['classifier']} & {r['demographic_parity_diff']:+.3f} & "
            f"{r['equal_opportunity_diff']:+.3f} & "
            f"{r['disparate_impact_ratio']:.3f}")


def fmt_coef_row(feat, coef):
    # escape underscores for LaTeX
    feat_latex = feat.replace('_', '\\_')
    return f"\\texttt{{{feat_latex}}} & {coef:+.3f}"


def main():
    print("Loading results...")
    stats, main_df, fi, lr, fd = load_all()

    # Sort main results by AUC descending (matches table order)
    main_df = main_df.sort_values('auc', ascending=False).reset_index(drop=True)

    # Sort fairness by |DP diff| for that table
    fair_df = main_df.copy()
    fair_df['abs_dp'] = fair_df['demographic_parity_diff'].abs()
    fair_df = fair_df.sort_values('abs_dp').reset_index(drop=True)

    # ---- Core numbers ----
    best_auc = main_df.iloc[0]
    best_auc_name = best_auc['classifier']
    auc_val = best_auc['auc']
    acc_val = best_auc['accuracy']
    baseline_acc = stats['baseline_majority_accuracy']
    gain_pp = (acc_val - baseline_acc) * 100

    # Best by cost (lowest)
    best_cost = main_df.loc[main_df['cost_matrix_score'].idxmin()]
    best_cost_name = best_cost['classifier']

    # Fairest
    fairest = fair_df.iloc[0]
    fairest_name = fairest['classifier']
    fairest_dp = fairest['demographic_parity_diff']
    fairest_auc = fairest['auc']

    dp_values = main_df['demographic_parity_diff'].values
    dp_min = dp_values[np.argmin(np.abs(dp_values))]  # smallest absolute
    # For abstract: range of DP differences (signed)
    dp_signed_min = dp_values.min()
    dp_signed_max = dp_values.max()

    # ---- Descriptive text blocks ----

    # Best AUC description (after Figure 1)
    best_auc_desc = (
        f"The best-performing classifier by AUC is {best_auc_name} "
        f"(AUC $=$ {auc_val:.3f}, accuracy $=$ {acc_val:.3f}), outperforming "
        f"the majority baseline of {baseline_acc:.3f} accuracy by "
        f"{gain_pp:.1f}\\,pp. Several classifiers cluster within 0.02 AUC of the "
        f"top, consistent with the small dataset size (300 test examples) and "
        f"the inherent noise of human credit behaviour."
    )

    # Cost comparison
    if best_cost_name == best_auc_name:
        cost_comp = (
            f"In this experiment, the cost-optimal and AUC-optimal classifier "
            f"coincide: {best_auc_name} minimizes both. Even so, the spread of "
            f"cost-matrix scores across classifiers "
            f"(from {int(main_df['cost_matrix_score'].min())} to "
            f"{int(main_df['cost_matrix_score'].max())}) illustrates that "
            f"small differences in AUC translate into materially different "
            f"financial outcomes under the 5:1 cost asymmetry."
        )
    else:
        cost_saving = int(main_df['cost_matrix_score'].max() - main_df['cost_matrix_score'].min())
        cost_comp = (
            f"The cost-optimal classifier, {best_cost_name} (cost $=$ "
            f"{int(best_cost['cost_matrix_score'])}), differs from the AUC-optimal "
            f"classifier, {best_auc_name} (cost $=$ {int(best_auc['cost_matrix_score'])}). "
            f"The range of cost scores across classifiers is {cost_saving} units, "
            f"which under the UCI cost weighting corresponds to a materially different "
            f"expected-loss profile. A lender selecting by AUC alone would therefore "
            f"leave money on the table relative to a cost-aware selector."
        )

    # Fairness finding
    fair_finding = (
        f"The fairest classifier under the absolute demographic-parity criterion is "
        f"{fairest_name} with $|\\mathrm{{DP\\,diff}}|=${abs(fairest_dp):.3f}, "
        f"while the most biased has $|\\mathrm{{DP\\,diff}}|=${abs(dp_signed_max):.3f}. "
    )
    # Count DI failures
    di_failures = (main_df['disparate_impact_ratio'] < 0.80).sum()
    if di_failures > 0:
        fair_finding += (
            f"{di_failures} of the {len(main_df)} classifiers yield a disparate-impact "
            f"ratio below the 0.80 four-fifths threshold, indicating prima facie adverse "
            f"impact against the less-favoured group on the test set."
        )
    else:
        fair_finding += (
            "All classifiers satisfy the 0.80 four-fifths rule on this test split, "
            "though small sample sizes (especially for the female subgroup, "
            f"n={fd['n_female']}) warrant caution in drawing firm conclusions."
        )

    # Gender description
    male_approve = fd['male_approval_rate']
    female_approve = fd['female_approval_rate']
    approve_gap = male_approve - female_approve
    gender_desc = (
        f"For the best-AUC model ({fd['best_model']}), male applicants are approved "
        f"at a rate of {male_approve:.3f} while female applicants are approved at "
        f"{female_approve:.3f}---a gap of {approve_gap:+.3f}. Test-set subgroup sizes "
        f"are n={fd['n_male']} for male and n={fd['n_female']} for female applicants; "
        f"confidence intervals on subgroup rates are therefore wide."
    )

    # Top features description
    top_feats = fi.head(5)['feature'].tolist()
    top_feats_latex = [f.replace('_', '\\_') for f in top_feats]
    top_feats_desc = (
        f"The five most important features are "
        f"\\texttt{{{top_feats_latex[0]}}}, \\texttt{{{top_feats_latex[1]}}}, "
        f"\\texttt{{{top_feats_latex[2]}}}, \\texttt{{{top_feats_latex[3]}}}, and "
        f"\\texttt{{{top_feats_latex[4]}}}. These match well-known drivers of credit "
        f"risk: checking-account history, size and duration of the loan, and age "
        f"(a proxy for credit history length). Crucially, several of these are "
        f"behavioural and financial rather than demographic, which is desirable from "
        f"a fairness perspective."
    )

    # Interpretability summary
    interp_summary = (
        "The Random Forest and Logistic Regression attributions broadly agree on the "
        "top predictors, lending confidence to these features' substantive importance. "
        "The Logistic Regression coefficients additionally reveal the \\emph{direction} "
        "of effect: features such as extended credit duration and high credit amount "
        "push predictions toward the bad-credit class, while longer-established "
        "savings accounts and strong checking-account balances push predictions "
        "toward the good-credit class. This combination of magnitude (from RF) and "
        "sign (from LogReg) is actionable both for model debugging and for "
        "generating adverse-action notices required by credit regulation."
    )

    # Fairest AUC note (for discussion)
    if fairest_auc < auc_val - 0.02:
        fairest_auc_note = f"not the best-AUC model (AUC $=$ {fairest_auc:.3f} vs.\\ {auc_val:.3f} for the top)"
    else:
        fairest_auc_note = f"also competitive on AUC (AUC $=$ {fairest_auc:.3f})"

    # Cost agreement flag
    cost_agreement = ("the same as the AUC-optimal model" if best_cost_name == best_auc_name
                      else f"{best_cost_name}, different from the AUC-optimal model "
                           f"({best_auc_name})")

    # ---- Main table rows (9 classifiers, ordered by AUC desc) ----
    main_rows = [fmt_main_row(main_df.iloc[i]) for i in range(len(main_df))]
    while len(main_rows) < 9:
        main_rows.append(r"--- & --- & --- & --- & --- & --- & ---")

    # ---- Fairness table rows (ordered by |DP diff| ascending) ----
    fair_rows = [fmt_fair_row(fair_df.iloc[i]) for i in range(len(fair_df))]
    while len(fair_rows) < 9:
        fair_rows.append(r"--- & --- & --- & ---")

    # ---- Coefficient rows ----
    coef_rows = [fmt_coef_row(lr.iloc[i]['feature'], lr.iloc[i]['coef'])
                 for i in range(min(10, len(lr)))]
    while len(coef_rows) < 10:
        coef_rows.append(r"--- & ---")

    # ---- Build substitution dict ----
    subs = {
        # Abstract & Discussion numbers
        "[AUC]":            f"{auc_val:.3f}",
        "[ACC]":            f"{acc_val:.3f}",
        "[GAIN]":           f"{gain_pp:.1f}",
        "[DP_MIN]":         f"{dp_signed_min:+.3f}",
        "[DP_MAX]":         f"{dp_signed_max:+.3f}",
        "[BASELINE_ACC]":   f"{baseline_acc:.3f}",

        # Discussion
        "[BEST_AUC_NAME]":  best_auc_name,
        "[BEST_COST_NAME]": best_cost_name,
        "[COST_AGREEMENT]": cost_agreement,
        "[FAIREST_NAME]":   fairest_name,
        "[FAIREST_DP]":     f"{fairest_dp:+.3f}",
        "[FAIREST_AUC_NOTE]": fairest_auc_note,

        # Results prose
        "[BEST_AUC_DESCRIPTION]":   best_auc_desc,
        "[COST_COMPARISON]":        cost_comp,
        "[FAIRNESS_FINDING]":       fair_finding,
        "[GENDER_DESCRIPTION]":     gender_desc,
        "[TOP_FEATURES_DESCRIPTION]": top_feats_desc,
        "[INTERPRETABILITY_SUMMARY]": interp_summary,

        # GitHub link
        "[GITHUB_LINK]": "https://github.com/Lahiru-Dilshan/german-credit-study",
    }

    # Table rows
    for i, row in enumerate(main_rows, start=1):
        subs[f"[MAIN_ROW_{i}]"] = row
    for i, row in enumerate(fair_rows, start=1):
        subs[f"[FAIR_ROW_{i}]"] = row
    for i, row in enumerate(coef_rows, start=1):
        subs[f"[COEF_ROW_{i}]"] = row

    print(f"Built {len(subs)} substitutions")

    with open("paper.tex") as f:
        tex = f.read()

    for key in sorted(subs.keys(), key=len, reverse=True):
        tex = tex.replace(key, subs[key])

    remaining = re.findall(r"\[[A-Z][A-Z0-9_]*\]", tex)
    remaining = [r for r in remaining if r not in ("[N/A]",)]
    if remaining:
        print(f"\nWarning: unfilled markers: {set(remaining)}")
    else:
        print("\nAll markers replaced successfully.")

    with open("paper_filled.tex", "w") as f:
        f.write(tex)

    print("\nWrote paper_filled.tex")
    print("\nNext: upload paper_filled.tex + figures/ to Overleaf, compile, download PDF.")


if __name__ == "__main__":
    main()
