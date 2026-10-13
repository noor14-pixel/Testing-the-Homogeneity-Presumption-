#!/usr/bin/env python3
"""
aggregation_test.py
====================================================================
Reproducible replication and aggregation test for:
  "When Discipline Is Not One Behavior" (Punjab MICS 2017-18)

WHAT THIS SCRIPT DOES, IN ORDER
--------------------------------------------------------------------
STEP 0   Load help_me.dta and build the analysis variables.
STEP 1   Fit the baseline 11-practice ordered logit exactly as in
         final_final_sproj.do (`ologit ... vce(cluster HH1)`,
         UNWEIGHTED, clustered on HH1). This reproduces Table 2 of
         the manuscript. The script checks the reproduced AMEs
         against the published Table 2 values and prints PASS/FAIL
         for each one, on purpose, so a silent numerical drift can
         never be mistaken for a validated replication.
STEP 2   Build the conventional aggregate discipline measures
         (a single "any violent discipline" binary, and the
         3-domain non-violent / psychological / physical grouping)
         and re-fit the identical specification with the 11
         practices replaced by each aggregate.
STEP 3   Run Wald tests, using the cluster-robust covariance matrix
         from the Step 1 (11-practice) model, of the restriction
         that practices within a grouping share a single
         coefficient. This is the formal version of "aggregation
         throws away information."
STEP 4   Print a summary table matching Table 3 of the manuscript.

REQUIREMENTS
--------------------------------------------------------------------
Python 3.9+, numpy, scipy, pandas (pandas can read .dta directly;
no Stata license or pyreadstat needed).

USAGE
--------------------------------------------------------------------
    python3 aggregation_test.py /path/to/help_me.dta

If no path is given, it looks for help_me.dta in the current
directory.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
--------------------------------------------------------------------
- It does not apply sampling weights. The original do-file specification
  has no [pw=] and no svyset; this is confirmed by the fact that the
  UNWEIGHTED replication below reproduces the published Table 2 to the
  decimal, while re-weighting by chweight does not. If you want the
  survey-weighted version instead, multiply the log-likelihood
  contributions by chweight in `negloglik()` below (marked with a
  comment) and re-run; expect materially different coefficients
  (e.g. explain_behaviour AME moves from about -3.90 to about -2.79).
- It does not implement the "beta_aggregate = sum of w_k * beta_k"
  formalization some external advice suggested. That identity holds
  for OLS with a linear aggregate; it does not hold here, because (a)
  the model is an ordered logit (nonlinear MLE, no such closed form)
  and (b) the conventional aggregate is a binary OR of dummies, not a
  linear sum, even under OLS. The Wald test in STEP 3 is the correct,
  rigorous substitute and is what the manuscript actually reports.
====================================================================
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Variable lists (must match final_final_sproj.do exactly)
# ------------------------------------------------------------------
DISC = [
    "take_away_privileges", "explain_behaviour", "shook_child",
    "shouted_yelled_child", "gave_something_else", "spanked_hit_slapped",
    "hit_bottom_with_item", "name_calling_dumb_lazy", "hit_face_head_ears",
    "hand_arm_leg", "beat_as_hard_one_could",
]
CTRL = [
    "poor", "second", "middle", "fourth", "Rural", "Higher", "Secondary",
    "Middle", "Primary", "Female", "cage", "school", "cdisability",
]
OUTCOME = "learning_score"
CLUSTER = "HH1"
WEIGHT = "chweight"  # present in the data but NOT used by default; see docstring

# Published Table 2 values (Cat0 = P(no skills), Cat3 = P(all 3 skills)),
# in percentage points, used only to self-check the replication in STEP 1.
TABLE2_CHECK = {
    "explain_behaviour":      (-3.90, 2.24),
    "gave_something_else":    (-5.04, 2.89),
    "shouted_yelled_child":   ( 3.21, -1.84),
    "name_calling_dumb_lazy": ( 3.37, -1.94),
    "spanked_hit_slapped":    ( 1.48, -0.85),
}
TOL = 0.05  # percentage points; replication must match published Table 2 within this


# ------------------------------------------------------------------
# Numerically stable logistic sigmoid, vectorized, handles +/-inf
# cutpoints cleanly (needed for the outer categories of the ordered
# logit where one cutpoint is -infinity or +infinity).
# ------------------------------------------------------------------
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    out[np.isposinf(z)] = 1.0
    out[np.isneginf(z)] = 0.0
    return out


def to_numeric(series):
    """Coerce a Stata-labeled column (categorical/object or numeric) to float 0/1/NaN."""
    if series.dtype.name in ("category", "object"):
        v = series.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=series.index)
        out[v.isin(["yes", "1", "1.0", "has functional difficulty"])] = 1.0
        out[v.isin(["no", "0", "0.0", "has no functional difficulty"])] = 0.0
        return out
    return pd.to_numeric(series, errors="coerce")


# ------------------------------------------------------------------
# Ordered logit: per-observation log-likelihood, given the packed
# parameter vector p = [beta (k), first cutpoint, log-increments...].
# Log-increments guarantee the cutpoints are monotonically increasing
# without needing constrained optimization.
# ------------------------------------------------------------------
def unpack_cutpoints(p, k, n_cats):
    first = p[k]
    incr = np.exp(p[k + 1:])  # length n_cats - 2, all positive
    interior = np.cumsum(np.concatenate([[first], incr]))
    return np.concatenate([[-np.inf], interior, [np.inf]])


def per_obs_loglik(p, X, y_idx, k, n_cats, w):
    beta = p[:k]
    cuts = unpack_cutpoints(p, k, n_cats)
    xb = X @ beta
    hi = cuts[y_idx + 1] - xb
    lo = cuts[y_idx] - xb
    prob = sigmoid(hi) - sigmoid(lo)
    prob = np.clip(prob, 1e-12, 1.0)
    # To use sampling weights instead of the unweighted specification,
    # replace `w * np.log(prob)` with `w * chweight_array * np.log(prob)`.
    return w * np.log(prob)


def neg_loglik(p, X, y_idx, k, n_cats, w):
    return -np.sum(per_obs_loglik(p, X, y_idx, k, n_cats, w))


def numeric_hessian(f, p, eps=1e-4):
    m = len(p)
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            pp = p.copy(); pp[i] += eps; pp[j] += eps; a = f(pp)
            pm = p.copy(); pm[i] += eps; pm[j] -= eps; b = f(pm)
            mp = p.copy(); mp[i] -= eps; mp[j] += eps; c = f(mp)
            mm = p.copy(); mm[i] -= eps; mm[j] -= eps; d = f(mm)
            H[i, j] = H[j, i] = (a - b - c + d) / (4 * eps * eps)
    return H


def fit_ordered_logit(df, xcols, cluster_col, outcome_col=OUTCOME, weight=None):
    """
    Fit an ordered logit by maximum likelihood, with (optionally) a
    fixed observation weight vector, and return coefficients, the
    cluster-robust covariance matrix (restricted to the beta block),
    the full parameter vector, cutpoints, category values, and the
    design matrix (needed downstream for AMEs).
    """
    y = df[outcome_col].values.astype(float)
    X = df[xcols].values.astype(float)
    cl = df[cluster_col].values
    n, k = X.shape
    cats = np.sort(np.unique(y))
    n_cats = len(cats)
    y_idx = np.searchsorted(cats, y)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)

    nll = lambda p: neg_loglik(p, X, y_idx, k, n_cats, w)

    p0 = np.concatenate([np.zeros(k), [0.0], np.zeros(n_cats - 2)])
    # Nelder-Mead first for robustness to a flat/curved start, then polish with BFGS.
    r = minimize(nll, p0, method="Nelder-Mead",
                 options={"maxiter": 80000, "maxfev": 80000, "xatol": 1e-9, "fatol": 1e-9})
    r = minimize(nll, r.x, method="BFGS", options={"maxiter": 5000, "gtol": 1e-9})
    p_hat = r.x

    # Cluster-robust ("CR1") sandwich covariance: numeric Hessian for the
    # bread, summed-within-cluster numeric scores for the meat.
    Hinv = np.linalg.inv(numeric_hessian(nll, p_hat))
    eps = 1e-5
    S = np.zeros((n, len(p_hat)))
    for j in range(len(p_hat)):
        pp = p_hat.copy(); pp[j] += eps
        pm = p_hat.copy(); pm[j] -= eps
        S[:, j] = (per_obs_loglik(pp, X, y_idx, k, n_cats, w)
                   - per_obs_loglik(pm, X, y_idx, k, n_cats, w)) / (2 * eps)
    score_df = pd.DataFrame(S)
    score_df["cl"] = cl
    G = score_df.groupby("cl").sum().values
    n_clusters = G.shape[0]
    V_full = Hinv @ (G.T @ G) @ Hinv * (n_clusters / (n_clusters - 1))

    beta = p_hat[:k]
    V_beta = V_full[:k, :k]
    cuts = unpack_cutpoints(p_hat, k, n_cats)

    return {
        "beta": beta, "V_beta": V_beta, "se": np.sqrt(np.diag(V_beta)),
        "p_full": p_hat, "V_full": V_full, "cuts": cuts, "cats": cats,
        "X": X, "xcols": list(xcols), "n": n, "n_clusters": n_clusters,
        "converged": r.success, "loglik": -r.fun,
    }


def ame_binary(fit, colname, weight=None):
    """
    Average marginal effect of switching a binary regressor from 0 to 1,
    for every outcome category, matching Stata's `margins, dydx()` applied
    to a 0/1 covariate (a discrete difference, not a derivative, since the
    regressor is binary).
    """
    beta, cuts, X, xcols = fit["beta"], fit["cuts"], fit["X"], fit["xcols"]
    j = xcols.index(colname)
    n_cats = len(cuts) - 1
    w = np.ones(X.shape[0]) if weight is None else np.asarray(weight, dtype=float)

    def cat_probs(Xm):
        xb = Xm @ beta
        return np.column_stack([sigmoid(cuts[c + 1] - xb) - sigmoid(cuts[c] - xb)
                                 for c in range(n_cats)])

    X0 = X.copy(); X0[:, j] = 0.0
    X1 = X.copy(); X1[:, j] = 1.0
    diff = cat_probs(X1) - cat_probs(X0)
    return np.average(diff, axis=0, weights=w) * 100.0  # percentage points


def wald_equal_coefs(fit, varnames_or_groups, label):
    """
    Wald test of H0: all coefficients within each group are equal to one
    another, using the cluster-robust covariance from `fit`. This is the
    formal aggregation-restriction test (Table 3, Panel C).

    Pass a single flat list (e.g. PSYCH) to test that ALL items in that
    one list share a coefficient (a chain of pairwise-adjacent equality
    restrictions, len(list)-1 restrictions).

    Pass a LIST OF LISTS (e.g. [NONVIOLENT, PSYCH, PHYSICAL]) to test,
    jointly, that each group internally shares a coefficient -- WITHOUT
    imposing any equality ACROSS groups. This is the correct way to test
    "the 3-domain aggregation restriction": it restricts within each of
    the three domains simultaneously, but never equates a non-violent
    coefficient to a psychological one, which a naive single chain over
    the concatenated list would incorrectly do.
    """
    beta, V, xcols = fit["beta"], fit["V_beta"], fit["xcols"]
    idx = {name: xcols.index(name) for name in xcols}
    k = len(beta)

    is_grouped = len(varnames_or_groups) > 0 and isinstance(varnames_or_groups[0], (list, tuple))
    groups = varnames_or_groups if is_grouped else [varnames_or_groups]

    R = []
    for group in groups:
        for a, b in zip(group[:-1], group[1:]):
            row = np.zeros(k)
            row[idx[a]] = 1.0
            row[idx[b]] = -1.0
            R.append(row)
    R = np.array(R)
    Rb = R @ beta
    M = R @ V @ R.T
    W = float(Rb @ np.linalg.solve(M, Rb))
    df = R.shape[0]
    pval = 1 - chi2.cdf(W, df)
    print(f"  {label:58s} chi2({df}) = {W:7.2f}   p = {pval:.4f}"
          + ("  ***" if pval < 0.001 else "  **" if pval < 0.01 else "  *" if pval < 0.05 else ""))
    return W, df, pval


# ------------------------------------------------------------------
# Domain groupings used throughout (must match Section 3.5 / 3-domain
# convention used in the literature this paper argues against).
# ------------------------------------------------------------------
NONVIOLENT = ["take_away_privileges", "explain_behaviour", "gave_something_else"]
PSYCH      = ["shouted_yelled_child", "name_calling_dumb_lazy"]
PHYSICAL   = ["shook_child", "spanked_hit_slapped", "hit_bottom_with_item",
              "hit_face_head_ears", "hand_arm_leg", "beat_as_hard_one_could"]
PHYSICAL_NO_BEAT = [v for v in PHYSICAL if v != "beat_as_hard_one_could"]
VIOLENT    = PSYCH + PHYSICAL
VIOLENT_NO_BEAT = [v for v in VIOLENT if v != "beat_as_hard_one_could"]


def load_and_build(path):
    df = pd.read_stata(path)
    for c in set(DISC + CTRL + [OUTCOME, CLUSTER, WEIGHT]):
        if c in df.columns:
            df[c] = to_numeric(df[c])
    # Aggregate measures, built exactly as in the manuscript's Section 5.2
    df["any_nonviolent"] = (df[NONVIOLENT].sum(axis=1) > 0).astype(float)
    df["any_psych"]      = (df[PSYCH].sum(axis=1) > 0).astype(float)
    df["any_physical"]   = (df[PHYSICAL].sum(axis=1) > 0).astype(float)
    df["any_violent"]    = (df[VIOLENT].sum(axis=1) > 0).astype(float)

    needed = DISC + CTRL + [OUTCOME, CLUSTER]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def main(path):
    print("=" * 78)
    print("STEP 0: Load data and build variables")
    print("=" * 78)
    df = load_and_build(path)
    print(f"Estimation sample: N = {len(df)}  (manuscript reports N = 15,590)")
    print(f"Clusters (HH1):     {df[CLUSTER].nunique()}  (manuscript reports 2,672)")
    for c in ["explain_behaviour", "shouted_yelled_child", "name_calling_dumb_lazy",
              "beat_as_hard_one_could", "hit_bottom_with_item"]:
        print(f"  prevalence {c:26s} {df[c].mean()*100:5.1f}%")

    print("\n" + "=" * 78)
    print("STEP 1: Baseline 11-practice ordered logit (replicates Table 2)")
    print("=" * 78)
    xcols = DISC + CTRL
    fit11 = fit_ordered_logit(df, xcols, CLUSTER)
    print(f"Converged: {fit11['converged']}   log-likelihood: {fit11['loglik']:.1f}")
    print(f"\n{'variable':26s}{'coef':>9s}{'se':>9s}{'z':>8s}{'p':>8s}")
    for i, name in enumerate(xcols):
        b, se = fit11["beta"][i], fit11["se"][i]
        z = b / se
        pv = 2 * (1 - norm.cdf(abs(z)))
        print(f"{name:26s}{b:9.4f}{se:9.4f}{z:8.2f}{pv:8.3f}")

    print("\nValidation against published Table 2 (AMEs, percentage points):")
    all_pass = True
    for name, (cat0_pub, cat3_pub) in TABLE2_CHECK.items():
        ame = ame_binary(fit11, name)
        cat0, cat3 = ame[0], ame[-1]
        ok0 = abs(cat0 - cat0_pub) < TOL
        ok3 = abs(cat3 - cat3_pub) < TOL
        all_pass = all_pass and ok0 and ok3
        status = "PASS" if (ok0 and ok3) else "FAIL"
        print(f"  {name:26s} Cat0: {cat0:6.2f} (pub {cat0_pub:6.2f})  "
              f"Cat3: {cat3:6.2f} (pub {cat3_pub:6.2f})   [{status}]")
    print(f"\n{'ALL REPLICATION CHECKS PASSED' if all_pass else 'REPLICATION CHECK FAILED -- DO NOT TRUST DOWNSTREAM RESULTS'}")
    if not all_pass:
        sys.exit(1)

    print("\n" + "=" * 78)
    print("STEP 2: Aggregate specifications (Table 3, Panel A)")
    print("=" * 78)
    specs = {
        "any_violent (single binary)": ["any_violent"],
        "3-domain (non-violent / psych / physical)": ["any_nonviolent", "any_psych", "any_physical"],
    }
    agg_fits = {}
    for label, dv in specs.items():
        f = fit_ordered_logit(df, dv + CTRL, CLUSTER)
        agg_fits[label] = f
        print(f"\n{label}  (log-likelihood = {f['loglik']:.1f})")
        print(f"{'variable':22s}{'coef':>9s}{'se':>9s}{'z':>8s}{'p':>8s}{'AME_Cat0':>10s}{'AME_Cat3':>10s}")
        for i, name in enumerate(dv):
            b, se = f["beta"][i], f["se"][i]
            z = b / se
            pv = 2 * (1 - norm.cdf(abs(z)))
            ame = ame_binary(f, name)
            print(f"{name:22s}{b:9.4f}{se:9.4f}{z:8.2f}{pv:8.3f}{ame[0]:10.2f}{ame[-1]:10.2f}")

    print("\n" + "=" * 78)
    print("STEP 3: Wald tests of the aggregation restriction (Table 3, Panel C)")
    print("(covariance matrix from the Step 1 eleven-practice model)")
    print("=" * 78)
    wald_equal_coefs(fit11, VIOLENT, "All 8 violent items share one coefficient")
    wald_equal_coefs(fit11, [NONVIOLENT, PSYCH, PHYSICAL],
                      "Three-domain restriction (within-domain only, 8 restrictions)")
    wald_equal_coefs(fit11, NONVIOLENT, "Non-violent items share one coefficient")
    wald_equal_coefs(fit11, PSYCH, "Psychological items share one coefficient")
    wald_equal_coefs(fit11, PHYSICAL, "Physical items share one coefficient")
    wald_equal_coefs(fit11, PHYSICAL_NO_BEAT,
                      "Physical items, EXCLUDING 'beat as hard as could'")
    wald_equal_coefs(fit11, VIOLENT_NO_BEAT,
                      "All violent items, EXCLUDING 'beat as hard as could'")

    print("\n" + "=" * 78)
    print("STEP 4: Range of practice-level AMEs pooled by each aggregate (Panel B)")
    print("=" * 78)
    for label, group in [("any_violent (8 items)", VIOLENT),
                          ("non-violent domain (3 items)", NONVIOLENT),
                          ("psychological domain (2 items)", PSYCH),
                          ("physical domain (6 items)", PHYSICAL)]:
        cat0s, cat3s = [], []
        for v in group:
            ame = ame_binary(fit11, v)
            cat0s.append(ame[0]); cat3s.append(ame[-1])
        print(f"  {label:32s} Cat0 range: [{min(cat0s):6.2f}, {max(cat0s):6.2f}]   "
              f"Cat3 range: [{min(cat3s):6.2f}, {max(cat3s):6.2f}]")

    print("\nDone. These numbers are what Section 5.2 / Table 3 of the manuscript report.")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "help_me.dta"
    main(data_path)
