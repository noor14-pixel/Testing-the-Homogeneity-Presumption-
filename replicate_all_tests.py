#!/usr/bin/env python3
"""
replicate_all_tests.py
====================================================================
One self-contained script that replicates every additional test run on
the Punjab MICS 2017-18 discipline / early-learning paper:

  PART 1  Baseline 11-practice ordered logit (unweighted, clustered on HH1)
          + cluster-robust Wald homogeneity tests. Checked against the
          manuscript values (35.87 and 21.95).
  PART 2  Nested additive count-index models and likelihood-ratio tests
          against the practice-level model (+ AIC / BIC).
  PART 3  Survey-weighted (chweight) robustness of the Wald tests and AMEs.
  PART 4  Belief, perceived child aggression and stimulation resources
          added to the outcome model: Wald tests and AMEs, unweighted and
          weighted, base vs extended on the SAME sample.

REQUIREMENTS
  Python 3.9+, numpy, pandas, scipy. No Stata or statsmodels needed.

USAGE
  python3 replicate_all_tests.py /path/to/help_me.dta
  (defaults to ./help_me.dta). A CSV of all Wald tests is written to
  ./wald_results.csv.

NOTES
  * Wald tests use the cluster-robust covariance (cluster = HH1) and test
    equality of LATENT-INDEX coefficients, as in the manuscript.
  * Likelihood-ratio tests and AIC/BIC use NON-clustered likelihoods (a
    clustered fit has only a pseudo-likelihood). The cluster-robust Wald
    test remains the primary inferential test.
  * Weighted fits maximise a pseudo-likelihood with chweight (rescaled to
    mean 1) and use the sandwich covariance clustered on HH1.
  * Variable codings for the added controls assume the standard MICS
    coding already decoded in the .dta (yes/no labels); see build_data().
====================================================================
"""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ variables
NONVIOLENT = ["take_away_privileges", "explain_behaviour", "gave_something_else"]
PSYCH      = ["shouted_yelled_child", "name_calling_dumb_lazy"]
PHYSICAL   = ["shook_child", "spanked_hit_slapped", "hit_bottom_with_item",
              "hit_face_head_ears", "hand_arm_leg", "beat_as_hard_one_could"]
VIOLENT    = PSYCH + PHYSICAL
DISC       = NONVIOLENT + VIOLENT
PHYS_NB    = [v for v in PHYSICAL if v != "beat_as_hard_one_could"]
VIOL_NB    = [v for v in VIOLENT if v != "beat_as_hard_one_could"]
CTRL = ["poor", "second", "middle", "fourth", "Rural", "Higher", "Secondary",
        "Middle", "Primary", "Female", "cage", "school", "cdisability"]
EXTRA = ["belief", "child_aggr", "books", "toy_home", "toy_shop", "toy_obj"]
OUTCOME, CLUSTER, WEIGHT = "learning_score", "HH1", "chweight"

PUBLISHED = {"viol8": 35.87, "three_domain": 21.95}   # manuscript Table 4, Panel C
RESULTS = []                                           # collected Wald results


# ------------------------------------------------------------------ helpers
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos]); out[~pos] = ez / (1.0 + ez)
    out[np.isposinf(z)] = 1.0; out[np.isneginf(z)] = 0.0
    return out


def to_numeric(s):
    """Stata-labelled column (category/object/numeric) -> float 0/1/NaN."""
    if s.dtype.name in ("category", "object"):
        v = s.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=s.index)
        out[v.isin(["yes", "1", "1.0", "has functional difficulty"])] = 1.0
        out[v.isin(["no", "0", "0.0", "has no functional difficulty"])] = 0.0
        return out
    return pd.to_numeric(s, errors="coerce")


def build_data(path):
    df = pd.read_stata(path)
    for c in set(DISC + CTRL + [OUTCOME, CLUSTER, WEIGHT]):
        df[c] = to_numeric(df[c])

    # added controls (PART 4)
    def yn(s, yes):
        v = s.astype(str).str.strip().str.lower()
        o = pd.Series(np.where(v.isin(yes), 1.0, 0.0), index=s.index)
        o[s.isna()] = np.nan
        return o

    df["belief"] = yn(df["belief_physical_punishment"], ["yes", "1", "1.0"])
    df["child_aggr"] = yn(df["child_aggressive"], ["yes"])
    df["books"] = (df["EC1"].astype(str).str.strip().str.lower() != "none").astype(float)

    def toy(s):
        v = s.astype(str).str.strip().str.lower()
        o = pd.Series(np.nan, index=s.index)
        o[v == "yes"] = 1.0; o[v == "no"] = 0.0      # 'dk' / 'no response' -> missing
        return o

    df["toy_home"], df["toy_shop"], df["toy_obj"] = toy(df["EC2A"]), toy(df["EC2B"]), toy(df["EC2C"])

    # additive counts (PART 2)
    df["viol_count"] = df[VIOLENT].sum(axis=1)
    df["nv_count"], df["ps_count"], df["ph_count"] = (df[NONVIOLENT].sum(axis=1),
                                                       df[PSYCH].sum(axis=1),
                                                       df[PHYSICAL].sum(axis=1))
    return df


# ------------------------------------------------------------------ ordered logit
def _cuts(p, k, ncat):
    interior = np.cumsum(np.concatenate([[p[k]], np.exp(p[k + 1:])]))
    return np.concatenate([[-np.inf], interior, [np.inf]])


def _ll_obs(p, X, yi, k, ncat, w):
    xb = X @ p[:k]; c = _cuts(p, k, ncat)
    pr = np.clip(sigmoid(c[yi + 1] - xb) - sigmoid(c[yi] - xb), 1e-12, 1.0)
    return w * np.log(pr)


def _hessian(f, p, eps=1e-4):
    m = len(p); H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            def ev(si, sj):
                q = p.copy(); q[i] += si * eps; q[j] += sj * eps; return f(q)
            H[i, j] = H[j, i] = (ev(1, 1) - ev(1, -1) - ev(-1, 1) + ev(-1, -1)) / (4 * eps * eps)
    return H


def fit(df, xcols, weight=None):
    """Ordered logit by ML; cluster-robust (CR1) sandwich on HH1."""
    y = df[OUTCOME].values.astype(float); X = df[xcols].values.astype(float)
    n, k = X.shape
    cats = np.sort(np.unique(y)); ncat = len(cats); yi = np.searchsorted(cats, y)
    w = np.ones(n) if weight is None else np.asarray(weight, float)
    nll = lambda p: -_ll_obs(p, X, yi, k, ncat, w).sum()
    r = minimize(nll, np.zeros(k + ncat - 1), method="BFGS", options={"maxiter": 5000, "gtol": 1e-7})
    p = r.x
    Hinv = np.linalg.inv(_hessian(nll, p))
    S = np.zeros((n, len(p))); e = 1e-5
    for j in range(len(p)):
        a, b = p.copy(), p.copy(); a[j] += e; b[j] -= e
        S[:, j] = (_ll_obs(a, X, yi, k, ncat, w) - _ll_obs(b, X, yi, k, ncat, w)) / (2 * e)
    G = pd.DataFrame(S).groupby(df[CLUSTER].values).sum().values
    ng = G.shape[0]
    V = Hinv @ (G.T @ G) @ Hinv * (ng / (ng - 1))
    return {"beta": p[:k], "V": V[:k, :k], "se": np.sqrt(np.diag(V[:k, :k])), "cuts": _cuts(p, k, ncat),
            "X": X, "xcols": list(xcols), "n": n, "npar": len(p), "ll": -r.fun, "ok": r.success}


def _probs(m, X):
    xb = X @ m["beta"]; c = m["cuts"]
    return np.column_stack([sigmoid(c[i + 1] - xb) - sigmoid(c[i] - xb) for i in range(len(c) - 1)])


def ame_binary(m, name, weight=None):
    """AME (pp) of a 0->1 switch on every outcome category."""
    j = m["xcols"].index(name); X0, X1 = m["X"].copy(), m["X"].copy()
    X0[:, j], X1[:, j] = 0.0, 1.0
    return np.average(_probs(m, X1) - _probs(m, X0), axis=0, weights=weight) * 100


def ame_plus_one(m, name):
    """AME (pp) of one additional practice (discrete +1) for a count regressor."""
    j = m["xcols"].index(name); X1 = m["X"].copy(); X1[:, j] += 1
    return (_probs(m, X1) - _probs(m, m["X"])).mean(axis=0) * 100


def wald(m, groups, label, tag=None, setting=""):
    """H0: coefficients equal within each group (no equality across groups)."""
    groups = groups if isinstance(groups[0], (list, tuple)) else [groups]
    idx = {v: i for i, v in enumerate(m["xcols"])}; R = []
    for g in groups:
        for a, b in zip(g[:-1], g[1:]):
            row = np.zeros(len(m["beta"])); row[idx[a]], row[idx[b]] = 1.0, -1.0; R.append(row)
    R = np.array(R); Rb = R @ m["beta"]
    W = float(Rb @ np.linalg.solve(R @ m["V"] @ R.T, Rb)); d = R.shape[0]; p = 1 - chi2.cdf(W, d)
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    print(f"  {label:44s} chi2({d}) = {W:7.2f}   p = {p:.4f} {star}")
    RESULTS.append({"setting": setting, "test": label, "chi2": round(W, 3), "df": d, "p": round(p, 5)})
    return W, d, p


def all_walds(m, setting):
    wald(m, VIOLENT, "All 8 violent items equal", setting=setting)
    wald(m, [NONVIOLENT, PSYCH, PHYSICAL], "Three-domain restriction (8 restr.)", setting=setting)
    wald(m, NONVIOLENT, "Non-violent items equal", setting=setting)
    wald(m, PSYCH, "Psychological items equal", setting=setting)
    wald(m, PHYSICAL, "Physical items equal", setting=setting)
    wald(m, PHYS_NB, "Physical items excl. beating", setting=setting)
    wald(m, VIOL_NB, "All violent items excl. beating", setting=setting)


def header(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ------------------------------------------------------------------ main
def main(path):
    df = build_data(path)
    base_cols = DISC + CTRL

    # ============ PART 1: baseline ============
    header("PART 1  Baseline 11-practice ordered logit (unweighted)")
    d0 = df.dropna(subset=DISC + CTRL + [OUTCOME, CLUSTER]).reset_index(drop=True)
    print(f"N = {len(d0)} (manuscript 15,590)   clusters = {d0[CLUSTER].nunique()} (manuscript 2,672)")
    B = fit(d0, base_cols)
    print(f"log-lik = {B['ll']:.2f}  (expected about -15143.13; SciPy may report 'not converged' "
          f"from a BFGS precision warning, which is harmless if this matches)\n")
    print("AMEs on P(no skills) / P(all three), pp:")
    for v in DISC:
        a = ame_binary(B, v); print(f"  {v:26s} {a[0]:7.2f} {a[-1]:8.2f}")
    print("\nCluster-robust Wald tests:")
    all_walds(B, "1 baseline unweighted")
    w1 = wald(B, VIOLENT, "[check] violent 8", setting="check")[0]
    w2 = wald(B, [NONVIOLENT, PSYCH, PHYSICAL], "[check] three-domain", setting="check")[0]
    ok = abs(w1 - PUBLISHED["viol8"]) < 0.05 and abs(w2 - PUBLISHED["three_domain"]) < 0.05
    print(f"\nREPLICATION vs manuscript (35.87 / 21.95): {'PASS' if ok else 'FAIL - do not trust downstream results'}")
    RESULTS[:] = [r for r in RESULTS if r["setting"] != "check"]
    if not ok:
        sys.exit(1)

    # ============ PART 2: nested count index ============
    header("PART 2  Nested additive count-index models (LR tests, non-clustered)")
    Bn = fit(d0, base_cols)  # same likelihood; LR uses log-lik only
    C = fit(d0, NONVIOLENT + ["viol_count"] + CTRL)
    D = fit(d0, ["nv_count", "ps_count", "ph_count"] + CTRL)
    aic = lambda m: -2 * m["ll"] + 2 * m["npar"]
    bic = lambda m: -2 * m["ll"] + np.log(m["n"]) * m["npar"]
    for lab, R, dfree in [("Violent count (8 items -> 1)", C, 7), ("Three-domain counts (11 -> 3)", D, 8)]:
        stat = 2 * (Bn["ll"] - R["ll"]); p = 1 - chi2.cdf(stat, dfree)
        print(f"  {lab:32s} LR chi2({dfree}) = {stat:6.2f}  p = {p:.4g} | AIC full {aic(Bn):.1f} vs {aic(R):.1f}"
              f" | BIC full {bic(Bn):.1f} vs {bic(R):.1f}")
        RESULTS.append({"setting": "2 nested LR", "test": lab, "chi2": round(stat, 3), "df": dfree, "p": round(p, 6)})
    print("  (AIC lower = better; BIC penalises extra parameters more heavily)\n")
    print("  AME of one more practice, pp: P(no skills) / P(all three)")
    for lab, m, names in [("violent count model", C, ["viol_count"]), ("three-domain model", D, ["nv_count", "ps_count", "ph_count"])]:
        for nm in names:
            a = ame_plus_one(m, nm); j = m["xcols"].index(nm); b, se = m["beta"][j], m["se"][j]
            print(f"  {lab:20s} {nm:11s} coef {b:7.4f} (se {se:.4f}, p {2*(1-norm.cdf(abs(b/se))):.4f})   {a[0]:6.2f} / {a[-1]:6.2f}")

    # ============ PART 3: weighted ============
    header("PART 3  Survey-weighted (chweight) vs unweighted")
    w = d0[WEIGHT].values
    print(f"chweight: min {w.min():.2f}  median {np.median(w):.2f}  max {w.max():.2f}  CV {w.std()/w.mean():.2f}  "
          f"Kish deff {len(w)*(w**2).sum()/w.sum()**2:.2f}")
    Wt = fit(d0, base_cols, weight=w / w.mean())
    print("\nAME on P(no skills), pp:   unweighted   weighted")
    for v in DISC:
        print(f"  {v:26s} {ame_binary(B, v)[0]:9.2f} {ame_binary(Wt, v, weight=w)[0]:11.2f}")
    print("\nWald tests, weighted:")
    all_walds(Wt, "3 baseline weighted")

    # ============ PART 4: belief / aggression / stimulation in outcome model ============
    header("PART 4  Add belief, child aggression, stimulation to the outcome model")
    d1 = df.dropna(subset=DISC + CTRL + [OUTCOME, CLUSTER] + EXTRA).reset_index(drop=True)
    print(f"Common sample N = {len(d1)} (base N = {len(d0)}); clusters = {d1[CLUSTER].nunique()}")
    print("prevalence %: " + "  ".join(f"{c}={d1[c].mean()*100:.1f}" for c in EXTRA))
    for lab, wv in [("UNWEIGHTED", None), ("WEIGHTED", d1[WEIGHT].values)]:
        ww = None if wv is None else wv / wv.mean()
        b1 = fit(d1, base_cols, weight=ww); e1 = fit(d1, base_cols + EXTRA, weight=ww)
        print(f"\n--- {lab} ---\nadded covariates (coef, se, p):")
        for c in EXTRA:
            j = e1["xcols"].index(c); b, se = e1["beta"][j], e1["se"][j]
            print(f"  {c:11s} {b:7.3f} {se:6.3f}  p = {2*(1-norm.cdf(abs(b/se))):.3f}")
        print("AME on P(no skills), pp:       base  extended")
        for v in DISC:
            print(f"  {v:26s} {ame_binary(b1, v, weight=wv)[0]:7.2f} {ame_binary(e1, v, weight=wv)[0]:8.2f}")
        print("Wald tests, base (common sample):");  all_walds(b1, f"4 {lab.lower()} base, common sample")
        print("Wald tests, EXTENDED:");              all_walds(e1, f"4 {lab.lower()} extended")

    pd.DataFrame(RESULTS).to_csv("wald_results.csv", index=False)
    print("\nWrote wald_results.csv. Done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "help_me.dta")
