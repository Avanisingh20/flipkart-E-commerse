"""Part 1 - Estimation, inference and a regression baseline (all 800 rows).

Run from anywhere:  python part1.py     ->  outputs/part1_results.txt
"""
import math

import numpy as np

from common import (load_records, mean_of, sample_cov, sample_sd, sample_var,
                    write_results)

X, y_outcome, value = load_records()
n = int(value.shape[0])
x4 = X[:, 3]  # avg_order_value_inr_hundreds
out = []


def add(key, val):
    out.append((key, val))


# ---- 1.1 Estimate with an interval --------------------------------------------------
mean_value = mean_of(value)
sd_value = sample_sd(value)
se_value = sd_value / math.sqrt(n)
add("n", n)
add("mean_value", mean_value)
add("sd_value", sd_value)
add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value)
add("ci95_high", mean_value + 1.96 * se_value)

# ---- 1.2 Test the planning claim H0: mu = 8.0 ----------------------------------------
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat)
add("reject_h0", abs(z_stat) > 1.96)

# ---- 1.3 Size of a future sample (E = 0.5, 95% confidence) -----------------------------
n_required = int(math.ceil((1.96 * sd_value / 0.5) ** 2))
add("n_required", n_required)


# ---- closed-form simple regression (used for 1.4 and again, on logs, in 1.6) ----------
def simple_regression(x, yy):
    slope = sample_cov(x, yy) / sample_var(x)
    intercept = mean_of(yy) - slope * mean_of(x)
    fitted = intercept + slope * x
    mean_y = mean_of(yy)
    tss = float(np.sum((yy - mean_y) ** 2))
    ess = float(np.sum((fitted - mean_y) ** 2))
    resid = yy - fitted
    rss = float(np.sum(resid ** 2))
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / yy.shape[0])
    mae = float(np.sum(np.abs(resid))) / yy.shape[0]
    return slope, intercept, fitted, resid, tss, ess, rss, r2, rmse, mae


# ---- 1.4 Simple regression, value on X4 --------------------------------------------------
(reg_slope, reg_intercept, fitted_raw, resid_raw,
 tss, ess, rss, r2, rmse, mae) = simple_regression(x4, value)
add("reg_slope", reg_slope)
add("reg_intercept", reg_intercept)
add("tss", tss)
add("ess", ess)
add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", r2)
add("rmse", rmse)
add("mae", mae)

# ---- 1.5 The same line by gradient descent -----------------------------------------------
mean_x4 = mean_of(x4)
sd_x4 = sample_sd(x4)
zx = (x4 - mean_x4) / sd_x4
m, c = 0.0, 0.0
eta = 0.1
cost_after = {}
for t in range(501):
    e = value - (m * zx + c)
    if t in (0, 1, 10, 100, 500):
        cost_after[t] = float(np.sum(e ** 2)) / n
    if t == 500:
        break
    grad_m = -2.0 * float(np.sum(zx * e)) / n
    grad_c = -2.0 * float(np.sum(e)) / n
    m -= eta * grad_m
    c -= eta * grad_c
add("gd_J0", cost_after[0])
add("gd_J1", cost_after[1])
add("gd_J10", cost_after[10])
add("gd_J100", cost_after[100])
add("gd_J500", cost_after[500])
gd_slope = m / sd_x4
gd_intercept = c - m * mean_x4 / sd_x4
add("gd_slope", gd_slope)
add("gd_intercept", gd_intercept)
add("gd_slope_gap", abs(gd_slope - reg_slope))


# ---- 1.6 Funnel check and the log remedy ------------------------------------------------
def half_residual_sds(fitted, resid):
    """Sort rows by fitted value (stable, ties by row index); SD of residuals in each half."""
    order = np.argsort(fitted, kind="stable")
    lower = resid[order[:n // 2]]
    upper = resid[order[n // 2:]]
    return sample_sd(lower), sample_sd(upper)


lo_raw, up_raw = half_residual_sds(fitted_raw, resid_raw)
add("resid_sd_lower_raw", lo_raw)
add("resid_sd_upper_raw", up_raw)
add("resid_sd_ratio_raw", up_raw / lo_raw)

ln_value = np.log(value)
ln_x4 = np.log(x4)
(ll_slope, ll_intercept, fitted_log, resid_log,
 _tss_l, _ess_l, _rss_l, ll_r2, ll_rmse, _mae_l) = simple_regression(ln_x4, ln_value)
add("loglog_intercept", ll_intercept)
add("loglog_slope", ll_slope)
add("loglog_r2", ll_r2)
add("loglog_rmse", ll_rmse)
lo_log, up_log = half_residual_sds(fitted_log, resid_log)
add("resid_sd_lower_log", lo_log)
add("resid_sd_upper_log", up_log)
add("resid_sd_ratio_log", up_log / lo_log)

# ---- 1.7 Multiple regression by the normal equation ----------------------------------------
# columns: constant, ln(X4), X1, X2, X3, X5, X6, X7  (X1..X7 sit in X[:, 0..6])
A = np.column_stack([np.ones(n), ln_x4, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ ln_value)
for j in range(8):
    add("mr_beta_%d" % j, float(beta[j]))
fitted_mr = A @ beta
mean_ln = mean_of(ln_value)
tss_mr = float(np.sum((ln_value - mean_ln) ** 2))
rss_mr = float(np.sum((ln_value - fitted_mr) ** 2))
mr_r2 = 1.0 - rss_mr / tss_mr


def adjusted_r2(r2_value, n_obs, p_predictors):
    return 1.0 - (1.0 - r2_value) * (n_obs - 1) / (n_obs - p_predictors - 1)


add("mr_r2", mr_r2)
add("mr_adj_r2", adjusted_r2(mr_r2, n, 7))
add("loglog_adj_r2", adjusted_r2(ll_r2, n, 1))

write_results("part1_results.txt", out)
print("part1 done: %d keys written to outputs/part1_results.txt" % len(out))
