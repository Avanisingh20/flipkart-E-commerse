"""Part 2 - Classification and evaluation.

Run from anywhere:  python part2.py
Writes outputs/part2_results.txt, outputs/part2_thresholds.csv, outputs/part2_knn_cv.csv
"""
import math

import numpy as np

from common import (KNN_KS, SplitData, choose_k, confusion, fit_main_logistic,
                    knn_cross_validate, knn_predict, metrics_from_cells, money_cost,
                    predict_proba, bce_from_logits, train_logistic, write_csv,
                    write_results)

d = SplitData()
out = []


def add(key, val):
    out.append((key, val))


# ---- 2.1 Split and standardise ---------------------------------------------------------
add("train_rows", int(d.y_train.shape[0]))
add("train_positives", int(np.sum(d.y_train)))
add("test_rows", int(d.y_test.shape[0]))
add("test_positives", int(np.sum(d.y_test)))

# ---- 2.2 Logistic regression from scratch ----------------------------------------------
w, b, losses = fit_main_logistic(d)
for t in (0, 1, 10, 100, 1000):
    add("loss_epoch_%d" % t, losses[t])
for j in range(7):
    add("weight_X%d" % (j + 1), float(w[j]))
add("bias", b)
add("test_bce", bce_from_logits(d.Z_test @ w + b, d.y_test))

# ---- 2.3 Learning-rate effect --------------------------------------------------------------
for eta, label in ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1.0")):
    _w, _b, ls = train_logistic(d.Z_train, d.y_train, eta, 200, record_at=(200,))
    add("loss_epoch_200_eta_%s" % label, ls[200])

# ---- 2.4 Odds ratios ---------------------------------------------------------------------------
for j in range(7):
    add("odds_ratio_X%d" % (j + 1), math.exp(float(w[j])))

# ---- 2.5 Evaluate on the test rows at two thresholds ----------------------------------------
p_test = predict_proba(d.Z_test, w, b)
p_train = predict_proba(d.Z_train, w, b)
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(d.y_test, (p_test >= tau).astype(int))
    m = metrics_from_cells(tp, fp, tn, fn)
    pre = "tau_%s_" % tau
    add(pre + "tp", tp)
    add(pre + "fp", fp)
    add(pre + "tn", tn)
    add(pre + "fn", fn)
    add(pre + "accuracy", m["accuracy"])
    add(pre + "precision", m["precision"])
    add(pre + "recall", m["recall"])
    add(pre + "f1", m["f1"])
    add(pre + "specificity", m["specificity"])
tp, fp, tn, fn = confusion(d.y_test, np.zeros_like(d.y_test))
add("blind_accuracy", metrics_from_cells(tp, fp, tn, fn)["accuracy"])

# ---- 2.6 Threshold sweep (ROC points) -----------------------------------------------------------
rows = []
for k in range(1, 10):
    tau = k / 10.0
    tp, fp, tn, fn = confusion(d.y_test, (p_test >= tau).astype(int))
    m = metrics_from_cells(tp, fp, tn, fn)
    rows.append([tau, tp, fp, tn, fn, m["recall"], m["fpr"]])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# ---- 2.7 K-nearest neighbours with 5-fold cross-validation ------------------------------------
cv = knn_cross_validate(d.Z_train, d.y_train, KNN_KS)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, s] for k, s in cv])
best_k = choose_k(cv)
add("knn_best_k", best_k)
pred_knn, _share = knn_predict(d.Z_test, d.Z_train, d.y_train, best_k)
tp, fp, tn, fn = confusion(d.y_test, pred_knn)
m = metrics_from_cells(tp, fp, tn, fn)
add("knn_tp", tp)
add("knn_fp", fp)
add("knn_tn", tn)
add("knn_fn", fn)
add("knn_accuracy", m["accuracy"])
add("knn_precision", m["precision"])
add("knn_recall", m["recall"])
add("knn_f1", m["f1"])

# ---- 2.8 Choose the alert threshold with money (training rows only) --------------------------
best_tau, best_cost = None, None
for k in range(1, 20):
    tau = k / 20.0
    tp, fp, tn, fn = confusion(d.y_train, (p_train >= tau).astype(int))
    cost = money_cost(fp, fn)
    if best_cost is None or cost < best_cost:  # strict: ties keep the smaller tau
        best_tau, best_cost = tau, cost
add("cost_tau", best_tau)
add("cost_train_at_tau", best_cost)


def test_cost(tau):
    tp_, fp_, tn_, fn_ = confusion(d.y_test, (p_test >= tau).astype(int))
    return money_cost(fp_, fn_)


add("cost_test_at_tau", test_cost(best_tau))
add("cost_test_at_0.5", test_cost(0.5))
add("cost_test_flag_nobody", money_cost(0, int(np.sum(d.y_test))))

write_results("part2_results.txt", out)
print("part2 done: %d keys written to outputs/part2_results.txt" % len(out))
