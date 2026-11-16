"""Part 4 - Model-choice and architecture memo numbers.

Run from anywhere:  python part4.py
Writes outputs/part4_results.txt, part4_splits.csv, part4_voting.csv,
       part4_adaboost.csv, part4_architecture.csv
Re-uses the Part 2 pipeline through common.py (same split, same standardisation, same
logistic model, same K chosen by the same cross-validation); nothing is re-tuned here.
"""
import math

import numpy as np

from common import (KNN_KS, SplitData, choose_k, confusion, fit_main_logistic,
                    knn_cross_validate, knn_predict, metrics_from_cells, money_cost,
                    predict_proba, write_csv, write_results)

d = SplitData()
out = []


def add(key, val):
    out.append((key, val))


# ---- 4.1 Which split is purest? (training rows, original units) ------------------------------
def entropy(pos, total):
    if total == 0:
        return 0.0
    h = 0.0
    for c in (pos, total - pos):
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def gini(pos, total):
    if total == 0:
        return 0.0
    p = pos / total
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


y_tr = d.y_train
n_tr = int(y_tr.shape[0])
pos_tr = int(np.sum(y_tr))
root_entropy = entropy(pos_tr, n_tr)
root_gini = gini(pos_tr, n_tr)
add("root_entropy", root_entropy)
add("root_gini", root_gini)

Xo = d.X_train  # columns: X1..X7 -> index 0..6
split_masks = {
    "S1": Xo[:, 5] == 1,      # X6 = 1 (cod_flag)
    "S2": Xo[:, 6] == 1,      # X7 = 1 (plus_member_flag)
    "S3": Xo[:, 2] > 0.5,     # X3 > 0.5 (discount_order_share)
    "S4": Xo[:, 0] <= 24,     # X1 <= 24 (customer_tenure_months)
}
split_rows = []
gain_info, gain_gini = {}, {}
for name in ("S1", "S2", "S3", "S4"):
    yes = split_masks[name]
    n_yes, n_no = int(np.sum(yes)), int(np.sum(~yes))
    p_yes, p_no = int(np.sum(y_tr[yes])), int(np.sum(y_tr[~yes]))
    ig = root_entropy - (n_yes / n_tr) * entropy(p_yes, n_yes) - (n_no / n_tr) * entropy(p_no, n_no)
    gg = root_gini - (n_yes / n_tr) * gini(p_yes, n_yes) - (n_no / n_tr) * gini(p_no, n_no)
    gain_info[name], gain_gini[name] = ig, gg
    split_rows.append([name, n_yes, p_yes, n_no, p_no, ig, gg])
write_csv("part4_splits.csv",
          ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
          split_rows)


def best_of(gains):
    best, best_g = None, -1.0
    for name in ("S1", "S2", "S3", "S4"):  # ties keep the earlier split
        if gains[name] > best_g + 1e-12:
            best, best_g = name, gains[name]
    return best


best_info = best_of(gain_info)
best_gini = best_of(gain_gini)
add("best_split_by_info_gain", best_info)
add("best_split_by_gini_gain", best_gini)

# ---- 4.2 A one-question model and two ways to combine models ---------------------------------
yes_tr = split_masks[best_info]
stump_rate_yes = float(np.sum(y_tr[yes_tr])) / int(np.sum(yes_tr))
stump_rate_no = float(np.sum(y_tr[~yes_tr])) / int(np.sum(~yes_tr))
add("stump_rate_yes", stump_rate_yes)
add("stump_rate_no", stump_rate_no)

# test-row probabilities of the three base models
w, b, _ = fit_main_logistic(d)
p_logit = predict_proba(d.Z_test, w, b)

best_k = choose_k(knn_cross_validate(d.Z_train, d.y_train, KNN_KS))
knn_class, p_knn = knn_predict(d.Z_test, d.Z_train, d.y_train, best_k)

# the stump needs the same split condition on the TEST rows (original units)
Xt = d.X_test
test_masks = {
    "S1": Xt[:, 5] == 1,
    "S2": Xt[:, 6] == 1,
    "S3": Xt[:, 2] > 0.5,
    "S4": Xt[:, 0] <= 24,
}
p_stump = np.where(test_masks[best_info], stump_rate_yes, stump_rate_no)

TAU = 0.3
pred_logit = (p_logit >= TAU).astype(int)
pred_knn = knn_class  # at least (K + 1) / 2 of K neighbours positive
pred_stump = (p_stump >= TAU).astype(int)
pred_hard = ((pred_logit + pred_knn + pred_stump) >= 2).astype(int)
p_mean = (p_logit + p_knn + p_stump) / 3.0
pred_soft = (p_mean >= TAU).astype(int)

vote_rows = []
model_cells = {}
for name, pred in (("logistic", pred_logit), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", pred_hard), ("soft_vote", pred_soft)):
    tp, fp, tn, fn = confusion(d.y_test, pred)
    m = metrics_from_cells(tp, fp, tn, fn)
    model_cells[name] = (tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
          vote_rows)

# Extra keys (not required by the brief): INR cost of each of the five Task 4.2 predictions
# on the test rows by the Task 2.8 rule (50 per false alarm + 400 per missed positive).
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = model_cells[name]
    add("voting_cost_%s" % name, money_cost(fp, fn))

# ---- 4.3 Two rounds of AdaBoost by hand ------------------------------------------------------------
y_ab = [+1, +1, +1, -1, -1, -1, +1, -1]
h1 = [+1, +1, -1, -1, -1, -1, -1, -1]
h2 = [-1, +1, +1, -1, +1, -1, +1, -1]
weights = [1.0 / 8.0] * 8
ada_rows = []
alphas = []
for rnd, h in ((1, h1), (2, h2)):
    eps = sum(wt for wt, yy, hh in zip(weights, y_ab, h) if hh != yy)
    alpha = 0.5 * math.log((1.0 - eps) / eps)
    unnorm = [wt * math.exp(-alpha * yy * hh) for wt, yy, hh in zip(weights, y_ab, h)]
    total = sum(unnorm)
    after = [u / total for u in unnorm]
    for i in range(8):
        ada_rows.append([rnd, i + 1, weights[i], unnorm[i], after[i]])
    add("round%d_error" % rnd, eps)
    add("round%d_alpha" % rnd, alpha)
    alphas.append(alpha)
    weights = after
write_csv("part4_adaboost.csv",
          ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
correct = 0
for i in range(8):
    score = alphas[0] * h1[i] + alphas[1] * h2[i]
    sign = 1 if score > 0 else (-1 if score < 0 else 0)
    if sign == y_ab[i]:
        correct += 1
add("ensemble_accuracy", correct / 8.0)

# ---- 4.4 Out-of-bag share ---------------------------------------------------------------------------
add("oob_fraction", (1.0 - 1.0 / n_tr) ** n_tr)


# ---- 4.5 Size the neural networks a later project would need (counting only) ------------------------
def dense_params(n_in, n_out):
    return n_in * n_out + n_out


def conv_params(k_h, k_w, c_in, n_filters):
    return k_h * k_w * c_in * n_filters + n_filters


def out_side(size, filt, pad, stride):
    return (size - filt + 2 * pad) // stride + 1


# Model A, option 1: 784 -> 16 -> 16 -> 10
mlp = [("mlp_hidden1", dense_params(28 * 28 * 1, 16)),
       ("mlp_hidden2", dense_params(16, 16)),
       ("mlp_output", dense_params(16, 10))]
# Model A, option 2: conv(5x5, 6, pad 2) -> pool -> conv(5x5, 16, pad 0) -> pool -> FC 32 -> FC 10
s1 = out_side(28, 5, 2, 1)       # 28
p1 = out_side(s1, 2, 0, 2)       # 14
s2 = out_side(p1, 5, 0, 1)       # 10
p2 = out_side(s2, 2, 0, 2)       # 5
flat = p2 * p2 * 16              # 400
cnn = [("cnn_conv1", conv_params(5, 5, 1, 6)),
       ("cnn_conv2", conv_params(5, 5, 6, 16)),
       ("cnn_dense1", dense_params(flat, 32)),
       ("cnn_dense2", dense_params(32, 10))]
write_csv("part4_architecture.csv", ["layer", "parameters"], [[k, v] for k, v in mlp + cnn])
add("mlp_total", sum(v for _, v in mlp))
add("cnn_total", sum(v for _, v in cnn))
add("fc_layer_params", dense_params(128 * 128 * 3, 1024))
add("conv_layer_params", conv_params(3, 3, 3, 64))
# Extra keys (not required by the brief): Model A option 2 output shapes (height x width x depth).
add("cnn_shape_conv1", "%dx%dx6" % (s1, s1))
add("cnn_shape_pool1", "%dx%dx6" % (p1, p1))
add("cnn_shape_conv2", "%dx%dx16" % (s2, s2))
add("cnn_shape_pool2", "%dx%dx16" % (p2, p2))
add("cnn_flatten_size", flat)
# Model B: W_xh (128 x 64), W_hh (128 x 128), W_hy (16 x 128), b_h (128), b_y (16)
add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)

write_results("part4_results.txt", out)
print("part4 done: %d keys written to outputs/part4_results.txt" % len(out))
print("CNN shapes: conv1 %dx%dx6, pool1 %dx%dx6, conv2 %dx%dx16, pool2 %dx%dx16, flatten %d, best K %d"
      % (s1, s1, p1, p1, s2, s2, p2, p2, flat, best_k))
