"""Shared helpers for part1.py ... part4.py.

Importing this module never runs a Part: it only defines functions and constants.
Everything here is written from scratch with plain Python and NumPy array arithmetic
(no ready-made regression, logistic, KNN, k-means, PCA or metric routines).
"""
import csv
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "records.csv")
OUT_DIR = os.path.join(ROOT, "outputs")

# Money rule of Task 2.8 (INR): a false alarm costs 50, a missed positive costs 400.
COST_FALSE_ALARM = 50
COST_MISSED_POSITIVE = 400


# --------------------------------------------------------------------------------------
# Reading data
# --------------------------------------------------------------------------------------
def load_records():
    """Return X (800 x 7: X1..X7 in file order), y (outcome, 0/1) and value (last column)."""
    with open(DATA_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = list(reader)
    X = np.array([[float(v) for v in r[1:8]] for r in rows], dtype=float)
    y = np.array([int(r[8]) for r in rows], dtype=int)
    value = np.array([float(r[9]) for r in rows], dtype=float)
    return X, y, value


# --------------------------------------------------------------------------------------
# Writing outputs (text files, "\n" line endings on every platform)
# --------------------------------------------------------------------------------------
def fmt(v):
    """bool -> yes/no, int -> integer, float -> 4 decimals, anything else -> text."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = "%.4f" % float(v)
        return "0.0000" if s == "-0.0000" else s
    return str(v)


def write_results(filename, pairs):
    """pairs: list of (key, value); writes outputs/<filename> as 'key = value' lines."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w", newline="\n") as f:
        for key, value in pairs:
            f.write("%s = %s\n" % (key, fmt(value)))


def write_csv(filename, header, rows):
    """header: list of column names; rows: list of lists (formatted with fmt)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(fmt(v) for v in row) + "\n")


# --------------------------------------------------------------------------------------
# Basic statistics (sample form, divisor n - 1)
# --------------------------------------------------------------------------------------
def mean_of(a):
    a = np.asarray(a, dtype=float)
    return float(np.sum(a)) / a.shape[0]


def sample_var(a):
    a = np.asarray(a, dtype=float)
    m = mean_of(a)
    return float(np.sum((a - m) ** 2)) / (a.shape[0] - 1)


def sample_sd(a):
    return math.sqrt(sample_var(a))


def sample_cov(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum((a - mean_of(a)) * (b - mean_of(b)))) / (a.shape[0] - 1)


def column_mean_sd(M):
    """Per-column mean and sample SD of a 2-D array."""
    n = M.shape[0]
    mu = np.sum(M, axis=0) / n
    sd = np.sqrt(np.sum((M - mu) ** 2, axis=0) / (n - 1))
    return mu, sd


# --------------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------------
def safe_div(a, b):
    return a / b if b != 0 else 0.0


def confusion(y_true, y_pred):
    """Return (tp, fp, tn, fn) for 0/1 arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def metrics_from_cells(tp, fp, tn, fn):
    """accuracy, precision, recall, f1, specificity, fpr (0 when a denominator is 0)."""
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "specificity": specificity, "fpr": fpr}


def money_cost(fp, fn):
    """Task 2.8 cost rule: INR 50 per false alarm plus INR 400 per missed positive."""
    return COST_FALSE_ALARM * fp + COST_MISSED_POSITIVE * fn


# --------------------------------------------------------------------------------------
# Task 2.1: stratified split and standardisation (shared by Parts 2 and 4)
# --------------------------------------------------------------------------------------
def stratified_split_indices(y):
    """First floor(7*count/10) rows of each class (file order) -> train, the rest -> test."""
    train, test = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = (7 * len(idx)) // 10
        train.extend(idx[:k].tolist())
        test.extend(idx[k:].tolist())
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


class SplitData:
    """Train/test arrays in original and standardised units (training statistics only)."""

    def __init__(self):
        X, y, value = load_records()
        self.train_idx, self.test_idx = stratified_split_indices(y)
        self.X_train = X[self.train_idx]
        self.X_test = X[self.test_idx]
        self.y_train = y[self.train_idx]
        self.y_test = y[self.test_idx]
        self.mean, self.sd = column_mean_sd(self.X_train)
        self.Z_train = (self.X_train - self.mean) / self.sd
        self.Z_test = (self.X_test - self.mean) / self.sd


# --------------------------------------------------------------------------------------
# Task 2.2 / 2.3: logistic regression by batch gradient descent
# --------------------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce_from_logits(z, y):
    """Mean binary cross-entropy -(y ln s + (1-y) ln(1-s)), s = sigmoid(z), in a stable form.

    ln(1 + e^z) - y z equals -(y ln s + (1 - y) ln (1 - s)) exactly.
    """
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def train_logistic(Z, y, eta, n_updates, record_at=()):
    """Start at zero, run n_updates full-batch updates. Returns weights, bias and the
    training loss after each requested number of updates (0 = before any update)."""
    n, p = Z.shape
    w = np.zeros(p)
    b = 0.0
    losses = {}
    for t in range(n_updates + 1):
        z = Z @ w + b
        if t in record_at:
            losses[t] = bce_from_logits(z, y)
        if t == n_updates:
            break
        err = sigmoid(z) - y
        w = w - eta * (Z.T @ err) / n
        b = b - eta * float(np.sum(err)) / n
    return w, b, losses


def predict_proba(Z, w, b):
    return sigmoid(Z @ w + b)


def fit_main_logistic(data):
    """The Task 2.2 model: eta = 0.1, 1000 updates on the 560 training rows."""
    return train_logistic(data.Z_train, data.y_train, 0.1, 1000, record_at=(0, 1, 10, 100, 1000))


# --------------------------------------------------------------------------------------
# Task 2.7: K-nearest neighbours (Euclidean, stable tie-break by file order)
# --------------------------------------------------------------------------------------
def distance_matrix(Q, R):
    """Euclidean distances, shape (len(Q), len(R)), by explicit differences."""
    diff = Q[:, None, :] - R[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def knn_positive_counts(Q, R, yR, ks):
    """For each K in ks, the number of positive labels among the K nearest reference rows
    of every query row. Neighbours are ranked by distance, ties by position in R."""
    D = distance_matrix(Q, R)
    order = np.argsort(D, axis=1, kind="stable")
    k_max = max(ks)
    labels = yR[order[:, :k_max]]
    cum = np.cumsum(labels, axis=1)
    return {k: cum[:, k - 1] for k in ks}


def knn_predict(Q, R, yR, k):
    """Return (predicted class, share of positive neighbours) for K = k (odd)."""
    counts = knn_positive_counts(Q, R, yR, [k])[k]
    pred = (counts >= (k + 1) / 2.0).astype(int)
    share = counts / float(k)
    return pred, share


KNN_KS = list(range(1, 22, 2))  # 1, 3, ..., 21


def knn_cross_validate(Z_train, y_train, ks=KNN_KS, n_folds=5):
    """Mean over folds of the held-out F1, for each K. Fold of a row = position mod 5.
    Uses the already standardised training features: nothing is re-standardised."""
    n = Z_train.shape[0]
    fold = np.arange(n) % n_folds
    f1_sum = {k: 0.0 for k in ks}
    for f in range(n_folds):
        q = np.where(fold == f)[0]
        r = np.where(fold != f)[0]
        counts = knn_positive_counts(Z_train[q], Z_train[r], y_train[r], ks)
        for k in ks:
            pred = (counts[k] >= (k + 1) / 2.0).astype(int)
            tp, fp, tn, fn = confusion(y_train[q], pred)
            f1_sum[k] += metrics_from_cells(tp, fp, tn, fn)["f1"]
    return [(k, f1_sum[k] / n_folds) for k in ks]


def choose_k(cv_scores):
    """Highest mean F1; ties go to the smaller K (ascending scan, strict improvement)."""
    best_k, best = None, -1.0
    for k, score in cv_scores:
        if score > best + 1e-12:
            best_k, best = k, score
    return best_k
