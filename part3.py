"""Part 3 - Segmentation and dimensionality reduction (all 800 rows, X1..X5).

Run from anywhere:  python part3.py
Writes outputs/part3_results.txt, outputs/part3_segments.csv, outputs/part3_hierarchical.csv
"""
import numpy as np

from common import column_mean_sd, load_records, write_csv, write_results

X_all, outcome, value = load_records()
X5 = X_all[:, :5]  # X1..X5, the continuous predictors
n = X5.shape[0]
mu, sd = column_mean_sd(X5)
Z = (X5 - mu) / sd  # standardised 800 x 5 matrix
out = []


def add(key, val):
    out.append((key, val))


# ---- 3.1 K-means from scratch with pinned starting rows --------------------------------
def kmeans(Zm, k):
    """Pinned K-means. Returns (labels, centroids, wcss, number of assignment passes)."""
    step = Zm.shape[0] // k
    centroids = Zm[[j * step for j in range(k)]].copy()
    labels = None
    passes = 0
    while True:
        diff = Zm[:, None, :] - centroids[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        new_labels = np.argmin(sq, axis=1)  # ties -> lower-numbered centroid
        passes += 1
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = Zm[labels == j]
            if members.shape[0] > 0:  # an empty cluster keeps its centroid
                centroids[j] = np.sum(members, axis=0) / members.shape[0]
    diff = Zm - centroids[labels]
    wcss = float(np.sum(diff * diff))
    return labels, centroids, wcss, passes


wcss = {}
runs = {}
for k in (2, 3, 4, 5, 6):
    labels_k, cents_k, w_k, passes_k = kmeans(Z, k)
    wcss[k] = w_k
    runs[k] = (labels_k, passes_k)
    add("wcss_k%d" % k, w_k)

# ---- 3.2 Elbow rule -------------------------------------------------------------------------
elbow_k = None
for k in (3, 4, 5):
    if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
        elbow_k = k
        break
add("elbow_k", elbow_k)
# Extra keys (not required by the brief): the successive WCSS drops quoted in the README.
for k in (2, 3, 4, 5):
    add("wcss_drop_%d_to_%d" % (k, k + 1), wcss[k] - wcss[k + 1])

# ---- 3.3 Profile the K = 3 segments (A, B, C by ascending mean X3) ----------------------------
labels3 = runs[3][0]
x3_means = [float(np.sum(X5[labels3 == j, 2])) / int(np.sum(labels3 == j)) for j in range(3)]
cluster_order = sorted(range(3), key=lambda j: x3_means[j])  # A, B, C
seg_names = ["A", "B", "C"]
seg_of_row = np.empty(n, dtype=int)  # 0, 1, 2 for A, B, C
seg_rows = []
for s, j in enumerate(cluster_order):
    mask = labels3 == j
    seg_of_row[mask] = s
    size = int(np.sum(mask))
    means = [float(np.sum(X5[mask, c])) / size for c in range(5)]
    rate = float(np.sum(outcome[mask])) / size
    mval = float(np.sum(value[mask])) / size
    seg_rows.append([seg_names[s], size] + means + [rate, mval])
    add("size_segment_%s" % seg_names[s], size)
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
           "outcome_rate", "mean_value"], seg_rows)

# ---- 3.4 Hierarchical clustering on the supplied dissimilarity table --------------------------
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
DIST = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def linkage_distance(ca, cb, rule):
    """ca, cb: tuples of member indices (0-based). Member-to-member distances from DIST."""
    ds = [DIST[i, j] for i in ca for j in cb]
    if rule == "single":
        return min(ds)
    if rule == "complete":
        return max(ds)
    return sum(ds) / len(ds)  # average


def cluster_label(members):
    return "+".join(names[i] for i in sorted(members))


def agglomerate(rule):
    clusters = [(i,) for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                h = linkage_distance(clusters[a], clusters[b], rule)
                if best is None or h < best[0]:
                    best = (h, a, b)
        h, a, b = best
        ca, cb = clusters[a], clusters[b]
        first, second = (ca, cb) if min(ca) < min(cb) else (cb, ca)
        merges.append((cluster_label(first), cluster_label(second), h))
        clusters = [c for idx, c in enumerate(clusters) if idx not in (a, b)]
        clusters.append(tuple(sorted(ca + cb)))
    return merges


hier_rows = []
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    heights = [m[2] for m in merges]
    for no, (ca, cb, h) in enumerate(merges, start=1):
        hier_rows.append([rule, no, ca, cb, h])
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    biggest = max(range(4), key=lambda i: gaps[i])  # gap between merge i+1 and merge i+2
    merges_below_cut = biggest + 1
    add("%s_clusters_at_cut" % rule, 6 - merges_below_cut)
write_csv("part3_hierarchical.csv",
          ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)

# ---- 3.5 PCA by the seven steps -------------------------------------------------------------------
mean_vec = np.sum(Z, axis=0) / n               # (1) mean vector (zeros after standardising)
Xc = Z - mean_vec                              # (2) centre
S = (Xc.T @ Xc) / (n - 1)                      # (3) covariance matrix
eigvals, eigvecs = np.linalg.eigh(S)           # (4) eigen-decomposition
order = np.argsort(-eigvals, kind="stable")    # (5) largest eigenvalue first
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]
for c in range(5):                             # sign convention: largest |entry| positive
    if eigvecs[np.argmax(np.abs(eigvecs[:, c])), c] < 0:
        eigvecs[:, c] = -eigvecs[:, c]
W = eigvecs[:, :2]                             # (6) keep the top two
proj = Xc @ W                                  # (7) project

total = float(np.sum(eigvals))
add("eigenvalue_sum", total)
for c in range(5):
    add("eigenvalue_%d" % (c + 1), float(eigvals[c]))
ratios = [float(eigvals[c]) / total for c in range(5)]
for c in range(5):
    add("explained_ratio_%d" % (c + 1), ratios[c])
cum = 0.0
cum_list = []
for c in range(5):
    cum += ratios[c]
    cum_list.append(cum)
    add("cumulative_ratio_%d" % (c + 1), cum)
add("components_for_80pct", next(c + 1 for c in range(5) if cum_list[c] >= 0.8))
for c in range(5):
    add("pc1_loading_X%d" % (c + 1), float(eigvecs[c, 0]))
for c in range(5):
    add("pc2_loading_X%d" % (c + 1), float(eigvecs[c, 1]))
for r in range(3):
    add("row%d_pc1" % (r + 1), float(proj[r, 0]))
    add("row%d_pc2" % (r + 1), float(proj[r, 1]))
for s in range(3):
    mask = seg_of_row == s
    add("mean_pc1_segment_%s" % seg_names[s], float(np.sum(proj[mask, 0])) / int(np.sum(mask)))

write_results("part3_results.txt", out)
print("part3 done: %d keys written to outputs/part3_results.txt" % len(out))
print("k-means assignment passes (k=2..6):", [runs[k][1] for k in (2, 3, 4, 5, 6)])
