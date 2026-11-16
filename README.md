Domain: E-commerce

## How to run

Python 3.10 or newer and NumPy are the only requirements (developed on Python 3.13 with NumPy 2.2.4, pinned in `requirements.txt`). Everything runs offline. From the root of a clean clone of this repository, run these commands in this order:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

- `python generate_data.py` is the provided script, committed verbatim. It writes `data/records.csv` and prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23` (rows, columns, churned records, MD5 of the file).
- `part1.py` to `part4.py` each regenerate their own files in `outputs/`. Shared helpers (loading, the pinned 560/240 split, logistic regression, KNN, metrics, output formatting) are in `common.py`; importing it never runs a Part. Parts 2 and 4 use the same split, standardisation, logistic model and KNN code through `common.py`.
- Running the whole sequence twice from an empty `outputs/` folder gives byte-identical files.
- All numbers below are quoted from the committed `outputs/` files, apart from a few simple values worked out directly from them (differences, halves and conversions of the column's hundreds of INR into INR). Money is in Indian Rupees (INR). The data are synthetic: no real Flipkart data is used.
- A few extra keys beyond those required by the brief are written for traceability of the README: `wcss_drop_*` (part3), `voting_cost_*` and `cnn_shape_*`, `cnn_flatten_size` (part4).

## Part 1 findings

**R1.1 — The planning claim is rejected.** The 800 order values have `mean_value` = 8.9382 (INR 894) with `se_value` = 0.3243, so the 95% interval is 8.3026 to 9.5738. Testing H0: μ = 8.0 gives `z_stat` = 2.8930, which is beyond the critical value 1.96, so `reject_h0` = yes. A Type I error here would be concluding that the true average order value differs from INR 800 and rewriting the category plan when, in fact, the average is still INR 800. The gap `mean_value` − 8.0 is 0.9382 column units, about INR 94 or roughly 12% above the plan. Statistically the gap is clear (even the lower end of the interval, 8.3026, is above 8.0). Practically it is a moderate planning error: on a large number of orders a 12% understatement of revenue per order is material, but it is small next to the spread of individual orders (`sd_value` = 9.1723, so the gap is about a tenth of one standard deviation), and, because the interval for the gap runs from 0.3026 to 1.5738 units, the planners should treat the plan as too low by somewhere between about INR 30 and INR 157 rather than by precisely INR 94.

**R1.2 — Sample size.** To estimate the average within ±0.5 units (INR 50) at 95% confidence, `n_required` = 1293 orders are needed, against 800 available. The 800 rows give a margin of 1.96 × 0.3243 ≈ 0.64 units (INR 64), so the planning team needs about 493 more orders (roughly 60% more data) to reach the ±INR 50 precision.

**R1.3 — Gradient descent versus the closed form.** `gd_slope_gap` = 0.0000 (limit 0.001): gradient descent and the formula give the same line (slope 0.6701, intercept −3.6598). The cost is a smooth bowl in (m, c) with a single minimum that is exactly the least-squares line, and with standardised X4 and η = 0.1, 500 updates are far more than needed (the cost is 40.3698 after 100 updates and still 40.3698 after 500).

**R1.4 — Funnel and log remedy.** `resid_sd_ratio_raw` = 3.7525: the residual spread in the upper half of fitted values (8.6891) is almost four times that of the lower half (2.3155), so the raw model shows a clear funnel. After taking logs, `resid_sd_ratio_log` = 1.0396 (0.5369 over 0.5165), so the log model removes the funnel: the spread is essentially constant.

**R1.5 — Multiple regression.** The predictors with |coefficient| above 0.05 in the Task 1.7 model (excluding the constant) are ln(X4) (`mr_beta_1` = 1.1576), X6, the cash-on-delivery flag (`mr_beta_6` = 0.1518), X3, the discount share (`mr_beta_4` = 0.1067) and X2, the complaint count (`mr_beta_3` = 0.0940). The others fall below the cut: X1 (−0.0074), X5 (0.0028) and X7 (0.0172). This is only a mechanical rule on coefficients in different units (a per-month, per-point, per-ratio, per-count and per-flag step are not comparable), so it is not a list of the predictors that matter: tenure, for example, is measured per month, so a small per-month coefficient says little about its importance. The extra predictors do earn their place: `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, an increase of 0.0842 even after adjusted R² penalises the six extra predictors.

## Part 2 findings

**R2.1 — Learning rates.** After 200 updates the losses from lowest to highest are: η = 1.0 (0.3703), η = 0.1 (0.3776) and η = 0.01 (0.5091). A learning rate that is too small (0.01) has barely moved from the starting loss of 0.6931 and would need many more updates, while a larger rate reaches the minimum much faster (η = 1.0 was fastest and still stable here on standardised features, although a rate that is too large can overshoot and oscillate).

**R2.2 — Largest weights.** The three largest absolute weights are X3 `discount_order_share` (+0.9013), X1 `customer_tenure_months` (−0.5554) and X2 `complaint_count_90d` (+0.5505). The largest odds ratio is `odds_ratio_X3` = 2.4629: a customer whose share of discounted orders is one training standard deviation higher has about 2.46 times the odds of churning, other predictors held fixed.

**R2.3 — Accuracy is a weak score.** `blind_accuracy` = 0.7875 (predict "nobody churns") against `tau_0.5_accuracy` = 0.8042: the model is only 0.0167 better than a rule that never catches a single churner, because four customers in five do not churn. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (true positives 24 to 38 of 51 churners) and lowered precision from 0.5455 to 0.4634 (false alarms 20 to 44). Accuracy even fell to 0.7625, below the blind rule, although the model became much more useful at finding churners.

**R2.4 — Logistic versus KNN.** At τ = 0.3 the logistic model has `tau_0.3_f1` = 0.5714 and `tau_0.3_recall` = 0.7451. KNN (best K = 5 by cross-validation) has `knn_f1` = 0.3696 and `knn_recall` = 0.3333. The logistic model is better on both.

**R2.5 — What to deploy.** I would deploy the logistic model with the threshold chosen by the cost rule, `cost_tau` = 0.15. On the test rows it costs `cost_test_at_tau` = INR 6,050, against `cost_test_at_0.5` = INR 11,800 at τ = 0.5 and `cost_test_flag_nobody` = INR 20,400 if nobody is flagged. The threshold was chosen on the training rows because the test rows must stay unseen: a threshold tuned on them would make the test cost an optimistic estimate of what will happen on future customers.

## Part 3 findings

**R3.1 — Elbow.** The pinned K-means gives WCSS drops of WCSS(2) − WCSS(3) = 659.8282, WCSS(3) − WCSS(4) = 108.0739, WCSS(4) − WCSS(5) = 171.0575 and WCSS(5) − WCSS(6) = 45.5466. The elbow rule gives `elbow_k` = 3: the drop from 3 to 4 (108.0739) is below half of the drop from 2 to 3 (329.9141), so a fourth cluster adds little. The drops do not shrink steadily (4 to 5 is larger than 3 to 4), because the pinned starting rows give one reproducible solution, not necessarily the global optimum for every K. K = 3 is used throughout.

**R3.2 — Segments** (from `outputs/part3_segments.csv`; X4 and `mean_value` are in hundreds of INR):

| Segment | Name | Size | Profile | outcome_rate | Action |
|---|---|---|---|---|---|
| A | Steady loyalists | 322 | longest tenure (47.5 months), fewest complaints (1.95), lowest discount share (0.2250), loyalty 62.2, `mean_value` 3.5976 | 0.0186 | Do not spend coupons here; keep them with non-price perks such as early access for loyalty members. |
| B | High-value, complaint-heavy shoppers | 255 | biggest baskets (X4 28.7192), most complaints (6.07), `mean_value` 17.1248 (the highest) | 0.2353 | Proactive service recovery: contact them after a complaint, and give a retention coupon when the churn model flags them, since losing one is costly. |
| C | Bargain-focused newcomers | 223 | shortest tenure (14.4 months), highest discount share (0.7118), lowest loyalty (38.5) | 0.4664 | Targeted retention coupon plus an onboarding nudge towards the loyalty tier, while capping blanket discounts. |

The segment with the highest `outcome_rate` is C, at 0.4664 (almost half of its customers churn). The segment sizes add up to 800. The data has no date column, so a seasonal segment cannot be identified and I did not use that label.

**R3.3 — Hierarchical clustering.** Single linkage cut (`single_clusters_at_cut` = 4): {E2, E3, E6}, {E1}, {E4} and {E5}. Complete linkage cut (`complete_clusters_at_cut` = 2): {E1, E4, E5} and {E2, E3, E6} (average linkage also gives 2). Single linkage uses the closest pair, so heights creep up in small steps (1.3, 1.4, 2.8, 2.9, 3.4), the biggest gap comes early (1.4 to 2.8) and the cut stops while only one small group has formed. Complete linkage uses the farthest pair, so the last merge of the two blocks jumps to 7.9 (from 2.9), which shows two well-separated groups.

**R3.4 — PCA.** After standardising, the mean vector is zero in every column (up to floating-point rounding), so centring changes nothing. PC1 (`explained_ratio_1` = 0.5227) contrasts tenured, loyal customers (X1 +0.5317, X5 +0.4418) with discount-heavy, complaint-prone, bigger-basket customers (X3 −0.5187, X2 −0.4100, X4 −0.2916). `components_for_80pct` = 3: two components keep 0.7593 of the variance and three keep 0.8616. The segment means on PC1 are A = 1.7773, B = −0.7831 and C = −1.6709: A is far from the other two, and the ordering is the reverse of the churn rates (0.0186, 0.2353, 0.4664), so PC1 is essentially a "stable loyalist to discount-driven newcomer" axis; B and C are much closer together on PC1, and the second component (loading mostly on X4 +0.6934 and X2 +0.5056, where B is higher than C in `part3_segments.csv`) is what would separate them.

## Part 4 findings

**R4.1 — Best split.** S4 (tenure ≤ 24 months) has the highest information gain (0.1252) and Gini gain (0.0606), ahead of S3 (0.1046 and 0.0494), because it gives the purest children: 86 of 188 short-tenure customers churn (0.4574) against only 33 of 372 longer-tenure customers (0.0887). A tree grown until every leaf is pure would memorise the training customers, noise included, so its training error would be near zero while it would generalise poorly. One rule that stops it from growing too deep is a maximum depth (a minimum number of records per leaf or a minimum gain per split would also work).

**R4.2 — Model to ship.** I would ship the logistic model (`f1` 0.5714, `recall` 0.7451 at τ = 0.3). The stump has `f1` 0.5532 and `recall` 0.7647, KNN `f1` 0.3696 and `recall` 0.3333, hard vote `f1` 0.5344 and `recall` 0.6863, soft vote `f1` 0.5507 and `recall` 0.7451. Combining models did not help: neither vote beats the logistic model alone, because the weak KNN pulls them down. The one-question stump coming so close shows that tenure carries much of the signal.

**R4.3 — AdaBoost.** After round 1 normalisation the two misclassified points (3 and 7) carry 0.2500 + 0.2500 = 0.5000 of the total weight. Round 2 trusts h2 more because h2 makes a smaller weighted error on the reweighted data (0.1667, against 0.2500 for h1 in round 1), so α2 = 0.8047 exceeds α1 = 0.5493, and h2 gets the two heavily weighted points right. The ensemble classifies 6 of the 8 points correctly (`ensemble_accuracy` = 0.7500).

**R4.4 — Architecture.** Model A, option 2 (28 × 28 × 1 input, ten classes):

| Layer | Output shape (height × width × depth) | Parameters |
|---|---|---|
| input | 28 × 28 × 1 | 0 |
| `cnn_conv1` (5 × 5, 6 filters, stride 1, padding 2) + ReLU | 28 × 28 × 6 | 156 |
| max-pool 2 × 2, stride 2 | 14 × 14 × 6 | 0 |
| `cnn_conv2` (5 × 5, 16 filters, stride 1, padding 0) + ReLU | 10 × 10 × 16 | 2,416 |
| max-pool 2 × 2, stride 2 | 5 × 5 × 16 | 0 |
| flatten | 400 | 0 |
| `cnn_dense1` (32 neurons) + ReLU | 32 | 12,832 |
| `cnn_dense2` (10 neurons) | 10 | 330 |
| **Total** (`cnn_total`) | | **15,734** |

Option 1, the fully connected 784 → 16 → 16 → 10 network, has 12,560 + 272 + 170 = 13,002 parameters (`mlp_total`). Convolution scales because filters are shared across positions: on a 128 × 128 × 3 input a fully connected layer of 1,024 neurons needs `fc_layer_params` = 50,332,672 parameters, while a convolutional layer of 64 filters of 3 × 3 needs only `conv_layer_params` = 1,792.

Model B (recurrent network): W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128, plus b_h (128) and b_y (16), giving `rnn_total` = 26,768. It does not grow with longer sequences because the same weights are reused at every time step; a longer review only means more repetitions of the same computation.

For 10 classes I would use ReLU in the hidden layers and a softmax output with categorical cross-entropy loss (the classes are mutually exclusive and softmax gives probabilities that sum to 1).

In a very deep plain network the gradient reaching an early layer is a product of many per-layer factors that are often smaller than 1, so it shrinks towards zero and the early layers stop learning. A skip connection adds a layer's input to its output, which gives the gradient a direct path back through the network that does not pass through that shrinking product.

## Model-choice memo

**To:** Head of the Retention Team

**Recommendation.** Put the logistic-regression churn model into production and flag a customer when predicted churn probability is at least 0.15.

**Why this model.** On the 240 held-out customers at threshold 0.3, it scores F1 0.5714 and recall 0.7451, against F1 0.3696 and recall 0.3333 for nearest neighbours. Neither voting scheme beat it (F1 0.5344 and 0.5507), so combining models only added complexity. A one-question rule on tenure (24 months or less) reached F1 0.5532, a useful sanity check.

**Threshold and cost.** The threshold 0.15 was chosen on training customers only, using the cost rule INR 50 per unneeded coupon plus INR 400 per lost customer. On the test customers it costs INR 6,050. The same logistic model costs INR 11,800 at 0.5, INR 7,400 at 0.3, and INR 20,400 if nobody is flagged. The logistic model is therefore both the recommendation and the comparator; the other four candidate models cost INR 7,350 to INR 14,800 at their stated rules (same cost rule, computed by me).

**Risk.** Only 51 test customers actually churned, so one missed customer moves the cost by INR 400; the result comes from a single split of synthetic data, and the INR 50 and INR 400 figures are rough estimates. Re-check the threshold on live data after launch.

**Models A and B.** Neither is worth building before the structured model is in production, because no picture or review-text data is in hand and the structured model already answers the churn question.

## Limitations

1. **Synthetic data.** Every row comes from the generator script, so the relationships (and the three hidden groups the clustering recovers) are built in; nothing here says anything about real customers, products or policies.
2. **A single split and few positives.** All test results come from one stratified 70/30 split with only 51 churners in the test rows, so rates move by about 0.02 per customer and a model difference of one or two customers (such as logistic against stump) is within noise. The best K = 5 and the threshold 0.15 are valid for this split only; the cross-validated F1 values are noisy (K = 1 and K = 3 tie at 0.4249).
3. **The cost figures are assumptions.** INR 50 per coupon and INR 400 per lost customer are rough and treated as constant across customers, although segment B customers spend far more per order than segment A. The threshold is chosen on training probabilities of a model fitted to the same rows, so the training cost is optimistic.
4. **Pinned K-means starting rows.** The starting rows give one reproducible solution that is not necessarily the global optimum, and K = 3 was chosen by an elbow rule that ignores other criteria; the hierarchical clustering uses the dissimilarity table supplied in the brief, not distances computed from the 800 records.
5. **Simple tests and models.** The z-test assumes the sample standard deviation can stand in for the population one and the order values are skewed (the standard deviation is larger than the mean), so a log-scale or median-based view could tell a different practical story. There is no time column, so seasonality and shopper behaviour over time cannot be studied, and the models are one-shot rather than monitored on live data.
