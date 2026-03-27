# coding: utf-8
"""Cox proportional hazards survival analysis with LightGBM.

Demonstrates the built-in ``cox`` objective for right-censored survival data.
Labels encode censoring via sign: +t means an event at time t, -t means
censored at time t.
"""

import numpy as np

import lightgbm as lgb

rng = np.random.RandomState(42)
n, p = 2000, 10
X = rng.randn(n, p)

# True log-hazard ratio: f(x) = x0 + 0.5*x1 - 0.3*x2
log_hazard = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]

# Exponential survival times (higher hazard -> shorter time)
event_times = rng.exponential(np.exp(-log_hazard))

# Random censoring (~30% censored)
censor_times = rng.exponential(np.median(event_times) / 0.3, n)
observed = event_times <= censor_times
times = np.minimum(event_times, censor_times)

# Encode labels: positive = event, negative = censored
y = np.where(observed, times, -times)

n_train = int(0.8 * n)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "cox",
    "metric": ["cox_nll", "concordance_index"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": 0,
}

evals_result = {}
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_val],
    valid_names=["val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.record_evaluation(evals_result),
    ],
)

print(f"Validation cox_nll:          {evals_result['val']['cox_nll'][gbm.best_iteration - 1]:.4f}")
print(f"Validation concordance index: {evals_result['val']['concordance_index'][gbm.best_iteration - 1]:.4f}")

# Predictions are log-hazard ratios (higher = more risk)
preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print(f"\nPrediction range: [{preds.min():.3f}, {preds.max():.3f}]")
