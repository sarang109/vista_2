"""Risk models the receiver might deploy. Every model exposes fit(X, y) and risk(X) (higher = riskier).

lightgbm   gradient-boosted trees; monotone constraints through `monotone_constraints`
xgboost    same, through XGBoost's `monotone_constraints`
mlp        scikit-learn MLP on standardised inputs (no monotone support)
mono_mlp   a small network that is monotone by construction in the chosen inputs: one hidden layer of
           ReLU units whose weights on monotone inputs are non-negative (after sign flipping) and whose
           output weights are non-negative, so the score cannot fall when a constrained input rises
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class LGBModel:
    kind = "lightgbm"

    def __init__(self, monotone: dict | None = None, seed: int = 0, rounds: int = 400, lr: float = 0.05,
                 leaves: int = 63, threads: int = 0, **extra):
        self.monotone = monotone or {}
        self.seed, self.rounds = seed, rounds
        self.params = {"objective": "binary", "learning_rate": lr, "num_leaves": leaves,
                       "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": seed,
                       "num_threads": threads, "deterministic": True, "force_row_wise": True}
        self.params.update(extra)

    def fit(self, X: pd.DataFrame, y, Xv=None, yv=None):
        import lightgbm as lgb
        self.columns = list(X.columns)
        params = dict(self.params)
        if self.monotone:
            params["monotone_constraints"] = [int(self.monotone.get(c, 0)) for c in self.columns]
            params["monotone_constraints_method"] = "advanced"
        ds = lgb.Dataset(X.to_numpy(np.float32), label=y, free_raw_data=True)
        cb, valid = [], []
        if Xv is not None:
            valid = [lgb.Dataset(Xv[self.columns].to_numpy(np.float32), label=yv, reference=ds)]
            cb = [lgb.early_stopping(50, verbose=False)]
        self.booster = lgb.train(params, ds, num_boost_round=self.rounds, valid_sets=valid, callbacks=cb)
        return self

    def raw(self, X) -> np.ndarray:
        A = X[self.columns].to_numpy(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
        return self.booster.predict(A, raw_score=True)

    def risk(self, X) -> np.ndarray:
        return _sigmoid(self.raw(X))


class XGBModel:
    kind = "xgboost"

    def __init__(self, monotone: dict | None = None, seed: int = 0, rounds: int = 400, lr: float = 0.05,
                 depth: int = 7, threads: int = 0):
        self.monotone = monotone or {}
        self.seed, self.rounds = seed, rounds
        self.params = {"objective": "binary:logistic", "eta": lr, "max_depth": depth, "subsample": 0.8,
                       "colsample_bytree": 0.8, "min_child_weight": 5, "tree_method": "hist", "seed": seed,
                       "nthread": threads or -1, "eval_metric": "auc"}

    def fit(self, X: pd.DataFrame, y, Xv=None, yv=None):
        import xgboost as xgb
        self.columns = list(X.columns)
        params = dict(self.params)
        if self.monotone:
            params["monotone_constraints"] = "(" + ",".join(str(int(self.monotone.get(c, 0))) for c in self.columns) + ")"
        d = xgb.DMatrix(X.to_numpy(np.float32), label=y)
        evals, es = [], None
        if Xv is not None:
            evals = [(xgb.DMatrix(Xv[self.columns].to_numpy(np.float32), label=yv), "val")]
            es = 50
        self.booster = xgb.train(params, d, self.rounds, evals=evals, early_stopping_rounds=es, verbose_eval=False)
        return self

    def risk(self, X) -> np.ndarray:
        import xgboost as xgb
        A = X[self.columns].to_numpy(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
        it = getattr(self.booster, "best_iteration", None)
        rng = (0, it + 1) if it is not None else (0, 0)
        return self.booster.predict(xgb.DMatrix(A), iteration_range=rng)


class _Scaler:
    def fit(self, A):
        self.mu = A.mean(0)
        self.sd = A.std(0) + 1e-6
        return self

    def __call__(self, A):
        return (A - self.mu) / self.sd


class MLPModel:
    kind = "mlp"

    def __init__(self, monotone: dict | None = None, seed: int = 0, hidden=(64, 32), epochs: int = 30):
        if monotone:
            raise ValueError("the plain MLP has no monotone constraints; use mono_mlp")
        self.seed, self.hidden, self.epochs = seed, hidden, epochs

    def fit(self, X: pd.DataFrame, y, Xv=None, yv=None):
        from sklearn.neural_network import MLPClassifier
        self.columns = list(X.columns)
        A = _transform(X.to_numpy(np.float64))
        self.scale = _Scaler().fit(A)
        self.clf = MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=self.epochs, early_stopping=True,
                                 random_state=self.seed, batch_size=512, learning_rate_init=1e-3, alpha=1e-4)
        self.clf.fit(self.scale(A), y)
        return self

    def risk(self, X) -> np.ndarray:
        A = X[self.columns].to_numpy(np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float64)
        return self.clf.predict_proba(self.scale(_transform(A)))[:, 1]


def _transform(A):
    """Signed log for heavy-tailed columns; leaves small integer codes nearly unchanged."""
    return np.sign(A) * np.log1p(np.abs(A))


class MonoMLP:
    """score = b + v . relu(U z + c), z = standardised inputs with monotone ones sign-flipped so that every
    constrained direction is "non-decreasing risk"; U[:, constrained] >= 0 and v >= 0 are enforced by
    projection after every step. Trained with Adam on the logistic loss."""
    kind = "mono_mlp"

    def __init__(self, monotone: dict | None = None, seed: int = 0, hidden: int = 64, epochs: int = 12,
                 lr: float = 3e-3, batch: int = 1024):
        self.monotone = monotone or {}
        self.seed, self.hidden, self.epochs, self.lr, self.batch = seed, hidden, epochs, lr, batch

    def _z(self, A):
        return self.scale(_transform(A)) * self.sign

    def fit(self, X: pd.DataFrame, y, Xv=None, yv=None):
        rng = np.random.default_rng(self.seed)
        self.columns = list(X.columns)
        A = X.to_numpy(np.float64)
        self.scale = _Scaler().fit(_transform(A))
        self.sign = np.array([float(self.monotone.get(c, 0)) or 1.0 for c in self.columns])
        self.mask = np.array([c in self.monotone and self.monotone[c] != 0 for c in self.columns])
        Z = self._z(A)
        y = np.asarray(y, dtype=float)
        n, p = Z.shape
        h = self.hidden
        U = rng.normal(0, 1 / np.sqrt(p), (p, h))
        U[self.mask] = np.abs(U[self.mask])
        c = np.zeros(h)
        v = np.abs(rng.normal(0, 1 / np.sqrt(h), h))
        b = np.log(y.mean() / (1 - y.mean()))
        params = [U, c, v, np.array([b])]
        m = [np.zeros_like(q) for q in params]
        s = [np.zeros_like(q) for q in params]
        t = 0
        pos_w = 1.0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for k in range(0, n, self.batch):
                idx = order[k:k + self.batch]
                zb, yb = Z[idx], y[idx]
                H = zb @ params[0] + params[1]
                R = np.maximum(H, 0)
                out = R @ params[2] + params[3][0]
                g = (_sigmoid(out) - yb) * np.where(yb > 0, pos_w, 1.0) / len(idx)
                gv = R.T @ g
                gb = np.array([g.sum()])
                gH = np.outer(g, params[2]) * (H > 0)
                gU = zb.T @ gH + 1e-4 * params[0]
                gc = gH.sum(0)
                t += 1
                for j, gr in enumerate((gU, gc, gv, gb)):
                    m[j] = 0.9 * m[j] + 0.1 * gr
                    s[j] = 0.999 * s[j] + 0.001 * gr * gr
                    params[j] = params[j] - self.lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(s[j] / (1 - 0.999 ** t)) + 1e-8)
                params[0][self.mask] = np.maximum(params[0][self.mask], 0.0)
                params[2] = np.maximum(params[2], 0.0)
        self.U, self.c, self.v, self.b = params[0], params[1], params[2], params[3][0]
        return self

    def risk(self, X) -> np.ndarray:
        A = X[self.columns].to_numpy(np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float64)
        Z = self._z(A)
        return _sigmoid(np.maximum(Z @ self.U + self.c, 0) @ self.v + self.b)


LEARNERS = {"lightgbm": LGBModel, "xgboost": XGBModel, "mlp": MLPModel, "mono_mlp": MonoMLP}


def make(kind: str, monotone: dict | None = None, seed: int = 0, **kw):
    return LEARNERS[kind](monotone=monotone, seed=seed, **kw)
