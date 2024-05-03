# -*- coding: utf-8 -*-

"""customized ridge with a multitude of options
"""

#%%
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator, Iterator, Optional

# Third Party
import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from statsmodels import api as sm
#%%
X = np.random.randn(3, 2)
w = np.random.randn(3)
X.T @ np.diag(w) @ X - np.einsum("ki,k,kj->ij", X, w, X)

#%%
def gen_data():  # generate features and a single targets
    n_features = 7
    truths = (
        (100, np.r_[[1, 10, 3, 0, 0, 0, 0]].T, 1., 1.), # duration of epoch 1, true coefs, bias, variance
        (50, np.r_[[0, 20, -1, 1, 0, 0, 0]].T, -1., .5), # epoch 2
    )
    n_samples = sum([c[0] for c in truths])
    X = make_low_rank_matrix(    
        n_samples=n_samples,
        n_features=n_features,
        effective_rank=2,
        tail_strength=0.1,
        random_state=42,
    )
    y = np.zeros(n_samples, dtype=float)

    ifrom, ito = 0, 0
    coefs = np.zeros((n_samples, n_features), dtype=float)
    biases = np.zeros(n_samples, dtype=float)
    for l, coef, bias, std in truths:
        ito += l
        y[ifrom: ito] = bias + X[ifrom: ito] @ coef + std * np.random.randn(l)
        coefs[ifrom: ito] = coef[None, :]
        biases[ifrom: ito] = bias
        ifrom = ito
    return X, y, coefs, biases

X, y, coefs, biases = gen_data()
#%%
def custom_ts_cv(*, n_samples, train_size, test_size, gap=0) -> Iterator[tuple[slice, slice]]:
    n_steps = np.ceil((n_samples - train_size - test_size - gap) / test_size).astype(int)
    assert n_steps >= 0
    first_test_start = train_size + gap
    test_edges = np.append(first_test_start, np.linspace(first_test_start + test_size, n_samples, n_steps + 1).astype(int))
    for test_start, test_end in zip(test_edges[:-1], test_edges[1:]):
        train_end = test_start - gap
        yield slice(train_end - train_size, train_end), slice(test_start, test_end)
#%%
for i1, i2 in custom_ts_cv(n_samples=150, train_size=100, test_size=29, gap=20):
    print(i1, i2)    

#%%
cv = lambda: custom_ts_cv(n_samples=len(X), train_size=20, test_size=2, gap=1)
model = RidgeCV(alphas=np.arange(.01, .201, .01), cv=cv()).fit(X, y[:, None])
#%%

def get_oos_resids(model, X, y, cv):
    model = deepcopy(model)
    resids = []
    for train_ix, test_ix in cv:
        model.fit(X[train_ix], y[train_ix])
        resids.append(model.predict(X[test_ix]) - y[test_ix])
    return np.vstack(resids)

oos_resids = get_oos_resids(Ridge(alpha=model.alpha_), X, y[:, None], cv())
print(np.mean(oos_resids**2))
#%%
def _prods_for_cv(X, y, *, cv, w: Optional[np.ndarray] = None) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    xtxs_train, xtys_train, xtxs_test, xtys_test, xms, xstds = [], [], [], [], [], []
    for train_ixs, test_ixs in cv:
        x_train = X[train_ixs]
        x_test = X[test_ixs]
        y_train = y[train_ixs]
        y_test = y[test_ixs]
        if w is not None:
            w_train, w_test = [z / z.sum() for z in (w[train_ixs], w[test_ixs])]            
            xm = w_train @ x_train
            xstd = np.sqrt(w_train @ x_train**2 - xm**2)
            x_train, x_test = [(x - xm) / xstd for x in (x_train, x_test)]
            sqw_train, sqw_test = [np.sqrt(z)[:, None] for z in (w_train, w_test)]
            x_train = x_train * sqw_train
            y_train = y_train * sqw_train
            x_test = x_test * sqw_test
            y_test = y_test * sqw_test
        else:
            xm = np.mean(x_train, axis=0)
            xstd = np.std(x_train, axis=0)
            x_train, x_test = [(x - xm) / xstd for x in (x_train, x_test)]
        xtxs_train.append(x_train.T @ x_train)
        xtxs_test.append(x_test.T @ x_test)
        xtys_train.append(x_train.T @ y_train)
        xtys_test.append(x_test.T @ y_test)        
        xms.append(xm)
        xstds.append(xstd)
        
    return xtxs_train, xtys_train, xtxs_test, xtys_test, xms, xstds
#%%
xtxs_train, xtys_train, xtxs_test, xtys_test, xms, xstds = _prods_for_cv(X, y, cv=cv())
#%%
_, _, _, _, xms1, xstds1 = _prods_for_cv(X, y, cv=cv(), w=np.ones(len(X), dtype=float))
#%%
def loss():
    pass
#%%
plt.plot(np.vstack(xms))
#%%
xms[0] - xms1[0]
#%%
def custom_rolling_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    *,
    w: Optional[np.ndarray] = None,
    train_size: int,
    test_size: int,
    gap: int,
) -> tuple[]:  # train_ixs, resids
    0
    
#%%
plt.plot(oos_resids[])
#%%
cross_val_score(model, X, y, scoring=make_scorer(mean_squared_error), cv=cv(), n_jobs=1)
#%%
sm.OLS(y, sm.add_constant(X)).fit().summary()
#%%
plt.imshow(np.corrcoef(X))
#%%
ridge_coef = Ridge(alpha=1e-4, fit_intercept=True).fit(X, y).coef_
#%%
plt.scatter(ridge_coef, coef)
