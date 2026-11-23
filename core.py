"""core.py - data, models and forecasting for the ERPT BTP."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

# Order matters for the orthogonalised IRF: global/exogenous first, CPI last.
COLS = ["d_fed", "d_brent", "d_usdinr", "d_repo", "d_cpi"]
IDX = {c: i for i, c in enumerate(COLS)}


def load_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer, parse_dates=["date"], index_col="date")
    df = df.resample("MS").mean()
    need = ["usdinr", "cpi", "repo", "brent", "fedfunds"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}. Need: date,{','.join(need)}")
    return df[need].ffill(limit=6).dropna()


def make_stationary(df):
    s = pd.DataFrame(index=df.index)
    s["d_fed"] = df.fedfunds.diff()
    s["d_brent"] = np.log(df.brent).diff()
    s["d_usdinr"] = np.log(df.usdinr).diff()
    s["d_repo"] = df.repo.diff()
    s["d_cpi"] = np.log(df.cpi).diff()
    return s.dropna()[COLS]


def growth_to_levels(g, hist):
    """Convert stationary-variable paths (steps,k) or (n,steps,k) back to levels."""
    g = np.asarray(g)
    if g.ndim == 2:
        g = g[None]
    cum = g.cumsum(axis=1)
    n = g.shape[0]
    out = {
        "usdinr": hist.usdinr.iloc[-1] * np.exp(cum[:, :, IDX["d_usdinr"]]),
        "brent": hist.brent.iloc[-1] * np.exp(cum[:, :, IDX["d_brent"]]),
        "repo": np.clip(hist.repo.iloc[-1] + cum[:, :, IDX["d_repo"]], 0, 15),
        "fedfunds": np.clip(hist.fedfunds.iloc[-1] + cum[:, :, IDX["d_fed"]], 0, 10),
    }
    log_hist = np.log(hist.cpi.values[-12:])
    log_path = np.log(hist.cpi.iloc[-1]) + cum[:, :, IDX["d_cpi"]]
    full = np.concatenate([np.tile(log_hist, (n, 1)), log_path], axis=1)
    out["cpi_yoy"] = 100 * (full[:, 12:] - full[:, :-12])
    return out


# ------------------------------------------------------------------ VAR
def fit_var(s, maxlags=6):
    res = VAR(s).fit(maxlags=maxlags, ic="bic")
    if res.k_ar == 0:
        res = VAR(s).fit(1)
    return res


def pass_through(res, shock_pct=10, horizon=24):
    """Cumulative % response of the CPI level to a shock_pct % INR depreciation."""
    irf = res.irf(horizon)
    resp = irf.orth_irfs[:, IDX["d_cpi"], IDX["d_usdinr"]]
    L = np.linalg.cholesky(np.asarray(res.sigma_u))
    size = L[IDX["d_usdinr"], IDX["d_usdinr"]]
    cum = 100 * np.cumsum(resp) * (shock_pct / 100) / size
    return pd.Series(cum, index=range(horizon + 1), name="CPI level response (%)")


def pt_by_regime(df, break_date="2016-10-01", horizon=12, shock_pct=10):
    out = {}
    for name, part in {"Pre Oct-2016": df.loc[:break_date], "Post Oct-2016": df.loc[break_date:]}.items():
        try:
            res = fit_var(make_stationary(part), maxlags=3)
            out[name] = float(pass_through(res, shock_pct, horizon).iloc[-1] / shock_pct)
        except Exception:
            out[name] = np.nan
    return out


# ------------------------------------------------------------- ML models
ML_KINDS = ["XGBoost", "Random Forest"]


def _make_estimator(kind):
    if kind == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=0, n_jobs=1)
    if kind == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                     random_state=0, n_jobs=1)
    raise ValueError(f"unknown model {kind}")


def _lag_matrix(s, lags):
    cols = {}
    for l in range(1, lags + 1):
        for c in s.columns:
            cols[f"{c}_L{l}"] = s[c].shift(l)
    return pd.DataFrame(cols, index=s.index)


def fit_ml(s, kind="XGBoost", lags=6):
    """One regressor per variable: next-month change ~ 6 lags of every variable."""
    X = _lag_matrix(s, lags)
    data = X.join(s).dropna()
    return {c: _make_estimator(kind).fit(data[X.columns], data[c]) for c in s.columns}


# ------------------------------------------------- model comparison
MODELS = ["VAR", "XGBoost", "Random Forest", "Drift (benchmark)"]
HORIZONS = [1, 3, 6, 12]


def _forecast_all(train, steps):
    s_tr = make_stationary(train)
    res = fit_var(s_tr)
    g = {
        "VAR": res.forecast(s_tr.values[-res.k_ar:], steps),
        "Drift (benchmark)": np.tile(s_tr.mean().values, (steps, 1)),
    }
    for kind in ML_KINDS:
        g[kind] = forecast_ml(fit_ml(s_tr, kind), s_tr, steps)
    return {m: growth_to_levels(x, train) for m, x in g.items()}


def rolling_backtest(df, n_origins=10, step=6, max_h=12):
    """Refit at several past dates, forecast max_h months, score at horizons 1, 3, 6, 12."""
    yoy = 100 * np.log(df.cpi).diff(12)
    truth = {"usdinr": df.usdinr, "brent": df.brent, "repo": df.repo, "cpi_yoy": yoy}
    first = len(df) - max_h - step * (n_origins - 1)
    if first < 60:
        raise ValueError("Not enough history for that many origins; lower n_origins.")
    rows = []
    for o in range(n_origins):
        cut = first + o * step
        lv = _forecast_all(df.iloc[:cut], max_h)
        for m, d in lv.items():
            for v, series in truth.items():
                actual = series.iloc[cut:cut + max_h].values
                pred = d[v][0]
                for h in HORIZONS:
                    rows.append({"model": m, "variable": v, "h": h, "origin": df.index[cut],
                                 "err": pred[h - 1] - actual[h - 1]})
    err = pd.DataFrame(rows)
    rmse = (err.groupby(["variable", "h", "model"])["err"]
               .apply(lambda x: np.sqrt(np.mean(x ** 2))).unstack("model")[MODELS])
    return rmse.round(3), err


# ------------------------------------------------- scenarios + forecasting
def scenario_fixed(df, brent="hold", fed="hold", steps=1, ramp=12):
    """Variables set by scenario instead of simulated.
    brent / fed: "free" (model decides), "hold" (latest level), or a target number.
    The move to the target is spread evenly over `ramp` months."""
    def ramp_path(total):
        p = np.zeros(steps)
        n = min(ramp, steps)
        p[:n] = total / ramp
        return p

    fixed = {}
    if brent != "free":
        last = df.brent.iloc[-1]
        target = last if brent == "hold" else float(brent)
        fixed[IDX["d_brent"]] = ramp_path(np.log(target / last))
    if fed != "free":
        last = df.fedfunds.iloc[-1]
        target = last if fed == "hold" else float(fed)
        fixed[IDX["d_fed"]] = ramp_path(target - last)
    return fixed


def simulate_paths(res, s, steps, n=1000, seed=0, fixed=None):
    """VAR Monte-Carlo; columns in `fixed` follow the scenario path exactly."""
    fixed = fixed or {}
    rng = np.random.default_rng(seed)
    k, p = res.neqs, res.k_ar
    c, A = np.asarray(res.intercept), np.asarray(res.coefs)
    L = np.linalg.cholesky(np.asarray(res.sigma_u))
    hist = np.repeat(s.values[-p:][None], n, axis=0)
    out = np.zeros((n, steps, k))
    for t in range(steps):
        y = np.tile(c, (n, 1))
        for i in range(p):
            y = y + hist[:, -1 - i, :] @ A[i].T
        y = y + rng.standard_normal((n, k)) @ L.T
        for j, path in fixed.items():
            y[:, j] = path[t]
        out[:, t] = y
        hist = np.concatenate([hist[:, 1:], y[:, None]], axis=1)
    return out


def forecast_ml(models, s, steps, lags=6, fixed=None):
    """Recursive ML forecast; columns in `fixed` follow the scenario path."""
    fixed = fixed or {}
    hist = s.copy()
    rows = []
    for t in range(steps):
        x = {f"{c}_L{l}": hist[c].iloc[-l] for l in range(1, lags + 1) for c in s.columns}
        X = pd.DataFrame([x])
        row = {c: float(m.predict(X)[0]) for c, m in models.items()}
        for j, path in fixed.items():
            row[COLS[j]] = float(path[t])
        rows.append([row[c] for c in s.columns])
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    return np.array(rows)
