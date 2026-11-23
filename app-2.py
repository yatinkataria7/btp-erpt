from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="Rupee & Inflation: ERPT India", layout="wide")
st.title("Rupee depreciation & inflation in India")
st.caption("VAR vs machine-learning forecasts and exchange-rate pass-through (BTP, IIT Ropar)")

# ------------------------------------------------------------- data
up = st.sidebar.file_uploader("Monthly CSV (optional)", type="csv")
if up is not None:
    df = core.load_csv(up)
elif Path("panel.csv").exists():
    df = core.load_csv("panel.csv")
else:
    st.error("Put panel.csv next to app.py or upload it.")
    st.stop()

last = df.index[-1]
st.sidebar.write(f"Data: {df.index[0]:%b %Y} to {last:%b %Y} ({len(df)} months)")
if (pd.Timestamp.today() - last).days > 150:
    st.sidebar.warning("Data looks out of date. Extend panel.csv before trusting the forecast.")

# ------------------------------------------------------------- controls
end_year = st.sidebar.number_input("Forecast up to year", last.year + 1, 2036, min(2030, 2036))
start_year = st.sidebar.slider("Train models from year", 2005, 2018, 2014,
                               help="Post-2014 data avoids the high-inflation 2008-13 regime.")
ci = st.sidebar.slider("VAR uncertainty band (%)", 50, 95, 90)
n_sims = st.sidebar.select_slider("VAR simulations", [200, 500, 1000, 2000], 1000)

st.sidebar.markdown("**Scenario inputs** (set by you, not forecast)")
BRENT_OPTS = {"Hold at latest level": "hold", "$60": 60, "$80": 80, "$100": 100,
              "Free (model decides)": "free"}
FED_OPTS = {"Hold at latest level": "hold", "Move to 3%": 3, "Move to 5%": 5,
            "Free (model decides)": "free"}
brent_lbl = st.sidebar.selectbox("Brent crude", list(BRENT_OPTS))
fed_lbl = st.sidebar.selectbox("US fed funds rate", list(FED_OPTS))
brent_opt, fed_opt = BRENT_OPTS[brent_lbl], FED_OPTS[fed_lbl]

sel = st.sidebar.multiselect("ML models to overlay", core.ML_KINDS, default=core.ML_KINDS)
ml_h = st.sidebar.slider("ML forecast horizon (months)", 6, 60, 24,
                         help="Recursive ML forecasts become unreliable beyond ~2 years.")

dft = df.loc[str(start_year):]
key = (len(dft), str(dft.index[0]), str(last))


@st.cache_resource
def fit_models(key, _dft):
    s = core.make_stationary(_dft)
    return s, core.fit_var(s), {k: core.fit_ml(s, k) for k in core.ML_KINDS}


@st.cache_resource
def ml_forecast(key, kind, steps, scen, _models, _s, _fixed):
    return core.forecast_ml(_models, _s, steps, fixed=_fixed)


@st.cache_data(show_spinner=False)
def run_backtest(key, n_origins, _dft):
    return core.rolling_backtest(_dft, n_origins=n_origins)


s, res, ml_models = fit_models(key, dft)

LABELS = {
    "cpi_yoy": "CPI inflation (YoY %)",
    "usdinr": "USD/INR (Rs per $)",
    "repo": "Repo rate (%)",
    "brent": "Brent crude ($/bbl)",
    "fedfunds": "US fed funds rate (%)",
}
HIST = {
    "cpi_yoy": 100 * np.log(df.cpi).diff(12),
    "usdinr": df.usdinr, "repo": df.repo, "brent": df.brent, "fedfunds": df.fedfunds,
}
COLORS = {"XGBoost": "#16a085", "Random Forest": "#8e44ad"}

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Pass-through", "Model comparison", "Data"])

# ------------------------------------------------------------ forecast
with tab1:
    steps = (end_year - last.year) * 12 + 12 - last.month
    idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    fixed = core.scenario_fixed(df, brent_opt, fed_opt, steps)
    paths = core.simulate_paths(res, s, steps, n=n_sims, fixed=fixed)
    lv = core.growth_to_levels(paths, df)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2

    steps_ml = min(steps, ml_h)
    idx_ml = idx[:steps_ml]
    fixed_ml = core.scenario_fixed(df, brent_opt, fed_opt, steps_ml)
    with st.spinner("Forecasting with ML models..."):
        ml = {k: core.growth_to_levels(
                  ml_forecast(key, k, steps_ml, (brent_opt, fed_opt), ml_models[k], s, fixed_ml), df)
              for k in sel}

    st.info(f"Scenario: Brent = {brent_lbl}; Fed funds = {fed_lbl}; models trained on "
            f"{start_year} onward. ML lines stop after {steps_ml} months.")
    if end_year - last.year > 3:
        st.warning("Beyond ~2-3 years these are scenario projections with wide uncertainty, "
                   "not reliable predictions. Read the bands, not the median line.")

    rows = []
    cols = st.columns(2)
    for i, (v, label) in enumerate(LABELS.items()):
        lo, med, hi = np.percentile(lv[v], [lo_q, 50, hi_q], axis=0)
        fig = go.Figure()
        fig.add_scatter(x=HIST[v].index, y=HIST[v], name="History", line=dict(color="#1f3864"))
        fig.add_scatter(x=idx, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=idx, y=lo, fill="tonexty", name=f"VAR {ci}% band",
                        line=dict(width=0), fillcolor="rgba(200,60,50,0.18)")
        fig.add_scatter(x=idx, y=med, name="VAR median", line=dict(color="#c0392b"))
        for k in sel:
            fig.add_scatter(x=idx_ml, y=ml[k][v][0], name=k, line=dict(color=COLORS[k], dash="dash"))
        fig.update_layout(title=label, height=340, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"))
        cols[i % 2].plotly_chart(fig)
        row = {"Variable": label, f"VAR median Dec {end_year}": round(med[-1], 2),
               f"VAR {lo_q:.0f}th": round(lo[-1], 2), f"VAR {hi_q:.0f}th": round(hi[-1], 2)}
        for k in sel:
            row[f"{k} ({steps_ml}m)"] = round(ml[k][v][0][-1], 2)
        rows.append(row)
    st.subheader(f"Outlook to December {end_year}")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# -------------------------------------------------------- pass-through
with tab2:
    c1, c2 = st.columns(2)
    shock = c1.slider("Rupee depreciation shock (%)", 1, 20, 10)
    hor = c2.slider("Horizon (months)", 6, 36, 24)
    irf = core.pass_through(res, shock, hor)
    fig = go.Figure(go.Scatter(x=irf.index, y=irf.values, mode="lines+markers"))
    fig.update_layout(title=f"CPI level response to a {shock}% depreciation",
                      xaxis_title="Months after shock", yaxis_title="% change in CPI level", height=380)
    st.plotly_chart(fig)
    st.metric(f"Cumulative pass-through after {hor} months", f"{irf.iloc[-1] / shock:.2f}")
    st.caption("Pass-through = % rise in CPI level per 1% depreciation (VAR, Cholesky ordering: "
               "Fed, Brent, USD/INR, repo, CPI).")
    st.subheader("Before vs after flexible inflation targeting (Oct 2016)")
    reg = core.pt_by_regime(df, horizon=12, shock_pct=shock)
    st.dataframe(pd.DataFrame({"12-month pass-through": reg}).round(3))
    st.caption("The post-2016 sample is small, so treat this comparison as indicative.")

# ---------------------------------------------------- model comparison
with tab3:
    st.write("Each model is re-fitted at several past dates, forecasts the next 12 months, and is "
             "scored against what actually happened (RMSE, lower is better). The drift benchmark "
             "just extrapolates the average historical growth.")
    n_or = st.select_slider("Number of forecast origins (6 months apart)", [6, 8, 10, 12], 8)
    if st.button("Run model comparison"):
        with st.spinner("Re-fitting all models at each origin (can take a minute)..."):
            try:
                st.session_state["bt"] = run_backtest(key, n_or, dft)
            except ValueError as e:
                st.error(f"{e} Try a smaller number of origins or an earlier training start year.")
    if "bt" in st.session_state:
        rmse, err = st.session_state["bt"]
        v = st.selectbox("Variable", list(LABELS), format_func=LABELS.get)
        sub = rmse.loc[v]
        c1, c2 = st.columns(2)
        c1.subheader("RMSE by horizon")
        c1.dataframe(sub)
        rel = sub.div(sub["Drift (benchmark)"], axis=0).round(2)
        c2.subheader("Relative to drift (below 1 = beats benchmark)")
        c2.dataframe(rel)
        fig = go.Figure()
        for m in sub.columns:
            fig.add_scatter(x=sub.index, y=sub[m], mode="lines+markers", name=m)
        fig.update_layout(xaxis_title="Forecast horizon (months)", yaxis_title="RMSE", height=340,
                          title=f"{LABELS[v]}: forecast error vs horizon")
        st.plotly_chart(fig)
        st.subheader("Winner by variable and horizon")
        win = rmse.idxmin(axis=1).unstack("h")
        win.index = [LABELS[i] for i in win.index]
        win.columns = [f"{h}-month" for h in win.columns]
        st.dataframe(win)
        st.caption("Few origins means noisy results, so a small RMSE gap is not proof one model is better.")

# ---------------------------------------------------------------- data
with tab4:
    st.dataframe(df)
    st.download_button("Download panel CSV", df.to_csv().encode(), "panel.csv")
