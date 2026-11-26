import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from indices import INDEX_FILES, get_index

st.set_page_config(page_title="Signal Validation", layout="wide")
st.title("🧪 Signal Validation – Rules vs ML vs Combined")
st.caption("Walk-forward test on unseen periods, same exits and charges for all three approaches.")

# =====================================================================
# 1. SETTINGS
# =====================================================================
st.sidebar.header("Stocks")
idx_name = st.sidebar.selectbox("Start from index", list(INDEX_FILES),
                                index=list(INDEX_FILES).index("Nifty Bank"))
base = get_index(idx_name)
default_syms = ", ".join(base["Symbol"].astype(str).str.strip()) if not base.empty else "HDFCBANK, ICICIBANK"
txt = st.sidebar.text_area("Symbols (edit if needed)", default_syms, height=140)
symbols = tuple(dict.fromkeys(
    s.strip().upper() + ("" if s.strip().upper().endswith((".NS", ".BO")) else ".NS")
    for s in txt.split(",") if s.strip()))

st.sidebar.header("Rules")
buy_th = st.sidebar.slider("Rule: buy if score ≥", 0, 80, 15, 5)

st.sidebar.header("ML")
horizon = st.sidebar.slider("ML: days ahead", 1, 10, 5)
min_move = st.sidebar.slider("ML: 'Up' if gain ≥ (%)", 0.0, 3.0, 1.0, 0.5)
prob_th = st.sidebar.slider("ML: buy if probability ≥ (%)", 50, 80, 60, 5)

st.sidebar.header("Trade exits (same for all)")
max_hold = st.sidebar.slider("Max holding days", 3, 40, 20)

TRADE_VALUE = 100000
TRAIN_START = 504      # ~2 years of initial training
STEP = 63              # retrain every ~3 months

st.caption(f"{len(symbols)} stocks selected")
if len(symbols) > 20:
    st.warning("Walk-forward ML retrains many times per stock – more than 20 stocks can take several minutes.")


# =====================================================================
# 2. INDICATORS, SCORE, FEATURES (same formulas as the other pages)
# =====================================================================
def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + ag / al)
    df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_Up"], df["BB_Low"] = mid + 2 * std, mid - 2 * std
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df


def score_series(d):
    c, e20, e50, s200 = d["Close"], d["EMA20"], d["EMA50"], d["SMA200"]
    trend = np.select([(c > e20) & (e20 > e50), (c < e20) & (e20 < e50)], [2, -2], 0)
    longt = np.where(s200.isna(), 0, np.where(c > s200, 1, -1))
    r = d["RSI"]
    rsi_p = np.select([r < 30, r > 70, r >= 50], [2, -2, 1], -1)
    m, s = d["MACD"], d["MACD_Signal"]
    up = (m.shift() < s.shift()) & (m > s)
    dn = (m.shift() > s.shift()) & (m < s)
    macd_p = np.select([up, dn, m > s], [2, -2, 1], -1)
    bb_p = np.select([c <= d["BB_Low"], c >= d["BB_Up"]], [1, -1], 0)
    v_avg = d["Volume"].rolling(20).mean()
    vol_p = np.where(d["Volume"] > 1.5 * v_avg, np.where(c >= d["Open"], 1, -1), 0)
    return pd.Series(np.round((trend + longt + rsi_p + macd_p + bb_p + vol_p) / 9 * 100),
                     index=d.index)


def make_features(d):
    c = d["Close"]
    f = pd.DataFrame(index=d.index)
    f["Ret_1d"], f["Ret_5d"], f["Ret_20d"] = c.pct_change(), c.pct_change(5), c.pct_change(20)
    f["Dist_EMA20"] = c / d["EMA20"] - 1
    f["Dist_EMA50"] = c / d["EMA50"] - 1
    f["Dist_SMA200"] = c / d["SMA200"] - 1
    f["EMA20_vs_EMA50"] = d["EMA20"] / d["EMA50"] - 1
    f["RSI"] = d["RSI"]
    f["MACD_Hist_pct"] = d["MACD_Hist"] / c
    f["BB_Position"] = (c - d["BB_Low"]) / (d["BB_Up"] - d["BB_Low"])
    f["Volume_Ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    f["ATR_pct"] = d["ATR"] / c
    f["Volatility_20d"] = f["Ret_1d"].rolling(20).std()
    f["Up_Days_10"] = (c.diff() > 0).rolling(10).sum()
    return f.replace([np.inf, -np.inf], np.nan)


def new_model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=30, random_state=42)


def delivery_charges(bp, sp, q):
    bv, sv = bp * q, sp * q
    to = bv + sv
    br, exch, sebi = 40.0, 0.0000297 * to, 0.000001 * to
    return br + 0.001 * to + 0.00015 * bv + exch + sebi + 0.18 * (br + exch + sebi) + 15.0


# =====================================================================
# 3. WALK-FORWARD ML PROBABILITIES
# =====================================================================
def walk_forward_probs(X, y, fut, horizon):
    n = len(X)
    p = np.full(n, np.nan)
    feat_ok = X["Volatility_20d"].notna().to_numpy()
    lab_ok = fut.notna().to_numpy()
    for ts in range(TRAIN_START, n, STEP):
        tr = np.arange(0, max(ts - horizon, 0))          # labels fully known before ts
        tr = tr[feat_ok[tr] & lab_ok[tr]]
        if len(tr) < 300 or y.iloc[tr].nunique() < 2:
            continue
        m = new_model().fit(X.iloc[tr], y.iloc[tr])
        te = np.arange(ts, min(ts + STEP, n))
        te = te[feat_ok[te]]
        if len(te):
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


# =====================================================================
# 4. TRADE SIMULATOR (identical for all three approaches)
# =====================================================================
def simulate(d, sig, start, max_hold):
    O, H, L, C = (d[k].to_numpy(float) for k in ["Open", "High", "Low", "Close"])
    A = d["ATR"].to_numpy(float)
    n, i, out = len(d), start, []
    while i < n - 1:
        if not sig[i] or np.isnan(A[i]):
            i += 1
            continue
        e = i + 1
        ep = O[e]
        stop, tgt = ep - 1.5 * A[i], ep + 3 * A[i]
        xp, reason, j = None, "", e
        for j in range(e, min(e + max_hold, n)):
            if j > e and O[j] <= stop:
                xp, reason = O[j], "Stop (gap)"
            elif j > e and O[j] >= tgt:
                xp, reason = O[j], "Target (gap)"
            elif L[j] <= stop:
                xp, reason = stop, "Stop-loss"
            elif H[j] >= tgt:
                xp, reason = tgt, "Target"
            if xp is not None:
                break
        if xp is None:
            xp, reason = C[j], "Time exit"
        q = int(TRADE_VALUE // ep)
        if q > 0:
            chg = delivery_charges(ep, xp, q)
            out.append({"Entry date": d.index[e].date(), "Exit date": d.index[j].date(),
                        "Entry ₹": round(ep, 2), "Exit ₹": round(float(xp), 2),
                        "Exit reason": reason,
                        "Net %": round(((xp - ep) * q - chg) / (ep * q) * 100, 3)})
        i = j + 1
    return pd.DataFrame(out)


# =====================================================================
# 5. VALIDATE ONE STOCK (cached)
# =====================================================================
@st.cache_data(ttl=21600, show_spinner=False)
def validate_stock(sym, buy_th, horizon, min_move, prob_th, max_hold):
    raw = yf.download(sym, period="6y", multi_level_index=False, progress=False)
    if raw.empty or len(raw) < TRAIN_START + 150:
        return None
    d = add_indicators(raw)
    sc = score_series(d)
    c = d["Close"]

    rule = ((sc >= buy_th) & (c > d["EMA20"]) & (d["EMA20"] > d["EMA50"]) &
            (d["MACD"] > d["MACD_Signal"]) & d["RSI"].between(45, 70)).to_numpy()

    X = make_features(d)
    fut = c.shift(-horizon) / c - 1
    y = (fut >= min_move / 100).astype(int)
    p = walk_forward_probs(X, y, fut, horizon)
    ml = np.nan_to_num(p, nan=0.0) >= prob_th / 100
    combo = rule & ml

    mask = ~np.isnan(p) & fut.notna().to_numpy()
    auc = (roc_auc_score(y[mask], p[mask])
           if mask.sum() > 50 and y[mask].nunique() == 2 else None)

    start = TRAIN_START
    bh = (float(c.iloc[-1]) / float(c.iloc[start]) - 1) * 100
    return {"Rules only": simulate(d, rule, start, max_hold),
            "ML only": simulate(d, ml, start, max_hold),
            "Combined": simulate(d, combo, start, max_hold),
            "auc": auc, "bh": bh, "test_from": d.index[start].date()}


def metrics(t):
    if t.empty:
        return {"Trades": 0, "Win rate %": None, "Avg net % / trade": None,
                "Profit factor": None, "Total net %": 0.0, "Max DD (% pts)": None}
    r = t.sort_values("Exit date")["Net %"]
    w, l = r[r > 0], r[r <= 0]
    pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float("inf")
    cum = r.cumsum()
    dd = (cum - cum.cummax()).min()
    return {"Trades": len(r), "Win rate %": round(len(w) / len(r) * 100, 1),
            "Avg net % / trade": round(r.mean(), 3), "Profit factor": round(pf, 2),
            "Total net %": round(r.sum(), 1), "Max DD (% pts)": round(dd, 1)}


# =====================================================================
# 6. RUN
# =====================================================================
if st.button("▶ Run validation", type="primary", disabled=not symbols):
    all_tr = {"Rules only": [], "ML only": [], "Combined": []}
    per_stock, failed = [], []
    bar = st.progress(0.0)
    for k, s in enumerate(symbols, 1):
        bar.progress(k / len(symbols), text=f"Validating {s} ({k}/{len(symbols)})")
        try:
            r = validate_stock(s, buy_th, horizon, min_move, prob_th, max_hold)
        except Exception:
            r = None
        if r is None:
            failed.append(s)
            continue
        row = {"Symbol": s.replace(".NS", ""), "Test from": r["test_from"],
               "ML AUC": None if r["auc"] is None else round(r["auc"], 3),
               "Buy & Hold %": round(r["bh"], 1)}
        for name in all_tr:
            t = r[name]
            if not t.empty:
                all_tr[name].append(t.assign(Symbol=row["Symbol"]))
            m = metrics(t)
            row[f"{name} trades"] = m["Trades"]
            row[f"{name} PF"] = m["Profit factor"]
        per_stock.append(row)
    bar.empty()
    st.session_state["val"] = {"tr": {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
                                      for k, v in all_tr.items()},
                               "per": pd.DataFrame(per_stock), "failed": failed}

# =====================================================================
# 7. RESULTS
# =====================================================================
if "val" in st.session_state:
    V = st.session_state["val"]
    summ = pd.DataFrame({k: metrics(v) for k, v in V["tr"].items()}).T
    per = V["per"]

    st.subheader("📊 Summary (all stocks, unseen periods, after charges)")
    st.dataframe(summ, use_container_width=True)
    if not per.empty and per["ML AUC"].notna().any():
        st.caption(f"Average walk-forward ML AUC: {per['ML AUC'].mean():.3f} "
                   "(0.50 = coin toss) | Average Buy & Hold over the test period: "
                   f"{per['Buy & Hold %'].mean():.1f}%")

    # Verdict
    pf = {k: (summ.loc[k, "Profit factor"] or 0) for k in summ.index}
    n_c = int(summ.loc["Combined", "Trades"])
    best_single = max(pf["Rules only"], pf["ML only"])
    if n_c < 30:
        st.warning(f"⚠️ Only {n_c} combined trades – not enough evidence. "
                   "Add more stocks or relax thresholds before deciding.")
    elif pf["Combined"] >= best_single + 0.1:
        st.success("✅ Combining helped on this test: the Combined profit factor beats both "
                   "single approaches by a clear margin. Confirm on a second index before adopting it.")
    else:
        better = "Rules only" if pf["Rules only"] >= pf["ML only"] else "ML only"
        st.error(f"❌ Combining did not clearly help. Keep them separate – '{better}' "
                 "performed better on this test.")
    if all(v < 1 for v in pf.values()):
        st.error("None of the approaches was profitable after charges on these stocks and settings.")

    # Cumulative net % curve
    fig = go.Figure()
    for name, t in V["tr"].items():
        if not t.empty:
            t = t.sort_values("Exit date")
            fig.add_trace(go.Scatter(x=pd.to_datetime(t["Exit date"]), y=t["Net %"].cumsum(),
                                     name=name, mode="lines"))
    fig.update_layout(title="Cumulative net % (sum of trade returns)", height=420,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-stock results")
    st.dataframe(per, hide_index=True, use_container_width=True)

    for name, t in V["tr"].items():
        with st.expander(f"All trades – {name} ({len(t)})"):
            if not t.empty:
                st.dataframe(t, hide_index=True, use_container_width=True)

    if V["failed"]:
        st.warning("Skipped (insufficient data): " + ", ".join(V["failed"]))
    st.info("Educational tool only – past results do not guarantee future performance.")
else:
    st.info("Choose stocks and settings, then click **▶ Run validation**.")