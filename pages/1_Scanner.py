import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from indices import INDEX_FILES, get_index

st.set_page_config(page_title="Swing Screener", layout="wide")
st.title("🔎 Swing-Trading Screener (NSE/BSE)")

# =====================================================================
# 1. STOCK UNIVERSE – Nifty indices or saved watchlists
# =====================================================================
@st.cache_data(ttl=86400, show_spinner="Loading index constituents...")
def index_constituents(name):
    return get_index(name)


came_from_watchlist = bool(st.session_state.get("scan_list"))
source = st.sidebar.radio("Scan from", ["Nifty indices", "My watchlists"],
                          index=1 if came_from_watchlist else 0, horizontal=True)

if source == "Nifty indices":
    picks = st.sidebar.multiselect("Select one or more indices", list(INDEX_FILES),
                                   default=["Nifty 50"])
    frames = []
    for n in picks:
        d0 = index_constituents(n)
        if d0.empty:
            st.sidebar.warning(f"{n}: list unavailable")
        else:
            frames.append(d0.assign(Index=n))
    universe = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["Symbol", "Industry"]))
    if not universe.empty and "Industry" in universe.columns:
        industries = sorted(universe["Industry"].dropna().unique())
        sel_ind = st.sidebar.multiselect("Filter by industry (optional)", industries)
        if sel_ind:
            universe = universe[universe["Industry"].isin(sel_ind)]
    start_syms = list(dict.fromkeys(universe["Symbol"].astype(str).str.strip()))
    st.sidebar.caption(f"{len(start_syms)} unique stocks from {len(picks)} index(es)")
else:
    saved = {}
    if os.path.exists("data/watchlists.json"):
        with open("data/watchlists.json", encoding="utf-8") as f:
            saved.update(json.load(f))
    saved.update(st.session_state.get("watchlists", {}))
    if not saved:
        st.sidebar.info("No watchlists yet – create one on the Watchlists page.")
        start_syms = []
    else:
        names = list(saved.keys())
        pre = st.session_state.get("scan_list")
        chosen = st.sidebar.selectbox("Watchlist", names,
                                      index=names.index(pre) if pre in names else 0)
        start_syms = saved[chosen]

tickers_text = st.sidebar.text_area("Symbols (edit if needed)",
                                    ", ".join(start_syms), height=160)

# =====================================================================
# 2. SETTINGS
# =====================================================================
st.sidebar.subheader("Rule settings")
buy_th = st.sidebar.slider("Buy if score ≥", 0, 80, 15, 5)
sell_th = st.sidebar.slider("Avoid if score ≤", -80, 0, -15, 5)
min_rr = st.sidebar.slider("Minimum risk/reward (1 : x)", 1.0, 3.0, 2.0, 0.5)

st.sidebar.subheader("ML (shown separately)")
use_ml = st.sidebar.checkbox("Include ML probability + AUC (slower)", False)
horizon = st.sidebar.slider("ML: days ahead", 1, 10, 5)
min_move = st.sidebar.slider("ML: 'Up' if gain ≥ (%)", 0.0, 3.0, 1.0, 0.5)

symbols = tuple(dict.fromkeys(
    s.strip().upper() + ("" if s.strip().upper().endswith((".NS", ".BO")) else ".NS")
    for s in tickers_text.split(",") if s.strip()))
st.caption(f"{len(symbols)} symbols selected")
if len(symbols) > 150:
    st.warning(f"{len(symbols)} stocks selected – scan may take 1–3 minutes. "
               "Avoid ticking ML for large lists.")


# =====================================================================
# 3. INDICATORS & RULE SCORE (same formulas as the Analyzer)
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


# =====================================================================
# 4. SWING ANALYSIS – trend, RSI, MACD, S/R, entry, stop, targets, R:R, decision
# =====================================================================
def swing_analysis(d, s_now):
    last = d.iloc[-1]
    close, atr = float(last["Close"]), float(last["ATR"])
    e20, e50 = float(last["EMA20"]), float(last["EMA50"])
    rsi = float(last["RSI"])

    # Trend
    if close > e20 > e50:
        trend = "🟢 Bullish"
    elif close < e20 < e50:
        trend = "🔴 Bearish"
    else:
        trend = "🟡 Sideways"

    # RSI interpretation
    if rsi < 30:
        rsi_view = "Oversold"
    elif rsi < 45:
        rsi_view = "Weak"
    elif rsi <= 60:
        rsi_view = "Neutral"
    elif rsi <= 70:
        rsi_view = "Strong"
    else:
        rsi_view = "Overbought"

    # MACD confirmation (crossover in last 3 sessions = fresh)
    above = d["MACD"] > d["MACD_Signal"]
    prev_above = above.shift(1, fill_value=False)
    cross_up = bool((above & ~prev_above).tail(3).any()) and bool(above.iloc[-1])
    cross_dn = bool((~above & prev_above).tail(3).any()) and not bool(above.iloc[-1])
    if cross_up:
        macd_view = "🟢 Bullish crossover"
    elif cross_dn:
        macd_view = "🔴 Bearish crossover"
    else:
        macd_view = "🟢 Bullish" if above.iloc[-1] else "🔴 Bearish"

    # Support & resistance from swing points (last ~6 months)
    recent = d.tail(120)
    lows, highs = recent["Low"], recent["High"]
    piv_low = lows[lows == lows.rolling(11, center=True).min()]
    piv_high = highs[highs == highs.rolling(11, center=True).max()]
    supports = sorted([float(v) for v in piv_low if v < close], reverse=True)
    resists = sorted([float(v) for v in piv_high if v > close])
    s1 = supports[0] if supports else float(lows.min())
    r1 = resists[0] if resists else None          # None = at/near highs (breakout zone)
    r2 = resists[1] if len(resists) > 1 else None

    # Entry range
    entry_lo = max(s1, close - 0.5 * atr)
    entry_hi = close + 0.25 * atr

    # Stop-loss: ATR-based, tucked under support if support is close
    stop = close - 1.5 * atr
    if s1 < stop and close - (s1 - 0.25 * atr) <= 2.5 * atr:
        stop = s1 - 0.25 * atr
    risk = close - stop

    # Targets
    t1 = min(r1, close + 2 * risk) if (r1 and r1 > close + 0.5 * risk) else close + 2 * risk
    t2 = r2 if (r2 and r2 > t1) else close + 3 * risk
    if t2 <= t1:
        t2 = t1 + risk
    rr1, rr2 = (t1 - close) / risk, (t2 - close) / risk

    # Final decision (rule-based only – ML is NOT used here)
    why = []
    if s_now < buy_th:
        why.append(f"score {s_now} < {buy_th}")
    if "Bullish" not in trend:
        why.append("trend not bullish")
    if not above.iloc[-1]:
        why.append("MACD bearish")
    if not 45 <= rsi <= 70:
        why.append(f"RSI {rsi:.0f} outside 45–70")
    if rr2 < min_rr:
        why.append(f"R:R {rr2:.1f} < {min_rr}")
    if r1 and (r1 - close) < risk:
        why.append("resistance too close")

    if "Bearish" in trend or s_now <= sell_th:
        decision = "🔴 AVOID"
    elif not why:
        decision = "🟢 BUY"
    elif s_now >= buy_th or cross_up:
        decision = "👀 WATCH"
    else:
        decision = "🟡 HOLD"

    return {
        "Decision": decision,
        "Trend": trend,
        "RSI": round(rsi, 1),
        "RSI view": rsi_view,
        "MACD": macd_view,
        "Support ₹": round(s1, 2),
        "Resistance ₹": round(r1, 2) if r1 else None,
        "Entry range ₹": f"{entry_lo:,.1f} – {entry_hi:,.1f}",
        "Stop ₹": round(stop, 2),
        "T1 ₹": round(t1, 2),
        "T2 ₹": round(t2, 2),
        "R:R T1": round(rr1, 2),
        "R:R T2": round(rr2, 2),
        "Why": "✅ All checks passed" if not why else "; ".join(why),
    }


# =====================================================================
# 5. ML – probability + AUC on unseen data (kept SEPARATE from Decision)
# =====================================================================
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


def ml_check(d):
    X = make_features(d)
    fut = d["Close"].shift(-horizon) / d["Close"] - 1
    y = (fut >= min_move / 100).astype(int)
    valid = fut.notna() & X["Volatility_20d"].notna()
    Xv, yv = X[valid], y[valid]
    if len(Xv) < 400 or yv.nunique() < 2:
        return None, None
    split = int(len(Xv) * 0.7)
    X_tr, y_tr = Xv.iloc[:split - horizon], yv.iloc[:split - horizon]
    X_te, y_te = Xv.iloc[split:], yv.iloc[split:]
    auc = None
    if y_tr.nunique() == 2 and y_te.nunique() == 2:
        m = new_model().fit(X_tr, y_tr)
        auc = float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    final = new_model().fit(Xv, yv)
    prob = float(final.predict_proba(X.iloc[[-1]])[:, 1][0]) * 100
    return prob, auc


def ml_confidence(auc):
    if auc is None:
        return "—"
    if auc >= 0.58:
        return "High"
    if auc >= 0.53:
        return "Moderate"
    return "None (ignore)"


# =====================================================================
# 6. RUN SCAN
# =====================================================================
@st.cache_data(ttl=900, show_spinner="Downloading prices...")
def download_all(symbols, period):
    return yf.download(list(symbols), period=period, group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)


if st.button("▶ Run scan", type="primary", disabled=not symbols):
    data = download_all(symbols, "5y" if use_ml else "2y")
    rows, failed = [], []
    bar = st.progress(0.0, text="Scanning...")

    for n, t in enumerate(symbols, 1):
        bar.progress(n / len(symbols), text=f"Scanning {t} ({n}/{len(symbols)})")
        try:
            d = data[t].dropna(how="all") if len(symbols) > 1 else data.dropna(how="all")
            if len(d) < 220:
                failed.append(t)
                continue
            d = add_indicators(d)
            sc = score_series(d)
            s_now = int(sc.iloc[-1])
            close, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])

            row = {"Symbol": t.replace(".NS", ""),
                   "Close ₹": round(close, 2),
                   "Change %": round((close / prev - 1) * 100, 2),
                   "Score": s_now,
                   "Score 5d ago": int(sc.iloc[-6]),
                   "Vol × avg": round(float(d["Volume"].iloc[-1]) /
                                      float(d["Volume"].tail(20).mean()), 2)}
            row.update(swing_analysis(d, s_now))

            if use_ml:
                prob, auc = ml_check(d)
                row["ML Prob %"] = None if prob is None else round(prob, 1)
                row["ML AUC"] = None if auc is None else round(auc, 3)
                row["ML confidence"] = ml_confidence(auc)
            rows.append(row)
        except Exception:
            failed.append(t)

    bar.empty()
    st.session_state["scan"] = pd.DataFrame(rows)
    st.session_state["failed"] = failed
    st.session_state["scan_time"] = pd.Timestamp.now().strftime("%d-%b-%Y %H:%M")

# =====================================================================
# 7. RESULTS
# =====================================================================
if "scan" in st.session_state and not st.session_state["scan"].empty:
    res = st.session_state["scan"]
    st.caption(f"Last scan: {st.session_state['scan_time']} | "
               "Decision = rule-based only. ML columns are shown separately for comparison.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Scanned", len(res))
    k2.metric("🟢 Buy", int((res["Decision"] == "🟢 BUY").sum()))
    k3.metric("👀 Watch", int((res["Decision"] == "👀 WATCH").sum()))
    k4.metric("🔴 Avoid", int((res["Decision"] == "🔴 AVOID").sum()))
    k5.metric("Market breadth (Bullish %)",
              f"{res['Trend'].str.contains('Bullish').mean() * 100:.0f}%")

    main_cols = ["Symbol", "Decision", "Score", "Score 5d ago", "Close ₹", "Entry range ₹",
                 "Stop ₹", "T1 ₹", "T2 ₹", "R:R T2", "Trend", "RSI", "RSI view", "MACD",
                 "Support ₹", "Resistance ₹", "Vol × avg"]
    if "ML Prob %" in res.columns:
        main_cols += ["ML Prob %", "ML AUC", "ML confidence"]
    main_cols += ["Why"]

    tabs = st.tabs(["🟢 Buy", "👀 Watch", "🔴 Avoid", "📋 All"])
    for tab, label in zip(tabs, ["🟢 BUY", "👀 WATCH", "🔴 AVOID", None]):
        with tab:
            v = res if label is None else res[res["Decision"] == label]
            v = v.sort_values("Score", ascending=(label == "🔴 AVOID"))
            if v.empty:
                st.info("No stocks in this group.")
            else:
                st.dataframe(v[main_cols], hide_index=True, use_container_width=True)

    # Score chart (top/bottom 25 for large lists)
    srt = res.sort_values("Score")
    srt = pd.concat([srt.head(25), srt.tail(25)]).drop_duplicates() if len(srt) > 60 else srt
    fig = go.Figure(go.Bar(
        x=srt["Score"], y=srt["Symbol"], orientation="h",
        marker_color=["green" if v >= buy_th else "red" if v <= sell_th else "gold"
                      for v in srt["Score"]]))
    fig.update_layout(title="Rule score by stock", height=max(400, 18 * len(srt)))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="scan_results.csv", mime="text/csv")

    if st.session_state["failed"]:
        st.warning("Skipped (no/insufficient data): " + ", ".join(st.session_state["failed"]))

    st.info("Before acting: open the stock in the Analyzer to check its backtest and "
            "Fundamentals. Educational tool only – not investment advice.")
else:
    st.info("Select stocks in the sidebar and click **▶ Run scan**.")