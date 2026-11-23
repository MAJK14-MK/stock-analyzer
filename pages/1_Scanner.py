import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Watchlist Scanner", layout="wide")
st.title("🔎 Watchlist Scanner")

# ---------- 1. Default watchlist (edit freely in the sidebar) ----------
NIFTY50 = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN",
           "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
           "HCLTECH", "M&M", "TITAN", "NTPC", "ULTRACEMCO", "ASIANPAINT", "POWERGRID",
           "TATASTEEL", "ONGC", "ADANIENT", "ADANIPORTS", "BAJAJFINSV", "NESTLEIND", "JSWSTEEL",
           "COALINDIA", "WIPRO", "GRASIM", "TECHM", "HINDALCO", "CIPLA", "DRREDDY", "SBILIFE",
           "HDFCLIFE", "BRITANNIA", "EICHERMOT", "APOLLOHOSP", "TATACONSUM", "BEL", "TRENT",
           "SHRIRAMFIN", "BAJAJ-AUTO", "HEROMOTOCO", "INDUSINDBK", "JIOFIN", "ETERNAL"]

# ---------- 2. Sidebar settings ----------
st.sidebar.header("Scanner settings")
#tickers_text = st.sidebar.text_area("Symbols (comma separated, NSE, without .NS)",
#                                    ", ".join(NIFTY50), height=220)
# Saved watchlists (from the Watchlists page)
import json, os
saved = {"Nifty 50 (built-in)": [s + ".NS" for s in NIFTY50]}
if os.path.exists("data/watchlists.json"):
    with open("data/watchlists.json", encoding="utf-8") as f:
        saved.update(json.load(f))
saved.update(st.session_state.get("watchlists", {}))

names = list(saved.keys())
pre = st.session_state.get("scan_list")
idx = names.index(pre) if pre in names else 0
chosen = st.sidebar.selectbox("Watchlist", names, index=idx)
tickers_text = st.sidebar.text_area("Symbols (edit if needed)",
                                    ", ".join(saved[chosen]), height=220)

buy_th = st.sidebar.slider("Buy if score ≥", 0, 80, 15, 5)
sell_th = st.sidebar.slider("Sell if score ≤", -80, 0, -15, 5)
use_ml = st.sidebar.checkbox("Include ML probability (slower)", False)
horizon = st.sidebar.slider("ML: days ahead", 1, 10, 5)
min_move = st.sidebar.slider("ML: 'Up' if gain ≥ (%)", 0.0, 3.0, 1.0, 0.5)

symbols = tuple(dict.fromkeys(                      # removes duplicates, keeps order
    s.strip().upper() + ("" if s.strip().upper().endswith((".NS", ".BO")) else ".NS")
    for s in tickers_text.split(",") if s.strip()))
st.caption(f"{len(symbols)} symbols in watchlist")

# ---------- 3. Indicators (same formulas as app.py) ----------
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
    f["Rule_Score"] = score_series(d)
    return f.replace([np.inf, -np.inf], np.nan)

def ml_probability(d, horizon, min_move):
    X = make_features(d)
    fut = d["Close"].shift(-horizon) / d["Close"] - 1
    y = (fut >= min_move / 100).astype(int)
    valid = fut.notna() & X["Volatility_20d"].notna()
    if valid.sum() < 300 or y[valid].nunique() < 2:
        return np.nan
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                       min_samples_leaf=30, random_state=42)
    m.fit(X[valid], y[valid])
    return float(m.predict_proba(X.iloc[[-1]])[:, 1][0]) * 100

def label(s):
    if s >= 40: return "🟢 STRONG BUY"
    if s >= buy_th: return "🟢 BUY"
    if s <= -40: return "🔴 STRONG SELL"
    if s <= sell_th: return "🔴 SELL"
    return "🟡 HOLD"

# ---------- 4. Download all symbols in one go ----------
@st.cache_data(ttl=900, show_spinner="Downloading prices for all symbols...")
def download_all(symbols, period):
    return yf.download(list(symbols), period=period, group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)

# ---------- 5. Run scan ----------
if st.button("▶ Run scan", type="primary"):
    data = download_all(symbols, "5y" if use_ml else "2y")
    rows, failed = [], []
    bar = st.progress(0.0, text="Scanning...")

    for n, t in enumerate(symbols, 1):
        bar.progress(n / len(symbols), text=f"Scanning {t} ({n}/{len(symbols)})")
        try:
            d = data[t].dropna(how="all")
            if len(d) < 220:
                failed.append(t); continue
            d = add_indicators(d)
            sc = score_series(d)
            last, prev = d.iloc[-1], d.iloc[-2]
            close = float(last["Close"])
            e20, e50 = float(last["EMA20"]), float(last["EMA50"])
            trend = "Up" if close > e20 > e50 else ("Down" if close < e20 < e50 else "Sideways")
            s_now = int(sc.iloc[-1])
            atr = float(last["ATR"])
            row = {
                "Symbol": t.replace(".NS", ""),
                "Close ₹": round(close, 2),
                "Change %": round((close / float(prev["Close"]) - 1) * 100, 2),
                "Score": s_now,
                "Score 5d ago": int(sc.iloc[-6]),
                "Signal": label(s_now),
                "Trend": trend,
                "RSI": round(float(last["RSI"]), 1),
                "Vol × avg": round(float(last["Volume"]) / float(d["Volume"].tail(20).mean()), 2),
                "Stop ₹": round(close - 1.5 * atr, 2) if s_now >= 0 else round(close + 1.5 * atr, 2),
                "Target ₹": round(close + 3 * atr, 2) if s_now >= 0 else round(close - 3 * atr, 2),
            }
            if use_ml:
                p = ml_probability(d, horizon, min_move)
                row["ML Prob %"] = round(p, 1) if not np.isnan(p) else None
                # Combined: score (-100..100) + ML scaled to same range, 50/50 weight
                row["Combined"] = round(0.5 * s_now + 0.5 * ((p - 50) * 2), 1) if not np.isnan(p) else s_now
            rows.append(row)
        except Exception:
            failed.append(t)

    bar.empty()
    st.session_state["scan"] = pd.DataFrame(rows)
    st.session_state["failed"] = failed
    st.session_state["scan_time"] = pd.Timestamp.now().strftime("%d-%b-%Y %H:%M")

# ---------- 6. Show results ----------
if "scan" in st.session_state and not st.session_state["scan"].empty:
    res = st.session_state["scan"]
    rank_col = "Combined" if "Combined" in res.columns else "Score"
    st.caption(f"Last scan: {st.session_state['scan_time']} | Ranked by: {rank_col}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scanned", len(res))
    k2.metric("Buy signals", int((res["Score"] >= buy_th).sum()))
    k3.metric("Sell signals", int((res["Score"] <= sell_th).sum()))
    k4.metric("Market breadth (Uptrend %)", f"{(res['Trend'] == 'Up').mean() * 100:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟢 Top Buy candidates")
        st.dataframe(res[res["Score"] >= buy_th].sort_values(rank_col, ascending=False).head(10),
                     hide_index=True, use_container_width=True)
    with c2:
        st.subheader("🔴 Top Sell candidates")
        st.dataframe(res[res["Score"] <= sell_th].sort_values(rank_col).head(10),
                     hide_index=True, use_container_width=True)

    # Score bar chart
    srt = res.sort_values("Score")
    fig = go.Figure(go.Bar(x=srt["Score"], y=srt["Symbol"], orientation="h",
                           marker_color=["green" if v >= buy_th else "red" if v <= sell_th else "gold"
                                         for v in srt["Score"]]))
    fig.update_layout(title="Score by stock", height=max(400, 18 * len(srt)))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full scan table"):
        st.dataframe(res.sort_values(rank_col, ascending=False), hide_index=True,
                     use_container_width=True)

    st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="scan_results.csv", mime="text/csv")

    if st.session_state["failed"]:
        st.warning("Skipped (no/insufficient data): " + ", ".join(st.session_state["failed"]))

    st.info("Tip: open a shortlisted stock on the main page to check its backtest and ML "
            "model quality before acting. Educational tool only – not investment advice.")
else:
    st.info("Click **▶ Run scan** to start.")