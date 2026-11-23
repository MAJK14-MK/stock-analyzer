import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- 1. Page setup ----------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 My Stock Analyzer")

# ---------- 2. Sidebar inputs ----------
st.sidebar.header("Settings")
#ticker = st.sidebar.text_input("Stock symbol (NSE add .NS)", "RELIANCE.NS").strip().upper()
default_sym = st.session_state.get("selected_symbol", "RELIANCE.NS")
ticker = st.sidebar.text_input("Stock symbol (NSE add .NS)", default_sym).strip().upper()
period = st.sidebar.selectbox("Display period", ["3mo", "6mo", "1y", "2y"], index=2)

st.sidebar.subheader("Show on chart")
show_ema = st.sidebar.checkbox("EMA 20 / 50", True)
show_sma200 = st.sidebar.checkbox("SMA 200", True)
show_bb = st.sidebar.checkbox("Bollinger Bands", False)

# ---------- 3. Load data ----------
# Always download 3 years so long indicators (SMA 200) have enough history,
# then show only the period the user selected.
@st.cache_data(ttl=900)
def load_data(symbol):
    return yf.download(symbol, period="3y", multi_level_index=False, progress=False)

# ---------- 4. Indicator calculations ----------
def add_indicators(df):
    df = df.copy()
    close = df["Close"]

    # Moving averages
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA200"] = close.rolling(200).mean()

    # RSI (14) - Wilder's method
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (20, 2)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_Mid"] = mid
    df["BB_Up"] = mid + 2 * std
    df["BB_Low"] = mid - 2 * std
    return df

raw = load_data(ticker)
if raw.empty or len(raw) < 60:
    st.error("No/insufficient data. Check the symbol (e.g. TCS.NS, INFY.NS, ^NSEI).")
    st.stop()

full = add_indicators(raw)
days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}[period]
df = full.tail(days)

# ---------- 5. Key price numbers ----------
last, prev = df.iloc[-1], df.iloc[-2]
close_now = float(last["Close"])
change = (close_now - float(prev["Close"])) / float(prev["Close"]) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"₹{close_now:,.2f}", f"{change:.2f}%")
c2.metric("Period High", f"₹{df['High'].max():,.2f}")
c3.metric("Period Low", f"₹{df['Low'].min():,.2f}")
c4.metric("RSI (14)", f"{float(last['RSI']):.1f}")

# ---------- 6. Indicator readings (plain English) ----------
ema20, ema50, sma200 = float(last["EMA20"]), float(last["EMA50"]), last["SMA200"]
rsi = float(last["RSI"])
macd, sig = float(last["MACD"]), float(last["MACD_Signal"])
p_macd, p_sig = float(prev["MACD"]), float(prev["MACD_Signal"])

if close_now > ema20 > ema50:
    trend = "🟢 Uptrend (Price > EMA20 > EMA50)"
elif close_now < ema20 < ema50:
    trend = "🔴 Downtrend (Price < EMA20 < EMA50)"
else:
    trend = "🟡 Sideways / mixed"

if pd.isna(sma200):
    long_term = "N/A (not enough history)"
elif close_now > sma200:
    long_term = "🟢 Above SMA 200 – long-term bullish"
else:
    long_term = "🔴 Below SMA 200 – long-term bearish"

if rsi > 70:
    rsi_read = "🔴 Overbought – possible pullback"
elif rsi < 30:
    rsi_read = "🟢 Oversold – possible bounce"
else:
    rsi_read = "🟡 Neutral"

if p_macd < p_sig and macd > sig:
    macd_read = "🟢 Fresh bullish crossover"
elif p_macd > p_sig and macd < sig:
    macd_read = "🔴 Fresh bearish crossover"
elif macd > sig:
    macd_read = "🟢 Bullish momentum"
else:
    macd_read = "🔴 Bearish momentum"

bb_up, bb_low = float(last["BB_Up"]), float(last["BB_Low"])
if close_now >= bb_up:
    bb_read = "🔴 At/above upper band – stretched"
elif close_now <= bb_low:
    bb_read = "🟢 At/below lower band – weak, watch for reversal"
else:
    bb_read = "🟡 Inside bands – normal"

vol_avg = float(df["Volume"].tail(20).mean())
vol_now = float(last["Volume"])
vol_read = "🟢 High volume (>1.5× avg)" if vol_now > 1.5 * vol_avg else "🟡 Normal volume"

st.subheader("Indicator Readings")
summary = pd.DataFrame({
    "Indicator": ["Short-term trend", "Long-term trend", "RSI (14)", "MACD", "Bollinger Bands", "Volume"],
    "Value": [f"EMA20 ₹{ema20:,.1f} | EMA50 ₹{ema50:,.1f}",
              "—" if pd.isna(sma200) else f"₹{float(sma200):,.1f}",
              f"{rsi:.1f}",
              f"MACD {macd:.2f} | Signal {sig:.2f}",
              f"Low ₹{bb_low:,.1f} – Up ₹{bb_up:,.1f}",
              f"{vol_now/vol_avg:.1f}× 20-day avg"],
    "Reading": [trend, long_term, rsi_read, macd_read, bb_read, vol_read],
})
st.dataframe(summary, hide_index=True, use_container_width=True)

# ---------- 7. Multi-panel chart ----------
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.5, 0.12, 0.18, 0.2],
                    subplot_titles=(ticker, "Volume", "RSI (14)", "MACD (12,26,9)"))

# Panel 1: price + overlays
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
if show_ema:
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20",
                             line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50",
                             line=dict(color="blue", width=1.5)), row=1, col=1)
if show_sma200:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200",
                             line=dict(color="black", width=2)), row=1, col=1)
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Up"], name="BB Upper",
                             line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Low"], name="BB Lower",
                             line=dict(color="gray", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

# Panel 2: volume (green = up day, red = down day)
vol_colors = ["green" if c >= o else "red" for o, c in zip(df["Open"], df["Close"])]
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors,
                     name="Volume", showlegend=False), row=2, col=1)

# Panel 3: RSI with 70/30 lines
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                         line=dict(color="purple")), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# Panel 4: MACD
hist_colors = ["green" if h >= 0 else "red" for h in df["MACD_Hist"]]
fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors,
                     name="Histogram", showlegend=False), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                         line=dict(color="blue")), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                         line=dict(color="orange")), row=4, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False,
                  legend=dict(orientation="h", y=1.04))
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
st.plotly_chart(fig, use_container_width=True)

# ---------- 8. Raw data ----------
with st.expander("Show data with indicators"):
    st.dataframe(df.tail(20).round(2))

# ===================== PHASE 3 =====================

# ---------- 9. Signal score ----------
st.subheader("🎯 Signal Score (rule-based)")

rows = []
def add(name, pts, maxp, why):
    rows.append({"Factor": name, "Points": pts, "Max": maxp, "Why": why})

# Short-term trend
if close_now > ema20 > ema50:
    add("Short-term trend", 2, 2, "Price > EMA20 > EMA50")
elif close_now < ema20 < ema50:
    add("Short-term trend", -2, 2, "Price < EMA20 < EMA50")
else:
    add("Short-term trend", 0, 2, "Mixed")

# Long-term trend
if pd.isna(sma200):
    add("Long-term trend", 0, 1, "Not enough history")
elif close_now > float(sma200):
    add("Long-term trend", 1, 1, "Above SMA 200")
else:
    add("Long-term trend", -1, 1, "Below SMA 200")

# RSI
if rsi < 30:
    add("RSI", 2, 2, "Oversold – bounce possible")
elif rsi > 70:
    add("RSI", -2, 2, "Overbought – pullback risk")
elif rsi >= 50:
    add("RSI", 1, 2, "Positive momentum (50–70)")
else:
    add("RSI", -1, 2, "Weak momentum (30–50)")

# MACD
if p_macd < p_sig and macd > sig:
    add("MACD", 2, 2, "Fresh bullish crossover")
elif p_macd > p_sig and macd < sig:
    add("MACD", -2, 2, "Fresh bearish crossover")
elif macd > sig:
    add("MACD", 1, 2, "MACD above signal")
else:
    add("MACD", -1, 2, "MACD below signal")

# Bollinger Bands
if close_now <= bb_low:
    add("Bollinger Bands", 1, 1, "At lower band")
elif close_now >= bb_up:
    add("Bollinger Bands", -1, 1, "At upper band")
else:
    add("Bollinger Bands", 0, 1, "Inside bands")

# Volume
up_day = close_now >= float(last["Open"])
if vol_now > 1.5 * vol_avg:
    add("Volume", 1 if up_day else -1, 1,
        "High volume on " + ("up" if up_day else "down") + " day")
else:
    add("Volume", 0, 1, "Normal volume")

score_df = pd.DataFrame(rows)
score = int(round(score_df["Points"].sum() / score_df["Max"].sum() * 100))

if score >= 40:
    signal = "🟢 STRONG BUY"
elif score >= 15:
    signal = "🟢 BUY"
elif score <= -40:
    signal = "🔴 STRONG SELL"
elif score <= -15:
    signal = "🔴 SELL"
else:
    signal = "🟡 HOLD"

s1, s2 = st.columns(2)
s1.metric("Signal", signal)
s2.metric("Score (-100 to +100)", score)
st.progress((score + 100) / 200)
st.dataframe(score_df, hide_index=True, use_container_width=True)

# ---------- 10. Trade plan (ATR based) ----------
st.subheader("🛡️ Trade Plan (ATR-based)")

st.sidebar.subheader("Risk settings")
capital = st.sidebar.number_input("Capital (₹)", min_value=1000, value=100000, step=5000)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)

h, l, c = full["High"], full["Low"], full["Close"]
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])

long_side = score >= 0
entry = close_now
stop = entry - 1.5 * atr if long_side else entry + 1.5 * atr
target = entry + 3 * atr if long_side else entry - 3 * atr
risk_per_share = abs(entry - stop)
qty = int(min((capital * risk_pct / 100) // risk_per_share, capital // entry)) if risk_per_share > 0 else 0

t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Direction", "LONG (Buy)" if long_side else "SHORT (Sell)")
t2.metric("Entry", f"₹{entry:,.2f}")
t3.metric("Stop-loss", f"₹{stop:,.2f}")
t4.metric("Target (1:2)", f"₹{target:,.2f}")
t5.metric("Quantity", f"{qty}")
st.caption(f"ATR(14) = ₹{atr:,.2f} | Max loss ≈ ₹{qty * risk_per_share:,.0f} "
           f"({risk_pct}% of capital). Short selling in cash market is intraday only.")

# ---------- 11. Brokerage & charges calculator ----------
st.subheader("💰 Brokerage & Charges Calculator")

k1, k2, k3, k4 = st.columns(4)
trade_type = k1.selectbox("Trade type", ["Delivery", "Intraday"])
buy_px = k2.number_input("Buy price (₹)", min_value=0.0,
                         value=float(round(entry if long_side else target, 2)))
sell_px = k3.number_input("Sell price (₹)", min_value=0.0,
                          value=float(round(target if long_side else entry, 2)))
qty_in = k4.number_input("Quantity", min_value=1, value=max(qty, 1), step=1)

with st.expander("Charge rates (edit as per your broker's contract note)"):
    r1, r2, r3 = st.columns(3)
    brok_order = r1.number_input("Brokerage per order (₹)", min_value=0.0, value=20.0)
    dp_charge = r2.number_input("DP charge per sell – delivery (₹)", min_value=0.0, value=15.0)
    exch_pct = r3.number_input("Exchange txn charge (%)", min_value=0.0,
                               value=0.00297, format="%.5f")

buy_val, sell_val = buy_px * qty_in, sell_px * qty_in
turnover = buy_val + sell_val
brokerage = 2 * brok_order

if trade_type == "Delivery":
    stt = 0.001 * turnover        # 0.1% on buy and sell
    stamp = 0.00015 * buy_val     # 0.015% on buy
    dp = dp_charge
else:
    stt = 0.00025 * sell_val      # 0.025% on sell
    stamp = 0.00003 * buy_val     # 0.003% on buy
    dp = 0.0

exch = exch_pct / 100 * turnover
sebi = 0.000001 * turnover        # ₹10 per crore
gst = 0.18 * (brokerage + exch + sebi)
total_charges = brokerage + stt + exch + sebi + gst + stamp + dp
gross = sell_val - buy_val
net = gross - total_charges

charges = pd.DataFrame({
    "Item": ["Brokerage", "STT", "Exchange txn", "SEBI fee", "GST (18%)",
             "Stamp duty", "DP charge", "TOTAL CHARGES", "Gross P&L", "NET P&L"],
    "Amount (₹)": [brokerage, stt, exch, sebi, gst, stamp, dp,
                   total_charges, gross, net],
})
charges["Amount (₹)"] = charges["Amount (₹)"].round(2)

m1, m2, m3 = st.columns(3)
m1.metric("Gross P&L", f"₹{gross:,.2f}")
m2.metric("Total charges", f"₹{total_charges:,.2f}")
m3.metric("Net P&L", f"₹{net:,.2f}")
st.caption(f"Break-even move: ₹{total_charges / qty_in:,.2f} per share")
st.dataframe(charges, hide_index=True, use_container_width=True)

st.warning("Educational tool only – not investment advice. Always backtest before trading.")

# ===================== PHASE 4 =====================
import numpy as np

st.header("🧪 Backtest – How did this score perform in the past?")

# ---------- 12. Backtest settings ----------
st.sidebar.subheader("Backtest settings")
bt_capital = st.sidebar.number_input("Backtest capital (₹)", min_value=10000, value=100000, step=10000)
entry_th = st.sidebar.slider("Buy when score ≥", 0, 80, 15, 5)
exit_th = st.sidebar.slider("Exit when score ≤", -80, 0, -15, 5)
max_hold = st.sidebar.slider("Max holding days", 3, 60, 20)

# ---------- 13. Score for every day (same rules as Phase 3) ----------
def score_series(d):
    c = d["Close"]
    e20, e50, s200 = d["EMA20"], d["EMA50"], d["SMA200"]
    trend = np.select([(c > e20) & (e20 > e50), (c < e20) & (e20 < e50)], [2, -2], 0)
    longt = np.where(s200.isna(), 0, np.where(c > s200, 1, -1))
    r = d["RSI"]
    rsi_p = np.select([r < 30, r > 70, r >= 50], [2, -2, 1], -1)
    m, s = d["MACD"], d["MACD_Signal"]
    cross_up = (m.shift() < s.shift()) & (m > s)
    cross_dn = (m.shift() > s.shift()) & (m < s)
    macd_p = np.select([cross_up, cross_dn, m > s], [2, -2, 1], -1)
    bb_p = np.select([c <= d["BB_Low"], c >= d["BB_Up"]], [1, -1], 0)
    v_avg = d["Volume"].rolling(20).mean()
    vol_p = np.where(d["Volume"] > 1.5 * v_avg, np.where(c >= d["Open"], 1, -1), 0)
    total = trend + longt + rsi_p + macd_p + bb_p + vol_p
    return pd.Series(np.round(total / 9 * 100), index=d.index)

# ---------- 14. Delivery charges (rates from Phase 3 expander) ----------
def delivery_charges(buy_px, sell_px, q):
    bv, sv = buy_px * q, sell_px * q
    to = bv + sv
    br = 2 * brok_order
    stt_c = 0.001 * to
    stamp_c = 0.00015 * bv
    ex_c = exch_pct / 100 * to
    sebi_c = 0.000001 * to
    gst_c = 0.18 * (br + ex_c + sebi_c)
    return br + stt_c + stamp_c + ex_c + sebi_c + gst_c + dp_charge

# ---------- 15. Run the backtest ----------
bt = full.copy()
bt["Score"] = score_series(bt)
bt["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
bt = bt.iloc[50:]          # skip warm-up days where indicators are unstable

O, H, L, C = (bt[k].to_numpy(dtype=float) for k in ["Open", "High", "Low", "Close"])
sc, atr_a, dates = bt["Score"].to_numpy(), bt["ATR"].to_numpy(), bt.index

cash = float(bt_capital)
in_pos = pending_entry = pending_exit = False
bt_qty = entry_i = 0
entry_px = stop_px = tgt_px = 0.0
trades, equity = [], []

for i in range(len(bt)):
    # 1) Manage open position
    if in_pos:
        exit_px, reason = None, ""
        if pending_exit:
            exit_px, reason = O[i], "Score exit"
        elif i - entry_i >= max_hold:
            exit_px, reason = O[i], "Time exit"
        elif O[i] <= stop_px:
            exit_px, reason = O[i], "Stop (gap down)"
        elif O[i] >= tgt_px:
            exit_px, reason = O[i], "Target (gap up)"
        elif L[i] <= stop_px:                       # stop checked first = conservative
            exit_px, reason = stop_px, "Stop-loss"
        elif H[i] >= tgt_px:
            exit_px, reason = tgt_px, "Target"

        if exit_px is not None:
            chg = delivery_charges(entry_px, exit_px, bt_qty)
            pnl = (exit_px - entry_px) * bt_qty - chg
            cash += exit_px * bt_qty - chg
            trades.append({"Entry date": dates[entry_i].date(), "Entry ₹": round(entry_px, 2),
                           "Exit date": dates[i].date(), "Exit ₹": round(float(exit_px), 2),
                           "Qty": bt_qty, "Days": i - entry_i, "Exit reason": reason,
                           "Charges ₹": round(chg, 2), "Net P&L ₹": round(pnl, 2)})
            in_pos = False

    # 2) New entry at today's open (signal came from yesterday's close)
    elif pending_entry:
        q = int(cash // O[i])
        if q > 0:
            in_pos, bt_qty, entry_i, entry_px = True, q, i, O[i]
            stop_px = entry_px - 1.5 * atr_a[i - 1]
            tgt_px = entry_px + 3 * atr_a[i - 1]
            cash -= entry_px * bt_qty

    # 3) End-of-day signal check (acted on next day)
    if in_pos:
        pending_exit = sc[i] <= exit_th
    else:
        pending_entry = sc[i] >= entry_th

    equity.append(cash + (bt_qty * C[i] if in_pos else 0))

# ---------- 16. Results ----------
eq = pd.Series(equity, index=dates)
bh = bt_capital * C / C[0]
strat_ret = (eq.iloc[-1] / bt_capital - 1) * 100
bh_ret = (C[-1] / C[0] - 1) * 100
max_dd = ((eq / eq.cummax()) - 1).min() * 100
tdf = pd.DataFrame(trades)

st.caption(f"Test period: {dates[0].date()} to {dates[-1].date()}  |  "
           f"Buy ≥ {entry_th}, Exit ≤ {exit_th}, Max hold {max_hold} days")

if tdf.empty:
    st.info("No trades triggered. Try lowering the 'Buy when score ≥' threshold.")
else:
    wins = tdf[tdf["Net P&L ₹"] > 0]
    losses = tdf[tdf["Net P&L ₹"] <= 0]
    win_rate = len(wins) / len(tdf) * 100
    pf = wins["Net P&L ₹"].sum() / abs(losses["Net P&L ₹"].sum()) if len(losses) else float("inf")

    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Strategy return", f"{strat_ret:.1f}%")
    b2.metric("Buy & Hold return", f"{bh_ret:.1f}%")
    b3.metric("Trades", len(tdf))
    b4.metric("Win rate", f"{win_rate:.0f}%")
    b5.metric("Profit factor", f"{pf:.2f}")
    b6.metric("Max drawdown", f"{max_dd:.1f}%")
    st.caption(f"Total charges paid: ₹{tdf['Charges ₹'].sum():,.0f}  |  "
               f"Avg win ₹{wins['Net P&L ₹'].mean() if len(wins) else 0:,.0f}  |  "
               f"Avg loss ₹{losses['Net P&L ₹'].mean() if len(losses) else 0:,.0f}")

    # Verdict
    if pf > 1.3 and strat_ret > bh_ret:
        st.success("✅ Strategy beat Buy & Hold with a healthy profit factor on this stock.")
    elif pf > 1.0:
        st.info("🟡 Profitable after charges, but check if it beats Buy & Hold and drawdown is acceptable.")
    else:
        st.error("❌ Lost money after charges on this stock – do not trade these settings.")

    # Equity curve
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=dates, y=eq, name="Strategy", line=dict(color="green", width=2)))
    fig_eq.add_trace(go.Scatter(x=dates, y=bh, name="Buy & Hold", line=dict(color="gray", dash="dot")))
    fig_eq.update_layout(title="Equity Curve (₹)", height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig_eq, use_container_width=True)

    # Price chart with trade markers
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=dates, y=C, name="Close", line=dict(color="black", width=1)))
    fig_tr.add_trace(go.Scatter(x=pd.to_datetime(tdf["Entry date"]), y=tdf["Entry ₹"], mode="markers",
                                name="Buy", marker=dict(symbol="triangle-up", size=11, color="green")))
    fig_tr.add_trace(go.Scatter(x=pd.to_datetime(tdf["Exit date"]), y=tdf["Exit ₹"], mode="markers",
                                name="Sell", marker=dict(symbol="triangle-down", size=11, color="red")))
    fig_tr.update_layout(title="Trades on Price Chart", height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig_tr, use_container_width=True)

    with st.expander(f"Show all {len(tdf)} trades"):
        st.dataframe(tdf, hide_index=True, use_container_width=True)

if in_pos:
    st.caption(f"ℹ️ A position is still open from {dates[entry_i].date()} at ₹{entry_px:,.2f} "
               f"(valued at today's close in the equity curve).")

# ===================== PHASE 5 =====================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.inspection import permutation_importance

st.header("🤖 ML Prediction – Probability of Price Going Up")

# ---------- 17. ML settings ----------
st.sidebar.subheader("ML settings")
horizon = st.sidebar.slider("Predict days ahead", 1, 10, 5)
min_move = st.sidebar.slider("Count as 'Up' if gain ≥ (%)", 0.0, 3.0, 1.0, 0.5)
prob_th = st.sidebar.slider("ML buy when probability ≥ (%)", 50, 80, 60, 5)

# ---------- 18. Features (inputs to the model) ----------
def make_features(d):
    c = d["Close"]
    f = pd.DataFrame(index=d.index)
    f["Ret_1d"] = c.pct_change()
    f["Ret_5d"] = c.pct_change(5)
    f["Ret_20d"] = c.pct_change(20)
    f["Dist_EMA20"] = c / d["EMA20"] - 1
    f["Dist_EMA50"] = c / d["EMA50"] - 1
    f["Dist_SMA200"] = c / d["SMA200"] - 1
    f["EMA20_vs_EMA50"] = d["EMA20"] / d["EMA50"] - 1
    f["RSI"] = d["RSI"]
    f["MACD_Hist_pct"] = d["MACD_Hist"] / c
    f["BB_Position"] = (c - d["BB_Low"]) / (d["BB_Up"] - d["BB_Low"])
    f["Volume_Ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    trr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(),
                     (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    f["ATR_pct"] = trr.ewm(alpha=1/14, adjust=False).mean() / c
    f["Volatility_20d"] = f["Ret_1d"].rolling(20).std()
    f["Up_Days_10"] = (c.diff() > 0).rolling(10).sum()
    f["Rule_Score"] = score_series(d)
    return f.replace([np.inf, -np.inf], np.nan)

def new_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=30, random_state=42)

# ---------- 19. Train & test (cached so it runs once per setting) ----------
@st.cache_data(ttl=3600, show_spinner="Training ML model...")
def train_model(symbol, horizon, min_move):
    d = yf.download(symbol, period="10y", multi_level_index=False, progress=False)
    if d.empty or len(d) < 400:
        return None
    d = add_indicators(d)
    X = make_features(d)
    fut_ret = d["Close"].shift(-horizon) / d["Close"] - 1
    y = (fut_ret >= min_move / 100).astype(int)

    valid = fut_ret.notna() & X["Volatility_20d"].notna()
    Xv, yv = X[valid], y[valid]
    split = int(len(Xv) * 0.7)
    X_tr, y_tr = Xv.iloc[:split - horizon], yv.iloc[:split - horizon]   # gap avoids leakage
    X_te, y_te = Xv.iloc[split:], yv.iloc[split:]

    m = new_model().fit(X_tr, y_tr)
    p_te = m.predict_proba(X_te)[:, 1]

    imp = permutation_importance(m, X_te, y_te, n_repeats=5, random_state=42, scoring="roc_auc")
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)

    final = new_model().fit(Xv, yv)                      # use all history for today
    p_today = float(final.predict_proba(X.iloc[[-1]])[:, 1][0])

    return {"d": d[["Open", "Close"]], "p_te": pd.Series(p_te, index=X_te.index),
            "y_te": y_te, "p_today": p_today, "importance": importance,
            "train_from": X_tr.index[0].date(), "test_from": X_te.index[0].date()}

res = train_model(ticker, horizon, min_move)

if res is None:
    st.info("Not enough history (need ~2 years+) for ML on this symbol.")
else:
    p_te, y_te = res["p_te"], res["y_te"]
    pred = (p_te >= prob_th / 100).astype(int)

    # ---------- 20. Today's prediction ----------
    p_now = res["p_today"] * 100
    ml_view = "🟢 Bullish" if p_now >= prob_th else ("🔴 Bearish" if p_now <= 100 - prob_th else "🟡 Neutral")
    rule_view = "🟢" if score >= 15 else ("🔴" if score <= -15 else "🟡")
    agree = "✅ Both agree" if ml_view[0] == rule_view else "⚠️ Models disagree – stay cautious"

    a1, a2, a3 = st.columns(3)
    a1.metric(f"P(≥{min_move}% up in {horizon} days)", f"{p_now:.0f}%")
    a2.metric("ML view", ml_view)
    a3.metric("ML vs Rule score", agree)
    st.progress(p_now / 100)

    # ---------- 21. Model quality on unseen data ----------
    st.subheader("📏 Model quality (tested on unseen data)")
    base_rate = y_te.mean() * 100
    acc = accuracy_score(y_te, (p_te >= 0.5).astype(int)) * 100
    prec = precision_score(y_te, pred, zero_division=0) * 100
    auc = roc_auc_score(y_te, p_te) if y_te.nunique() > 1 else 0.5

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Base rate (random 'Up')", f"{base_rate:.0f}%")
    q2.metric(f"Precision at ≥{prob_th}%", f"{prec:.0f}%", f"{prec - base_rate:+.0f} pts vs base")
    q3.metric("Accuracy", f"{acc:.0f}%")
    q4.metric("AUC (0.5 = coin toss)", f"{auc:.2f}")
    st.caption(f"Trained from {res['train_from']} | Tested from {res['test_from']} | "
               f"Signals in test: {int(pred.sum())}")

    if auc >= 0.58 and prec - base_rate >= 5:
        st.success("✅ Model shows a real edge over random on unseen data for this stock.")
    elif auc >= 0.53:
        st.info("🟡 Weak edge – use only as a supporting filter with the rule score.")
    else:
        st.error("❌ No edge – model is close to a coin toss on this stock. Ignore its signal.")

    # ---------- 22. ML backtest on test period ----------
    st.subheader("🧪 ML strategy backtest (test period only)")
    dd = res["d"]
    O5, C5 = dd["Open"].to_numpy(float), dd["Close"].to_numpy(float)
    pos_idx = dd.index.get_indexer(p_te.index)
    cash5, next_free, trades5 = float(bt_capital), 0, []

    for k, i in enumerate(pos_idx):
        if i < next_free or p_te.iloc[k] < prob_th / 100:
            continue
        ent, ex = i + 1, i + horizon          # buy next open, sell at close after horizon
        if ex >= len(dd):
            break
        q = int(cash5 // O5[ent])
        if q == 0:
            continue
        chg = delivery_charges(O5[ent], C5[ex], q)
        pnl = (C5[ex] - O5[ent]) * q - chg
        cash5 += pnl
        trades5.append({"Entry date": dd.index[ent].date(), "Entry ₹": round(O5[ent], 2),
                        "Exit date": dd.index[ex].date(), "Exit ₹": round(C5[ex], 2),
                        "Qty": q, "Prob %": round(p_te.iloc[k] * 100, 1),
                        "Charges ₹": round(chg, 2), "Net P&L ₹": round(pnl, 2)})
        next_free = ex + 1

    t5 = pd.DataFrame(trades5)
    i0 = pos_idx[0]
    bh5 = (C5[-1] / C5[i0] - 1) * 100
    if t5.empty:
        st.info("No ML trades in test period. Try lowering the probability threshold.")
    else:
        ml_ret = (cash5 / bt_capital - 1) * 100
        w5 = t5[t5["Net P&L ₹"] > 0]
        l5 = t5[t5["Net P&L ₹"] <= 0]
        pf5 = w5["Net P&L ₹"].sum() / abs(l5["Net P&L ₹"].sum()) if len(l5) else float("inf")
        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("ML strategy return", f"{ml_ret:.1f}%")
        n2.metric("Buy & Hold (same period)", f"{bh5:.1f}%")
        n3.metric("Trades", len(t5))
        n4.metric("Win rate", f"{len(w5) / len(t5) * 100:.0f}%")
        n5.metric("Profit factor", f"{pf5:.2f}")
        with st.expander(f"Show all {len(t5)} ML trades"):
            st.dataframe(t5, hide_index=True, use_container_width=True)

    # ---------- 23. What drives the model ----------
    st.subheader("🔍 What the model relies on most")
    imp = res["importance"].head(10)
    fig_imp = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h",
                               marker_color="teal"))
    fig_imp.update_layout(height=380, xaxis_title="Drop in AUC when feature is shuffled")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Higher bar = model depends more on that input. Near-zero/negative = not useful.")