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
ticker = st.sidebar.text_input("Stock symbol (NSE add .NS)", "RELIANCE.NS").strip().upper()
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