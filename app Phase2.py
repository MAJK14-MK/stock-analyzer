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