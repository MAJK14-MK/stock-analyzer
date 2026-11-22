import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# 1. Page title
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 My Stock Analyzer")

# 2. User inputs (sidebar)
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock symbol (NSE add .NS)", "RELIANCE.NS")
period = st.sidebar.selectbox("Time period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# 3. Download data (cached so it doesn't reload every click)
@st.cache_data
def load_data(symbol, period):
    return yf.download(symbol, period=period, multi_level_index=False)

df = load_data(ticker, period)

# 4. Handle wrong symbol
if df.empty:
    st.error("No data found. Check the symbol (e.g. TCS.NS, INFY.NS).")
    st.stop()

# 5. Key numbers
last = df["Close"].iloc[-1]
prev = df["Close"].iloc[-2]
change = (last - prev) / prev * 100

c1, c2, c3 = st.columns(3)
c1.metric("Last Close", f"₹{last:,.2f}", f"{change:.2f}%")
c2.metric("Period High", f"₹{df['High'].max():,.2f}")
c3.metric("Period Low", f"₹{df['Low'].min():,.2f}")

# 6. Candlestick chart
fig = go.Figure(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name=ticker))
fig.update_layout(height=550, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# 7. Raw data table
with st.expander("Show raw data"):
    st.dataframe(df.tail(20))