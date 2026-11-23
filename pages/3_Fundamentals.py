import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Fundamentals", layout="wide")
st.title("🏦 Fundamentals")

default_sym = st.session_state.get("selected_symbol", "RELIANCE.NS")
ticker = st.text_input("Yahoo code (e.g. TCS.NS, 500325.BO)", default_sym).strip().upper()


# ---------- 1. Load data (cached 6 hours) ----------
@st.cache_data(ttl=21600, show_spinner="Loading fundamentals...")
def load_fund(symbol):
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    def safe(fn):
        try:
            df = fn()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    return {"info": info,
            "inc": safe(lambda: t.income_stmt),
            "qinc": safe(lambda: t.quarterly_income_stmt),
            "holders": safe(lambda: t.major_holders)}


data = load_fund(ticker)
info, inc, qinc = data["info"], data["inc"], data["qinc"]

if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
    st.error("No fundamental data found. Check the code in Symbol Finder.")
    st.stop()


# ---------- 2. Helpers ----------
def g(key):
    v = info.get(key)
    if isinstance(v, (int, float)) and not pd.isna(v):
        return float(v)
    return None


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def num(v, d=2):
    return "—" if v is None else f"{v:,.{d}f}"


def row_growth(df, row):
    """Returns (YoY growth, 3-year CAGR) from an annual statement row."""
    if df.empty or row not in df.index:
        return None, None
    s = df.loc[row].dropna()
    s = s.sort_index(ascending=False)          # newest first
    yoy = cagr = None
    if len(s) >= 2 and s.iloc[1] > 0:
        yoy = s.iloc[0] / s.iloc[1] - 1
    if len(s) >= 4 and s.iloc[3] > 0 and s.iloc[0] > 0:
        cagr = (s.iloc[0] / s.iloc[3]) ** (1 / 3) - 1
    return yoy, cagr


# ---------- 3. Company header ----------
price = g("currentPrice") or g("regularMarketPrice")
mcap = g("marketCap")
st.subheader(info.get("longName", ticker))
st.caption(f"{info.get('sector', '—')} | {info.get('industry', '—')}")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Price", f"₹{num(price)}")
h2.metric("Market cap", "—" if mcap is None else f"₹{mcap / 1e7:,.0f} Cr")
h3.metric("52-week high", f"₹{num(g('fiftyTwoWeekHigh'))}")
h4.metric("52-week low", f"₹{num(g('fiftyTwoWeekLow'))}")

# ---------- 4. Key ratios ----------
pe, pb = g("trailingPE"), g("priceToBook")
roe, margin = g("returnOnEquity"), g("profitMargins")
de, cr = g("debtToEquity"), g("currentRatio")     # yfinance D/E is in % (35 = 0.35×)
rev_yoy, rev_cagr = row_growth(inc, "Total Revenue")
np_yoy, np_cagr = row_growth(inc, "Net Income")

st.subheader("📊 Key ratios")
ratios = pd.DataFrame({
    "Group": ["Valuation", "Valuation", "Valuation", "Profitability", "Profitability",
              "Balance sheet", "Balance sheet", "Growth", "Growth", "Growth", "Growth"],
    "Metric": ["P/E (trailing)", "P/B", "EPS (₹)", "ROE", "Net profit margin",
               "Debt / Equity", "Current ratio", "Revenue growth (YoY)",
               "Revenue CAGR (3y)", "Profit growth (YoY)", "Profit CAGR (3y)"],
    "Value": [num(pe, 1), num(pb, 1), num(g("trailingEps")), pct(roe), pct(margin),
              "—" if de is None else f"{de / 100:.2f}×", num(cr),
              pct(rev_yoy), pct(rev_cagr), pct(np_yoy), pct(np_cagr)],
})
st.dataframe(ratios, hide_index=True, use_container_width=True)

# ---------- 5. Fundamental score ----------
rows = []


def rule(name, value, good, bad, why_good, why_bad):
    if value is None:
        return
    if good(value):
        rows.append({"Check": name, "Points": 1, "Reading": "🟢 " + why_good})
    elif bad(value):
        rows.append({"Check": name, "Points": -1, "Reading": "🔴 " + why_bad})
    else:
        rows.append({"Check": name, "Points": 0, "Reading": "🟡 Average"})


rule("Valuation (P/E)", pe, lambda v: 0 < v < 25, lambda v: v > 50 or v <= 0,
     "Reasonably valued", "Expensive or loss-making")
rule("ROE", roe, lambda v: v >= 0.15, lambda v: v < 0.08,
     "Strong return on equity (≥15%)", "Weak return on equity (<8%)")
rule("Profit margin", margin, lambda v: v >= 0.10, lambda v: v < 0,
     "Healthy margin (≥10%)", "Loss-making")
rule("Debt / Equity", de, lambda v: v < 50, lambda v: v > 150,
     "Low debt", "High debt")
rule("Revenue growth", rev_yoy, lambda v: v >= 0.10, lambda v: v < 0,
     "Growing ≥10%", "Revenue falling")
rule("Profit growth", np_yoy, lambda v: v >= 0.10, lambda v: v < 0,
     "Growing ≥10%", "Profit falling")
rule("Liquidity (current ratio)", cr, lambda v: v >= 1.5, lambda v: v < 1,
     "Comfortable", "Tight liquidity")

st.subheader("🎯 Fundamental score")
if rows:
    fdf = pd.DataFrame(rows)
    fscore = int(round(fdf["Points"].sum() / len(fdf) * 100))
    if fscore >= 40:
        verdict = "🟢 Strong fundamentals"
    elif fscore >= 0:
        verdict = "🟡 Average fundamentals"
    else:
        verdict = "🔴 Weak fundamentals"
    f1, f2 = st.columns(2)
    f1.metric("Verdict", verdict)
    f2.metric("Score (-100 to +100)", fscore)
    st.progress((fscore + 100) / 200)
    st.dataframe(fdf, hide_index=True, use_container_width=True)
    st.session_state["fund_score"] = {ticker: fscore}
    if info.get("sector") == "Financial Services":
        st.caption("ℹ️ Banks/NBFCs: Debt/Equity and current ratio are not meaningful – judge on ROE and growth.")
else:
    st.info("Not enough data to score this stock.")

# ---------- 6. Quarterly revenue & profit ----------
st.subheader("📈 Quarterly revenue & net profit (₹ Cr)")
if not qinc.empty and "Total Revenue" in qinc.index and "Net Income" in qinc.index:
    q = qinc.loc[["Total Revenue", "Net Income"]].T.sort_index().dropna(how="all") / 1e7
    labels = [d.strftime("%b-%y") for d in q.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=q["Total Revenue"], name="Revenue", marker_color="steelblue"))
    fig.add_trace(go.Bar(x=labels, y=q["Net Income"], name="Net profit", marker_color="seagreen"))
    fig.update_layout(barmode="group", height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Quarterly results not available for this stock.")

# ---------- 7. Shareholding ----------
holders = data["holders"]
if not holders.empty:
    with st.expander("👥 Shareholding (as reported by Yahoo)"):
        st.dataframe(holders, use_container_width=True)
        st.caption("Insider % is close to promoter holding for most Indian companies, "
                   "but confirm on NSE/BSE shareholding pattern before relying on it.")

st.caption("Source: Yahoo Finance. Data can lag or be missing for smaller companies. "
           "Educational tool only – not investment advice.")