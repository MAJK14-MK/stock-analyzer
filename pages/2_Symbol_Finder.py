import io
import requests
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Symbol Finder", layout="wide")
st.title("🔤 Stock Symbol Finder (NSE & BSE)")

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
COLS = ["Company", "Symbol", "Yahoo Code", "Exchange", "Series", "ISIN"]

# ---------- 1. Load NSE master list (official NSE file) ----------
@st.cache_data(ttl=86400, show_spinner="Loading NSE list...")
def load_nse():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"SYMBOL": "Symbol", "NAME OF COMPANY": "Company",
                            "SERIES": "Series", "ISIN NUMBER": "ISIN"})
    df["Exchange"] = "NSE"
    df["Yahoo Code"] = df["Symbol"].str.strip() + ".NS"
    return df[COLS]

# ---------- 2. Load BSE master list (BSE public API) ----------
@st.cache_data(ttl=86400, show_spinner="Loading BSE list...")
def load_bse():
    url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    params = {"Group": "", "Scripcode": "", "industry": "",
              "segment": "Equity", "status": "Active"}
    h = {**HEADERS, "Referer": "https://www.bseindia.com/",
         "Origin": "https://www.bseindia.com"}
    r = requests.get(url, params=params, headers=h, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={"SCRIP_CD": "Symbol", "Scrip_Name": "Company",
                            "GROUP": "Series", "ISIN_NUMBER": "ISIN"})
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Exchange"] = "BSE"
    df["Yahoo Code"] = df["Symbol"] + ".BO"      # Yahoo uses BSE scrip code
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]

def send_to_analyzer(code):
    st.session_state["selected_symbol"] = code
    st.switch_page("app.py")

tab1, tab2 = st.tabs(["📋 Master list search", "🌐 Live Yahoo search"])

# ================= TAB 1: Master list =================
with tab1:
    frames, errors = [], []
    for name, loader in [("NSE", load_nse), ("BSE", load_bse)]:
        try:
            frames.append(loader())
        except Exception as e:
            errors.append(f"{name}: {e}")

    if errors:
        st.warning("Could not load: " + " | ".join(errors) +
                   "  → use the 'Live Yahoo search' tab instead.")

    if frames:
        master = pd.concat(frames, ignore_index=True)

        c1, c2 = st.columns([3, 1])
        query = c1.text_input("Search company name, symbol or ISIN",
                              placeholder="e.g. reliance, tata motors, INE002A01018")
        exch = c2.multiselect("Exchange", ["NSE", "BSE"], default=["NSE", "BSE"])

        view = master[master["Exchange"].isin(exch)]
        if query:
            q = query.strip().lower()
            mask = (view["Company"].str.lower().str.contains(q, na=False, regex=False) |
                    view["Symbol"].str.lower().str.contains(q, na=False, regex=False) |
                    view["ISIN"].astype(str).str.lower().str.contains(q, na=False, regex=False))
            view = view[mask]

        m1, m2, m3 = st.columns(3)
        m1.metric("NSE stocks", int((master["Exchange"] == "NSE").sum()))
        m2.metric("BSE stocks", int((master["Exchange"] == "BSE").sum()))
        m3.metric("Matches", len(view))

        st.caption("👉 Click a row to select it")
        event = st.dataframe(view.head(500), hide_index=True, use_container_width=True,
                             on_select="rerun", selection_mode="single-row")

        rows = event.selection.rows
        if rows:
            pick = view.head(500).iloc[rows[0]]
            st.success(f"Selected: **{pick['Company']}** → Yahoo code **{pick['Yahoo Code']}**")
            b1, b2 = st.columns(2)
            if b1.button("✅ Check price"):
                d = yf.download(pick["Yahoo Code"], period="5d",
                                multi_level_index=False, progress=False)
                if d.empty:
                    st.error("Yahoo has no data for this code.")
                else:
                    st.metric("Last close", f"₹{float(d['Close'].iloc[-1]):,.2f}")
            if b2.button("📈 Open in Analyzer", type="primary"):
                send_to_analyzer(pick["Yahoo Code"])

        st.download_button("⬇ Download full list (CSV)",
                           master.to_csv(index=False).encode("utf-8"),
                           "nse_bse_symbols.csv", "text/csv")

# ================= TAB 2: Live Yahoo search =================
with tab2:
    q2 = st.text_input("Type company name", placeholder="e.g. infosys", key="yq")
    india_only = st.checkbox("India (NSE/BSE) only", True)
    if q2:
        try:
            quotes = yf.Search(q2, max_results=20).quotes
            ydf = pd.DataFrame(quotes)
            if ydf.empty:
                st.info("No results.")
            else:
                if india_only and "exchange" in ydf:
                    ydf = ydf[ydf["exchange"].isin(["NSI", "BSE", "BOM"])]
                keep = [c for c in ["symbol", "shortname", "longname", "exchange",
                                    "quoteType"] if c in ydf.columns]
                ydf = ydf[keep].rename(columns={"symbol": "Yahoo Code",
                                                "shortname": "Name"})
                ev2 = st.dataframe(ydf, hide_index=True, use_container_width=True,
                                   on_select="rerun", selection_mode="single-row")
                if ev2.selection.rows:
                    code = ydf.iloc[ev2.selection.rows[0]]["Yahoo Code"]
                    if st.button(f"📈 Open {code} in Analyzer", type="primary"):
                        send_to_analyzer(code)
        except Exception as e:
            st.error(f"Yahoo search failed: {e}")