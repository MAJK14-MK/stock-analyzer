import json
import os
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Watchlists", layout="wide")
st.title("⭐ My Watchlists")

FILE = "data/watchlists.json"
DEFAULT = {
    "Nifty 50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
                 "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS"],
    "My Portfolio": ["RELIANCE.NS", "TCS.NS"],
}


# ---------- 1. Load / save ----------
def load_lists():
    if os.path.exists(FILE):
        try:
            with open(FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT.copy()


def save_lists(lists):
    os.makedirs("data", exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(lists, f, indent=2)


def clean(text):
    """Turn 'tcs, infy, 500325.bo' into ['TCS.NS', 'INFY.NS', '500325.BO']."""
    out = []
    for s in text.replace("\n", ",").split(","):
        s = s.strip().upper()
        if not s:
            continue
        if not s.endswith((".NS", ".BO")) and not s.startswith("^"):
            s += ".NS"
        if s not in out:
            out.append(s)
    return out


if "watchlists" not in st.session_state:
    st.session_state["watchlists"] = load_lists()
lists = st.session_state["watchlists"]

# ---------- 2. Choose a list ----------
names = list(lists.keys())
c1, c2 = st.columns([2, 1])
current = c1.selectbox("Select watchlist", names) if names else None

with c2.popover("➕ New watchlist"):
    new_name = st.text_input("Name", placeholder="e.g. Banking")
    if st.button("Create"):
        if new_name.strip() and new_name.strip() not in lists:
            lists[new_name.strip()] = []
            save_lists(lists)
            st.rerun()
        else:
            st.error("Enter a new, unique name.")

if current:
    symbols = lists[current]
    st.caption(f"{len(symbols)} stocks in **{current}**")

    # ---------- 3. Add from the Symbol Finder selection ----------
    picked = st.session_state.get("selected_symbol")
    if picked and picked not in symbols:
        if st.button(f"➕ Add last selected stock ({picked})"):
            symbols.append(picked)
            save_lists(lists)
            st.rerun()

    # ---------- 4. Edit the list ----------
    edited = st.text_area("Symbols (comma or one per line; .NS added automatically)",
                          ", ".join(symbols), height=150)
    b1, b2, b3 = st.columns(3)
    if b1.button("💾 Save changes", type="primary"):
        lists[current] = clean(edited)
        save_lists(lists)
        st.success("Saved.")
        st.rerun()
    if b2.button("📊 Use in Scanner"):
        st.session_state["scan_list"] = current
        st.switch_page("pages/1_Scanner.py")
    if b3.button("🗑️ Delete this watchlist"):
        del lists[current]
        save_lists(lists)
        st.rerun()

    # ---------- 5. Quick price view ----------
    if symbols and st.button("🔄 Show latest prices"):
        data = yf.download(symbols, period="5d", group_by="ticker",
                           auto_adjust=True, progress=False, threads=True)
        rows = []
        for s in symbols:
            try:
                d = data[s].dropna() if len(symbols) > 1 else data.dropna()
                last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
                rows.append({"Symbol": s, "Close ₹": round(last, 2),
                             "Change %": round((last / prev - 1) * 100, 2)})
            except Exception:
                rows.append({"Symbol": s, "Close ₹": None, "Change %": None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------- 6. Backup / restore ----------
st.divider()
st.subheader("💾 Backup")
d1, d2 = st.columns(2)
d1.download_button("⬇ Download backup (JSON)", json.dumps(lists, indent=2),
                   "watchlists.json", "application/json")
up = d2.file_uploader("⬆ Restore from backup", type="json")
if up is not None and st.button("Restore"):
    st.session_state["watchlists"] = json.load(up)
    save_lists(st.session_state["watchlists"])
    st.success("Restored.")
    st.rerun()