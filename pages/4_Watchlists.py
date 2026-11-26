import io
import json
import os
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Watchlists", layout="wide")

FILE = "data/watchlists.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
GREEN, RED, GREY = "#16a34a", "#dc2626", "#6b7280"
NA = (None, None, None, None)


# =====================================================================
# 1. SAVE / LOAD GROUPS  (same file the Scanner uses)
# =====================================================================

#from watchlist_store import load_lists, save_lists
from watchlist_store import load_lists, save_lists, storage_mode



if "watchlists" not in st.session_state:
    st.session_state["watchlists"] = load_lists()
lists = st.session_state["watchlists"]
st.caption(f"💾 Watchlists saved to: **{storage_mode()}**")


# =====================================================================
# 2. MASTER STOCK LIST FOR SEARCH (NSE live + saved BSE list)
# =====================================================================
@st.cache_data(ttl=86400, show_spinner="Loading stock list...")
def load_master():
    frames = []
    try:
        r = requests.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv",
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        n = pd.read_csv(io.StringIO(r.text))
        n.columns = n.columns.str.strip()
        sym = n["SYMBOL"].astype(str).str.strip()
        frames.append(pd.DataFrame({"Code": sym + ".NS", "Short": sym,
                                    "Company": n["NAME OF COMPANY"].astype(str), "Exch": "NSE"}))
    except Exception:
        pass
    if os.path.exists("data/bse_list.csv"):
        b = pd.read_csv("data/bse_list.csv", dtype=str)
        code = b["SCRIP_CD"].astype(str).str.strip()
        short = b["scrip_id"].fillna(code) if "scrip_id" in b.columns else code
        name = b["Scrip_Name"] if "Scrip_Name" in b.columns else short
        frames.append(pd.DataFrame({"Code": code + ".BO", "Short": short.str.strip(),
                                    "Company": name.astype(str), "Exch": "BSE"}))
    if not frames:
        return pd.DataFrame(columns=["Code", "Short", "Company", "Exch", "Label"])
    m = pd.concat(frames, ignore_index=True).drop_duplicates("Code")
    m["Label"] = m["Short"] + "  ·  " + m["Company"] + "  ·  " + m["Exch"]
    return m


master = load_master()
short_name = dict(zip(master["Code"], master["Short"]))   # 500325.BO -> RELIANCE


# =====================================================================
# 3. LIVE PRICES (cached 60 seconds) with BSE fallback
# =====================================================================
def fetch(cs):
    res = {}
    if not cs:
        return res
    d = yf.download(list(cs), period="5d", group_by="ticker", auto_adjust=False,
                    progress=False, threads=True)
    for c in cs:
        try:
            x = d[c] if isinstance(d.columns, pd.MultiIndex) else d
            cl = x["Close"].dropna()
            if len(cl) < 2:
                continue
            ltp, prev = float(cl.iloc[-1]), float(cl.iloc[-2])
            res[c] = (ltp, ltp - prev, (ltp / prev - 1) * 100)
        except Exception:
            pass
    return res


@st.cache_data(ttl=60, show_spinner=False)
def quotes(codes):
    """Returns {code: (LTP, change, change %, source code used)}.
    If a code has no Yahoo data, tries NAME.BO, then NAME.NS."""
    got = fetch(tuple(codes))
    out = {c: got[c] + (c,) for c in codes if c in got}

    alt = {}
    for c in codes:
        if c in out or c.startswith("^"):
            continue
        name = str(short_name.get(c, c.split(".")[0])).strip().upper()
        alt[c] = [x for x in (name + ".BO", name + ".NS") if x != c]

    cands = tuple(dict.fromkeys(x for v in alt.values() for x in v))
    got2 = fetch(cands)
    for c, options in alt.items():
        for o in options:
            if o in got2:
                out[c] = got2[o] + (o,)
                break
    return out


# =====================================================================
# 4. INDEX BAR
# =====================================================================
idx = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
iq = quotes(tuple(idx.values()))
cols = st.columns(len(idx) + 1)
for col, (name, code) in zip(cols, idx.items()):
    ltp, chg, pct, _ = iq.get(code, NA)
    if ltp is None:
        col.metric(name, "—")
    else:
        col.metric(name, f"{ltp:,.2f}", f"{chg:+,.2f} ({pct:+.2f}%)")
if cols[-1].button("🔄 Refresh prices"):
    quotes.clear()
    st.rerun()

st.divider()

# =====================================================================
# 5. GROUP BAR – select / new / rename / delete
# =====================================================================
names = list(lists.keys())
cur = st.session_state.get("wl_group")
if cur not in names:
    cur = names[0]

g1, g2, g3, g4 = st.columns([5, 1.2, 1.2, 1.2])
cur = g1.radio("Group", names, index=names.index(cur), horizontal=True,
               format_func=lambda n: f"{n} ({len(lists[n])})", label_visibility="collapsed")
st.session_state["wl_group"] = cur

with g2.popover("＋ New group"):
    new_name = st.text_input("Group name", key="new_grp")
    if st.button("Create", key="create_grp"):
        n = new_name.strip()
        if n and n not in lists:
            lists[n] = []
            save_lists(lists)
            st.session_state["wl_group"] = n
            st.rerun()
        else:
            st.error("Enter a new, unique name.")

with g3.popover("✏️ Rename"):
    ren = st.text_input("New name", value=cur, key="ren_grp")
    if st.button("Save name", key="save_ren"):
        r = ren.strip()
        if r and r not in lists:
            st.session_state["watchlists"] = {(r if k == cur else k): v for k, v in lists.items()}
            save_lists(st.session_state["watchlists"])
            st.session_state["wl_group"] = r
            st.rerun()

with g4.popover("🗑️ Delete"):
    st.write(f"Delete group **{cur}**?")
    if st.button("Yes, delete", key="del_grp", disabled=len(lists) == 1):
        del lists[cur]
        save_lists(lists)
        st.session_state["wl_group"] = None
        st.rerun()
    if len(lists) == 1:
        st.caption("At least one group is needed.")

codes = lists[cur]

# =====================================================================
# 6. SEARCH & ADD
# =====================================================================
s1, s2 = st.columns([5, 1])
pick = s1.selectbox("Search", master["Label"].tolist(), index=None,
                    placeholder="🔍 Search eg: infy, tata steel, hdfc bank, 500325",
                    label_visibility="collapsed")
if s2.button("＋ Add", type="primary", disabled=pick is None, use_container_width=True):
    code = master.loc[master["Label"] == pick, "Code"].iloc[0]
    if code in codes:
        st.toast(f"{short_name.get(code, code)} is already in {cur}")
    else:
        codes.append(code)
        save_lists(lists)
        st.toast(f"Added {short_name.get(code, code)} to {cur}")
        st.rerun()

last = st.session_state.get("selected_symbol")
if last and last not in codes:
    if st.button(f"＋ Add last viewed stock ({last})"):
        codes.append(last)
        save_lists(lists)
        st.rerun()

if master.empty:
    st.caption("ℹ️ Stock list unavailable right now – use the Symbol Finder page to add stocks.")

# =====================================================================
# 7. WATCHLIST TABLE
# =====================================================================
if not codes:
    st.info(f"**{cur}** is empty – search above and click **＋ Add**.")
    st.stop()

q = quotes(tuple(codes))
rows = []
for c in codes:
    ltp, chg, pct, src = q.get(c, NA)
    exch = "BSE" if c.endswith(".BO") else ("INDEX" if c.startswith("^") else "NSE")
    if src and src != c:
        exch += " (via NSE)" if src.endswith(".NS") else " (via name)"
    rows.append({
        "Code": c,
        "Symbol": short_name.get(c, c.replace(".NS", "").replace(".BO", "")),
        "Exch": exch,
        "Chg": float("nan") if chg is None else chg,
        "Chg %": float("nan") if pct is None else pct,
        "": "" if chg is None else ("▲" if chg >= 0 else "▼"),
        "LTP": float("nan") if ltp is None else ltp,
    })
view = pd.DataFrame(rows)

h1, h2 = st.columns([4, 2])
h1.markdown(f"**{cur}** ({len(codes)})")
sort = h2.selectbox("Sort", ["My order", "% change ↓", "% change ↑", "Name A–Z"],
                    label_visibility="collapsed")
if sort == "% change ↓":
    view = view.sort_values("Chg %", ascending=False, na_position="last")
elif sort == "% change ↑":
    view = view.sort_values("Chg %", ascending=True, na_position="last")
elif sort == "Name A–Z":
    view = view.sort_values("Symbol")
view = view.reset_index(drop=True)


def color_row(r):
    if pd.isna(r["Chg"]):
        return [f"color:{GREY}"] * len(r)
    c = GREEN if r["Chg"] >= 0 else RED
    return [f"color:{c}; font-weight:600" if k in ("Symbol", "", "LTP")
            else f"color:{c}" if k == "Chg %"
            else f"color:{GREY}" for k in r.index]


shown = view.drop(columns="Code")
styled = (shown.style.apply(color_row, axis=1)
          .format({"Chg": "{:+,.2f}", "Chg %": "{:+.2f}%", "LTP": "{:,.2f}"}, na_rep="—"))

event = st.dataframe(styled, hide_index=True, use_container_width=True,
                     height=min(38 + 35 * len(view), 700),
                     on_select="rerun", selection_mode="multi-row")
st.caption("Tick one or more rows to act on them. Prices refresh every 60 seconds "
           "(Yahoo data may be delayed). '—' = no Yahoo data for this stock.")

# =====================================================================
# 8. ROW ACTIONS
# =====================================================================
sel = [view.loc[i, "Code"] for i in event.selection.rows]
if sel:
    first = sel[0]
    st.markdown(f"**Selected:** {', '.join(short_name.get(c, c) for c in sel)}")
    a1, a2, a3, a4, a5 = st.columns(5)
    if a1.button("📈 Analyzer"):
        st.session_state["selected_symbol"] = first
        st.switch_page("app.py")
    if a2.button("🏦 Fundamentals"):
        st.session_state["selected_symbol"] = first
        st.switch_page("pages/3_Fundamentals.py")
    in_order = sort == "My order" and len(sel) == 1
    if a3.button("⬆ Move up", disabled=not in_order):
        i = codes.index(first)
        if i > 0:
            codes[i - 1], codes[i] = codes[i], codes[i - 1]
            save_lists(lists)
            st.rerun()
    if a4.button("⬇ Move down", disabled=not in_order):
        i = codes.index(first)
        if i < len(codes) - 1:
            codes[i + 1], codes[i] = codes[i], codes[i + 1]
            save_lists(lists)
            st.rerun()
    if a5.button(f"🗑️ Remove ({len(sel)})"):
        lists[cur] = [c for c in codes if c not in sel]
        save_lists(lists)
        st.rerun()

    others = [n for n in lists if n != cur]
    if others:
        c1, c2 = st.columns([3, 1])
        target = c1.selectbox("Copy selected to group", others)
        if c2.button("📋 Copy"):
            added = [c for c in sel if c not in lists[target]]
            lists[target].extend(added)
            save_lists(lists)
            st.success(f"Copied {len(added)} stock(s) to {target}.")

st.divider()
if st.button(f"📊 Scan this group ({cur}) in Scanner"):
    st.session_state["scan_list"] = cur
    st.switch_page("pages/1_Scanner.py")

# =====================================================================
# 9. BACKUP / RESTORE
# =====================================================================
with st.expander("💾 Backup / restore watchlists"):
    b1, b2 = st.columns(2)
    b1.download_button("⬇ Download backup (JSON)", json.dumps(lists, indent=2),
                       "watchlists.json", "application/json")
    up = b2.file_uploader("⬆ Restore from backup", type="json")
    if up is not None and st.button("Restore"):
        st.session_state["watchlists"] = json.load(up)
        save_lists(st.session_state["watchlists"])
        st.success("Restored.")
        st.rerun()