import io
import os
import requests
import pandas as pd

FOLDER = "data/indices"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

# Index name : NSE file name
INDEX_FILES = {
    # Broad market
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty Next 50": "ind_niftynext50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "Nifty Midcap 50": "ind_niftymidcap50list.csv",
    "Nifty Midcap 100": "ind_niftymidcap100list.csv",
    "Nifty Midcap 150": "ind_niftymidcap150list.csv",
    "Nifty Smallcap 100": "ind_niftysmallcap100list.csv",
    "Nifty Smallcap 250": "ind_niftysmallcap250list.csv",
    # Sectoral
    "Nifty Bank": "ind_niftybanklist.csv",
    "Nifty Private Bank": "ind_nifty_privatebanklist.csv",
    "Nifty PSU Bank": "ind_niftypsubanklist.csv",
    "Nifty Financial Services": "ind_niftyfinancelist.csv",
    "Nifty IT": "ind_niftyitlist.csv",
    "Nifty Auto": "ind_niftyautolist.csv",
    "Nifty Pharma": "ind_niftypharmalist.csv",
    "Nifty Healthcare": "ind_niftyhealthcarelist.csv",
    "Nifty FMCG": "ind_niftyfmcglist.csv",
    "Nifty Metal": "ind_niftymetallist.csv",
    "Nifty Energy": "ind_niftyenergylist.csv",
    "Nifty Oil & Gas": "ind_niftyoilgaslist.csv",
    "Nifty Realty": "ind_niftyrealtylist.csv",
    "Nifty Media": "ind_niftymedialist.csv",
    "Nifty Consumer Durables": "ind_niftyconsumerdurableslist.csv",
}


def download_index(name):
    """Download the constituent list from NSE (two sources tried)."""
    fname = INDEX_FILES[name]
    for base in ("https://archives.nseindia.com/content/indices/",
                 "https://niftyindices.com/IndexConstituent/"):
        try:
            r = requests.get(base + fname, headers=HEADERS, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip()
            if "Symbol" in df.columns:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def get_index(name):
    """Use the saved copy first (fast, works on cloud); download if missing."""
    path = os.path.join(FOLDER, INDEX_FILES[name])
    if os.path.exists(path):
        return pd.read_csv(path)
    return download_index(name)


# Run "python indices.py" to save/refresh all lists into data/indices/
if __name__ == "__main__":
    os.makedirs(FOLDER, exist_ok=True)
    for name, fname in INDEX_FILES.items():
        df = download_index(name)
        if df.empty:
            print(f"FAILED  {name}")
        else:
            df.to_csv(os.path.join(FOLDER, fname), index=False)
            print(f"Saved   {name}: {len(df)} stocks")