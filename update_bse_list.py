import os
import requests
import pandas as pd

url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
params = {"Group": "", "Scripcode": "", "industry": "",
          "segment": "Equity", "status": "Active"}
headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*",
           "Referer": "https://www.bseindia.com/",
           "Origin": "https://www.bseindia.com"}

r = requests.get(url, params=params, headers=headers, timeout=30)
r.raise_for_status()
df = pd.DataFrame(r.json())
os.makedirs("data", exist_ok=True)
df.to_csv("data/bse_list.csv", index=False)
print(f"Saved {len(df)} BSE stocks to data/bse_list.csv")