import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # the stock-analyzer folder
DATA = ROOT / "data"
FILE = DATA / "watchlists.json"
BACKUPS = DATA / "backups"
DEFAULT = {"Default": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]}


def _read(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("empty or invalid")
    return data


def load_lists():
    """Main file first; if missing or damaged, use the newest backup."""
    try:
        return _read(FILE)
    except Exception:
        pass
    if BACKUPS.exists():
        for b in sorted(BACKUPS.glob("watchlists_*.json"), reverse=True):
            try:
                return _read(b)
            except Exception:
                continue
    return {k: list(v) for k, v in DEFAULT.items()}


def save_lists(lists):
    """Save, then keep a timestamped backup (last 30)."""
    DATA.mkdir(exist_ok=True)
    BACKUPS.mkdir(exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lists, f, indent=2)
    os.replace(tmp, FILE)                        # write safely in one step
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(BACKUPS / f"watchlists_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(lists, f, indent=2)
    for old in sorted(BACKUPS.glob("watchlists_*.json"), reverse=True)[30:]:
        old.unlink()