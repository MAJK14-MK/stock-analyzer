import base64
import json
import os
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FILE = DATA / "watchlists.json"
BACKUPS = DATA / "backups"
GH_PATH = "data/watchlists.json"
DEFAULT = {"Default": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]}


# ---------- GitHub settings (from secrets; None if not set) ----------
def _gh():
    try:
        import streamlit as st
        s = st.secrets
        return s["GITHUB_TOKEN"], s["GITHUB_REPO"], s.get("GITHUB_BRANCH", "watchlists-data")
    except Exception:
        return None


def _headers(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"}


def _url(repo):
    return f"https://api.github.com/repos/{repo}/contents/{GH_PATH}"


def storage_mode():
    return "GitHub" if _gh() else "Local only"


def _warn(msg):
    try:
        import streamlit as st
        st.warning(msg)
    except Exception:
        print(msg)


# ---------- Local helpers ----------
def _read(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("empty or invalid")
    return data


def _save_local(lists):
    DATA.mkdir(exist_ok=True)
    BACKUPS.mkdir(exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(lists, f, indent=2)
    os.replace(tmp, FILE)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(BACKUPS / f"watchlists_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(lists, f, indent=2)
    for old in sorted(BACKUPS.glob("watchlists_*.json"), reverse=True)[30:]:
        old.unlink()


def _load_local():
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


# ---------- Public functions used by the pages ----------
def load_lists():
    """GitHub first (shared master copy); fall back to local file/backups."""
    cfg = _gh()
    if cfg:
        token, repo, branch = cfg
        try:
            r = requests.get(_url(repo), params={"ref": branch},
                             headers=_headers(token), timeout=15)
            if r.status_code == 200:
                data = json.loads(base64.b64decode(r.json()["content"]).decode("utf-8"))
                if isinstance(data, dict) and data:
                    try:
                        _save_local(data)          # keep a local copy too
                    except Exception:
                        pass
                    return data
            elif r.status_code != 404:
                _warn(f"Could not read watchlists from GitHub ({r.status_code}). Using local copy.")
        except Exception as e:
            _warn(f"GitHub not reachable ({e}). Using local copy.")
    return _load_local()


def save_lists(lists):
    """Save locally, then to GitHub (one commit per change)."""
    try:
        _save_local(lists)
    except Exception:
        pass                                       # cloud disk may be read-only
    cfg = _gh()
    if not cfg:
        return
    token, repo, branch = cfg
    try:
        h = _headers(token)
        cur = requests.get(_url(repo), params={"ref": branch}, headers=h, timeout=15)
        sha = cur.json().get("sha") if cur.status_code == 200 else None
        body = {"message": f"Update watchlists {datetime.now():%Y-%m-%d %H:%M}",
                "content": base64.b64encode(json.dumps(lists, indent=2).encode("utf-8")).decode(),
                "branch": branch}
        if sha:
            body["sha"] = sha
        r = requests.put(_url(repo), json=body, headers=h, timeout=20)
        if r.status_code not in (200, 201):
            _warn(f"Saved locally, but GitHub save failed ({r.status_code}): {r.text[:150]}")
    except Exception as e:
        _warn(f"Saved locally, but GitHub save failed: {e}")