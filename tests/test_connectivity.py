"""Smoke test: confirm yfinance (Yahoo) and SEC EDGAR are reachable for PYPL.

Run:  python tests/test_connectivity.py
Not a unit test of the model -- just a network/data-source reachability check.
"""
import sys

SEC_HEADERS = {"User-Agent": "AI Equity Analyst research@example.com"}
PYPL_CIK = "0001633917"


def check_yfinance():
    import yfinance as yf
    t = yf.Ticker("PYPL")
    hist = t.history(period="5d")
    if hist.empty:
        return False, "no price history returned"
    last_close = float(hist["Close"].iloc[-1])
    return True, f"last_close={last_close:.2f}  rows={len(hist)}"


def check_edgar():
    import requests
    url = f"https://data.sec.gov/submissions/CIK{PYPL_CIK}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=20)
    r.raise_for_status()
    j = r.json()
    recent = j["filings"]["recent"]
    forms, dates = recent["form"], recent["filingDate"]
    last10k = next(((forms[i], dates[i]) for i in range(len(forms)) if forms[i] == "10-K"), None)
    last10q = next(((forms[i], dates[i]) for i in range(len(forms)) if forms[i] == "10-Q"), None)
    return True, f"name={j.get('name')} cik={j.get('cik')} tickers={j.get('tickers')} 10-K={last10k} 10-Q={last10q}"


def main():
    ok = True
    for name, fn in (("yfinance/Yahoo", check_yfinance), ("SEC EDGAR", check_edgar)):
        try:
            passed, msg = fn()
            print(f"[{'OK ' if passed else 'FAIL'}] {name}: {msg}")
            ok = ok and passed
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
