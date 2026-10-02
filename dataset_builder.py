"""
Build a multi-resolution stock dataset for every listed US and Indian (NSE) stock.

Layout (gzip-compressed CSV; pd.read_csv opens it directly):
    OUTPUT_DIR/universe.csv                    ticker list used for the run
    OUTPUT_DIR/failed.csv                      tickers Yahoo had no data for
    OUTPUT_DIR/<MARKET>/1d/<SYMBOL>.csv.gz     daily, full history (1962+ for the oldest)
    OUTPUT_DIR/<MARKET>/1h/<SYMBOL>.csv.gz     hourly, last ~730 days
    OUTPUT_DIR/<MARKET>/5m/<SYMBOL>.csv.gz     5-minute, last ~60 days
    OUTPUT_DIR/<MARKET>/2m/<SYMBOL>.csv.gz     2-minute, last ~60 days
    OUTPUT_DIR/<MARKET>/1m/<SYMBOL>.csv.gz     1-minute, last ~30 days
where <MARKET> is US (stocks + ETFs), India (NSE) or Macro.

Universe: every US common stock and ETF (NASDAQ/NYSE/AMEX), every NSE stock, plus
~90 macro series (indices, VIX, yields, FX, commodity futures, crypto).

Intraday files are MERGED with what is already on disk, so re-running the script
regularly (at least every ~3 weeks) grows the intraday history beyond Yahoo's
rolling window. Prices are stored raw (auto_adjust=False); daily files keep an
"Adj Close" column plus Dividends / Stock Splits for adjusting later.

Usage:
    python3 dataset_builder.py                      # everything (~14,000 tickers)
    python3 dataset_builder.py --markets in,macro   # NSE + macro only
    python3 dataset_builder.py --no-etfs            # skip the ~5,700 US ETFs
    python3 dataset_builder.py --sample 50          # random 50, for testing
"""
import argparse
import datetime as dt
import io
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

yf.config.debug.hide_exceptions = False

OUTPUT_DIR = os.path.expanduser("~/stock_dataset")

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NSE_EQUITY = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

NON_COMMON = re.compile(
    r"warrant|\bunits?\b|\brights?\b|preferred|notes due|debenture|depositary shares? representing a",
    re.I,
)

MACRO = [
    # US indices, volatility, treasury yields
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^NYA", "^W5000", "^OEX", "^SOX",
    "^VIX", "^VVIX", "^IRX", "^FVX", "^TNX", "^TYX",
    # Indian indices
    "^NSEI", "^BSESN", "^NSEBANK", "^CNXIT", "^CNXAUTO", "^CNXPHARMA", "^CNXFMCG", "^CNXMETAL",
    "^CNXREALTY", "^CNXENERGY", "^CNXPSUBANK", "^CNXINFRA", "^CNXMEDIA", "^NSMIDCP", "^CNX100",
    "^CNX200", "^CRSLDX", "^INDIAVIX",
    # Global indices
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI", "000001.SS", "^KS11", "^AXJO",
    "^GSPTSE", "^BVSP", "^TWII", "^STI",
    # Currencies
    "INR=X", "EURUSD=X", "GBPUSD=X", "JPY=X", "CNY=X", "AUDUSD=X", "CAD=X", "CHF=X", "DX-Y.NYB",
    # Futures: equity index, energy, metals, grains, rates, softs, livestock
    "ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "GC=F", "SI=F",
    "HG=F", "PL=F", "PA=F", "ZC=F", "ZW=F", "ZS=F", "ZN=F", "ZB=F", "ZF=F", "ZT=F", "KC=F",
    "CC=F", "SB=F", "CT=F", "LE=F", "HE=F",
    # Crypto (trades 24/7, so the densest intraday files)
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
    "LINK-USD", "LTC-USD",
]

REFRESH_HOURS = 12  # files newer than this are not re-fetched

_rate_limit_until = 0.0
_rate_lock = threading.Lock()


def load_us_universe(include_etfs):
    nas = pd.read_csv(NASDAQ_LISTED, sep="|")[:-1]  # last row is a timestamp footer
    oth = pd.read_csv(OTHER_LISTED, sep="|")[:-1]
    nas = nas.rename(columns={"Symbol": "symbol", "Security Name": "name"})
    oth = oth.rename(columns={"ACT Symbol": "symbol", "Security Name": "name"})
    df = pd.concat([nas, oth], ignore_index=True)
    df = df[df["Test Issue"] == "N"]
    if not include_etfs:
        df = df[df["ETF"] == "N"]
    df = df[~df["name"].str.contains(NON_COMMON, na=False)]
    df = df[~df["symbol"].str.contains(r"\$", na=True)]  # preferred series like ABR$D
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    df["market"] = "US"
    return df[["symbol", "name", "market"]]


def load_in_universe():
    resp = requests.get(NSE_EQUITY, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "name"})
    df["symbol"] = df["symbol"] + ".NS"
    df["market"] = "India"
    return df[["symbol", "name", "market"]]


def load_universe(markets, include_etfs):
    parts = []
    if "macro" in markets:
        parts.append(pd.DataFrame({"symbol": MACRO, "name": MACRO, "market": "Macro"}))
    if "us" in markets:
        parts.append(load_us_universe(include_etfs))
    if "in" in markets:
        parts.append(load_in_universe())
    return pd.concat(parts, ignore_index=True).drop_duplicates("symbol")


def history(ticker, **kwargs):
    """Ticker.history with rate-limit backoff. Returns an empty frame if Yahoo has no data."""
    global _rate_limit_until
    for attempt in range(6):
        wait = _rate_limit_until - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            return ticker.history(auto_adjust=False, timeout=30, **kwargs)
        except (YFPricesMissingError, YFTzMissingError):
            return pd.DataFrame()
        except YFRateLimitError:
            with _rate_lock:
                _rate_limit_until = max(_rate_limit_until, time.time() + 60 * (attempt + 1))
            print(f"  rate limited, pausing all threads {60 * (attempt + 1)}s")
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("gave up after repeated rate limiting")


def fetch_1m(ticker):
    """Yahoo serves 1m bars for the last 30 days, but only 7 days per request."""
    end = dt.datetime.now()
    oldest = end - dt.timedelta(days=29)
    parts = []
    while end > oldest:
        start = max(end - dt.timedelta(days=7), oldest)
        parts.append(history(ticker, start=start, end=end, interval="1m"))
        end = start
    parts = [p for p in parts if not p.empty]
    return clean(pd.concat(parts)) if parts else pd.DataFrame()


def clean(df):
    """
    Drop the bars Yahoo pads or garbles: rows with no prices (dividend/split dates),
    rows with a zero price (listing-day placeholders) and zero-volume bars 5x off both
    neighbours (glitches). Also sorts, since chunked downloads arrive newest-first.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    ohlc = df[["Open", "High", "Low", "Close"]]
    keep = ~ohlc.isna().all(axis=1) & ~(ohlc <= 0).any(axis=1)
    c = df["Close"].to_numpy(dtype=float)
    if len(c) >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            prev, cur, nxt = c[:-2], c[1:-1], c[2:]
            spike = ((cur / prev > 5) & (cur / nxt > 5)) | ((prev / cur > 5) & (nxt / cur > 5))
        flags = np.zeros(len(c), dtype=bool)
        flags[1:-1] = np.nan_to_num(spike)
        keep &= ~(flags & (df["Volume"].to_numpy() == 0))
    return df[keep]


def merge_with_existing(path, new):
    """Append new bars to the stored file, rescaling old bars if a split happened in between."""
    if not os.path.exists(path):
        return new
    old = pd.read_csv(path, index_col=0)
    old.index = pd.to_datetime(old.index, utc=True).tz_convert(new.index.tz)
    overlap = old.index.intersection(new.index)
    if len(overlap):
        ratio = (old.loc[overlap, "Close"] / new.loc[overlap, "Close"]).median()
        if abs(ratio - 1) > 0.05:  # prices moved on the overlap -> split/consolidation
            old[["Open", "High", "Low", "Close"]] /= ratio
            old["Volume"] *= ratio
    merged = pd.concat([old, new])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def is_fresh(path):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH_HOURS * 3600


INTRADAY = {
    "1h": lambda t: history(t, period="730d", interval="1h"),
    "5m": lambda t: history(t, period="60d", interval="5m"),
    "2m": lambda t: history(t, period="60d", interval="2m"),
    "1m": fetch_1m,
}
INTERVALS = ["1d", *INTRADAY]


def fetch_symbol(symbol, market, output_dir):
    paths = {iv: os.path.join(output_dir, market, iv, f"{symbol}.csv.gz") for iv in INTERVALS}
    if all(is_fresh(p) for p in paths.values()):
        return symbol, "skipped"

    ticker = yf.Ticker(symbol)

    # Daily first: if Yahoo has no daily history the symbol is unknown, so skip intraday
    if not is_fresh(paths["1d"]):
        daily = clean(history(ticker, period="max", interval="1d"))
        if daily.empty:
            return symbol, "no data"
        daily.to_csv(paths["1d"])

    for iv, fetch in INTRADAY.items():
        if is_fresh(paths[iv]):
            continue
        df = fetch(ticker)
        if df.empty:
            continue
        merge_with_existing(paths[iv], clean(df)).to_csv(paths[iv])

    return symbol, "ok"


def dir_size(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--markets", default="us,in,macro", help="comma list of: us, in, macro")
    parser.add_argument("--no-etfs", action="store_true", help="skip the ~5,700 US ETFs")
    parser.add_argument("--sample", type=int, help="only fetch N random tickers (for testing)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    universe = load_universe(args.markets.split(","), not args.no_etfs)
    for market in universe["market"].unique():
        for iv in INTERVALS:
            os.makedirs(os.path.join(args.output, market, iv), exist_ok=True)
    if args.sample:
        universe = universe.sample(n=min(args.sample, len(universe)))
    universe.to_csv(os.path.join(args.output, "universe.csv"), index=False)
    symbols = list(zip(universe["symbol"], universe["market"]))
    print(f"{len(symbols)} tickers ({universe['market'].value_counts().to_dict()}) -> {args.output}")

    started = time.time()
    counts = {"ok": 0, "skipped": 0, "no data": 0, "error": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_symbol, s, m, args.output): s for s, m in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            symbol = futures[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = "error"
                print(f"  ERROR {symbol}: {e}")
            counts[status] += 1
            if status in ("no data", "error"):
                failed.append((symbol, status))
            if i % 25 == 0 or i == len(symbols):
                elapsed = time.time() - started
                eta = elapsed / i * (len(symbols) - i)
                print(f"[{i}/{len(symbols)}] {counts}  elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")

    pd.DataFrame(failed, columns=["symbol", "status"]).to_csv(os.path.join(args.output, "failed.csv"), index=False)
    print(f"Done in {(time.time() - started) / 60:.1f} min. Dataset size: {dir_size(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
