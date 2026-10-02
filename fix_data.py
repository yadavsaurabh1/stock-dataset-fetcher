"""
Fix the data problems found by audit.py, in place, logging every change.

Applied to every <MARKET>/<interval>/<SYMBOL>.csv.gz:
    sort            timestamps into order (the first 1m download saved 7-day chunks newest-first)
    dedupe          repeated timestamps (keep the last)
    empty rows      bars with no Open/High/Low/Close at all: Yahoo pads dividend and split
                    dates with blank bars
    zero prices     bars with any Open/High/Low/Close <= 0 (placeholder rows on listing day)
    glitch spikes   a bar whose close is 5x above or below BOTH neighbours, with zero volume:
                    a Yahoo glitch, not a trade (Tata Steel 9 -> 98 -> 9)
    split-bug bars  daily bars whose Close is 20%+ outside the High-Low range while Open is
                    inside it: Yahoo adjusted Close for a split but not Open/High/Low. Both
                    versions can't be right, so the bar is dropped.

Left alone, on purpose: tiny High/Low mismatches under 0.5% (rounding), old commodity
futures whose settlement Close sits outside the day's range, and negative or zero
Adj Close on very old prices (Yahoo's backward dividend adjustment overshoots; the raw
prices are correct). These are documented in the dataset card instead.

Every dropped row is appended to OUTPUT_DIR/fixes.csv (market, interval, symbol, datetime,
reason, Open, High, Low, Close, Volume) so the change can be reviewed or reversed.

Usage:
    python3 fix_data.py --dry-run        # report what would change, touch nothing
    python3 fix_data.py
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from audit import INTERVALS, MARKETS, single_bar_spikes
from dataset_builder import OUTPUT_DIR

OHLC = ["Open", "High", "Low", "Close"]


def fix_frame(df, interval):
    """Returns (fixed frame, dropped rows with a `reason` column, sorted/deduped flags)."""
    ts = pd.to_datetime(df.index, utc=True)
    sorted_ = not ts.is_monotonic_increasing
    if sorted_:
        df = df.iloc[np.argsort(ts, kind="stable")]
    deduped = int(df.index.duplicated().sum())
    if deduped:
        df = df[~df.index.duplicated(keep="last")]

    reasons = pd.Series("", index=df.index)
    empty = df[OHLC].isna().all(axis=1)
    reasons[empty] = "empty row"
    nonpos = (df[OHLC] <= 0).any(axis=1) & ~empty
    reasons[nonpos] = "zero price"
    spike = pd.Series(single_bar_spikes(df["Close"]), index=df.index) & (df["Volume"] == 0) & (reasons == "")
    reasons[spike] = "glitch spike"
    if interval == "1d":
        o, h, lo, c = (df[k] for k in OHLC)
        inside = (o <= h * 1.005) & (o >= lo * 0.995)
        split_bug = ((c < lo * 0.8) | (c > h * 1.2)) & inside & (reasons == "")
        reasons[split_bug] = "close outside range (split bug)"

    dropped = df[reasons != ""].assign(reason=reasons[reasons != ""])
    return df[reasons == ""], dropped, sorted_, deduped


def fix_file(args):
    market, interval, path, dry_run = args
    symbol = os.path.basename(path)[: -len(".csv.gz")]
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        return {"market": market, "interval": interval, "symbol": symbol, "error": str(e)[:200]}, None
    fixed, dropped, sorted_, deduped = fix_frame(df, interval)
    changed = sorted_ or deduped or len(dropped)
    if changed and not dry_run:
        fixed.to_csv(path + ".tmp", compression="gzip")
        os.replace(path + ".tmp", path)
    log = None
    if len(dropped):
        log = dropped[OHLC + ["Volume", "reason"]].reset_index(names="datetime")
        log.insert(0, "symbol", symbol)
        log.insert(0, "interval", interval)
        log.insert(0, "market", market)
    return {"market": market, "interval": interval, "symbol": symbol, "sorted": int(sorted_),
            "deduped": deduped, "dropped": len(dropped)}, log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--markets", default=",".join(MARKETS))
    args = parser.parse_args()

    jobs = []
    for market in args.markets.split(","):
        for iv in INTERVALS:
            folder = os.path.join(OUTPUT_DIR, market, iv)
            jobs += [(market, iv, os.path.join(folder, f), args.dry_run) for f in os.listdir(folder)
                     if f.endswith(".csv.gz") and not f.startswith("._")]
    print(f"{'checking' if args.dry_run else 'fixing'} {len(jobs):,} files")
    results, logs = [], []
    with ProcessPoolExecutor() as pool:
        for result, log in pool.map(fix_file, jobs, chunksize=200):
            results.append(result)
            if log is not None:
                logs.append(log)
    report = pd.DataFrame(results)
    if "error" in report and report["error"].notna().any():
        print("unreadable:", report[report["error"].notna()][["market", "interval", "symbol", "error"]].to_string())
    summary = report.groupby(["market", "interval"])[["sorted", "deduped", "dropped"]].sum()
    print(summary.to_string())
    fixes = pd.concat(logs) if logs else pd.DataFrame()
    print("\nrows dropped by reason:")
    print(fixes.groupby("reason").size().to_string() if len(fixes) else "none")
    if not args.dry_run:
        fixes.to_csv(os.path.join(OUTPUT_DIR, "fixes.csv"), index=False)
        print(f"\nlog of dropped rows: {os.path.join(OUTPUT_DIR, 'fixes.csv')}")


if __name__ == "__main__":
    main()
