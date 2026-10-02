"""
Audit the price dataset for data problems. Read-only: writes a per-file report.

Checks every <MARKET>/<interval>/<SYMBOL>.csv.gz for:
    dup_ts      repeated timestamps            unsorted    timestamps out of order
    empty_rows  rows with no OHLC at all       nonpos      an Open/High/Low/Close <= 0
    neg_adj     Adj Close <= 0                 bad_bar     High below Open/Close/Low, or Low above them
    neg_volume  Volume < 0                     spikes      one bar 5x off its neighbours that reverts at once
    max_gap_days  longest hole in daily data (calendar days)
and lists stocks traded recently (daily data in the last week) with no intraday file.

Usage:
    python3 audit.py                  # writes OUTPUT_DIR/audit.csv and prints a summary
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dataset_builder import OUTPUT_DIR

MARKETS = ["US", "India", "Macro"]
INTERVALS = ["1d", "1h", "5m", "2m", "1m"]
SPIKE = 5.0  # a bar this many times off both neighbours, in the same direction


def single_bar_spikes(close):
    """Bars whose close is SPIKE x above (or below) both the previous and next close."""
    c = close.to_numpy(dtype=float)
    if len(c) < 3:
        return np.zeros(len(c), dtype=bool)
    prev, cur, nxt = c[:-2], c[1:-1], c[2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (cur / prev > SPIKE) & (cur / nxt > SPIKE)
        down = (prev / cur > SPIKE) & (nxt / cur > SPIKE)
    flags = np.zeros(len(c), dtype=bool)
    flags[1:-1] = np.nan_to_num(up | down)
    return flags


def audit_file(args):
    market, interval, path = args
    symbol = os.path.basename(path)[: -len(".csv.gz")]
    row = {"market": market, "interval": interval, "symbol": symbol}
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        return {**row, "error": str(e)[:200]}
    ts = pd.to_datetime(df.index, utc=True)
    ohlc = df[["Open", "High", "Low", "Close"]]
    o, h, lo, c = (ohlc[k] for k in ohlc)
    tol = 1e-6 * c.abs().clip(lower=1)
    row.update(
        rows=len(df),
        first=str(ts.min())[:10] if len(df) else "",
        last=str(ts.max())[:10] if len(df) else "",
        dup_ts=int(ts.duplicated().sum()),
        unsorted=int(not ts.is_monotonic_increasing),
        empty_rows=int(ohlc.isna().all(axis=1).sum()),
        nonpos=int((ohlc <= 0).any(axis=1).sum()),
        neg_adj=int((df["Adj Close"] <= 0).sum()) if "Adj Close" in df else 0,
        bad_bar=int(((h < pd.concat([o, c, lo], axis=1).max(axis=1) - tol)
                     | (lo > pd.concat([o, c, h], axis=1).min(axis=1) + tol)).sum()),
        neg_volume=int((df["Volume"] < 0).sum()),
        spikes=int(single_bar_spikes(c).sum()),
        max_gap_days=(float(ts.to_series().diff().max().total_seconds() / 86400)
                      if interval == "1d" and len(df) > 1 else 0.0),
    )
    return row


def main():
    jobs = []
    for market in MARKETS:
        for iv in INTERVALS:
            folder = os.path.join(OUTPUT_DIR, market, iv)
            jobs += [(market, iv, os.path.join(folder, f)) for f in os.listdir(folder)
                     if f.endswith(".csv.gz") and not f.startswith("._")]
    print(f"auditing {len(jobs):,} files")
    with ProcessPoolExecutor() as pool:
        report = pd.DataFrame(pool.map(audit_file, jobs, chunksize=200))
    report.to_csv(os.path.join(OUTPUT_DIR, "audit.csv"), index=False)

    checks = ["dup_ts", "unsorted", "empty_rows", "nonpos", "neg_adj", "bad_bar", "neg_volume", "spikes"]
    summary = report.groupby(["market", "interval"])[checks].agg(lambda s: f"{(s > 0).sum()} files / {s.sum():,} rows")
    print(summary.to_string())
    if "error" in report:
        print("unreadable files:", report["error"].notna().sum())
    daily = report[report["interval"] == "1d"]
    print("daily files with a gap over 30 days:", (daily["max_gap_days"] > 30).sum())

    # Recently traded stocks with no intraday file
    recent = daily[daily["last"] >= (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")]
    for iv in INTERVALS[1:]:
        have = set(report.loc[report["interval"] == iv, ["market", "symbol"]].itertuples(index=False, name=None))
        traded = recent[["market", "symbol"]].itertuples(index=False, name=None)
        missing = [(m, s) for m, s in traded if (m, s) not in have]
        print(f"traded this week but no {iv} file: {len(missing)}", missing[:8])


if __name__ == "__main__":
    main()
