"""
Upload the dataset to Hugging Face as plain CSV, one CSV per stock and one dataset repo
per market: <user>/us-stocks-and-search-trends, <user>/india-stocks-and-search-trends and
<user>/macro-markets.

Repo layout:
    prices/<interval>/<SYMBOL>.csv       1d, 1h, 5m, 2m, 1m
    prices_1/<interval>/<SYMBOL>.csv     US only: symbols starting L-Z (Hugging Face
                                         allows at most 10k files per folder)
    trends/<SYMBOL>.csv                  weekly Google search interest, US and India only
    symbols.csv, search_topics.csv, README.md (dataset card)

Every file keeps a `symbol` column, so a whole folder can be loaded as one table.
Prices are written with 7 significant digits: Yahoo serves float32 prices, so
772.7899780273438 is really 772.79.

One folder at a time is converted into a local staging folder, uploaded and deleted, so
the ~17 GB never has to fit on disk at once. Files already on Hugging Face with the same
content hash are skipped, so an interrupted upload resumes where it stopped.

Usage:
    python3 hf_upload.py --dry-run --markets Macro   # convert locally, upload nothing
    python3 hf_upload.py --public                    # upload all three
    python3 hf_upload.py --public --markets India
    python3 hf_upload.py --card-only                 # only refresh the dataset cards
"""
import argparse
import datetime as dt
import hashlib
import os
import shutil
import tempfile
import threading
import time

import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # they flood the log in background runs
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi  # noqa: E402

from dataset_builder import OUTPUT_DIR

REPOS = {"US": "us-stocks-and-search-trends", "India": "india-stocks-and-search-trends", "Macro": "macro-markets"}
PRICE_INTERVALS = ["1d", "1h", "5m", "2m", "1m"]
TRENDS_INTERVAL = "1wk"  # the only trends resolution uploaded, so trends/ has no interval folders
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"]
TREND_COLUMNS = ["value", "raw", "window"]
# Column types for the dataset card; without them the viewer guesses from the first files
PRICE_FEATURES = {"symbol": "string", "datetime": "string", **{c: "float64" for c in PRICE_COLUMNS}}
TREND_FEATURES = {"symbol": "string", "datetime": "string", "value": "float64", "raw": "float64", "window": "string"}
SPLIT_MARKETS = {"US"}  # too many symbols for one folder: L-Z go to prices_1/
SPLIT_CONFIGS = {"1d", "1h", "1m"}  # too big for the HF viewer in one piece: L-Z half gets its own config
CSV_RULE = "*.csv filter=lfs diff=lfs merge=lfs -text"
STALL_MINUTES = 30  # an upload with no new commit for this long is treated as hung
_upload = {"repo": None, "since": 0.0}


def start_watchdog(api):
    """
    A dropped connection can leave an upload waiting forever. If no commit lands for
    STALL_MINUTES, exit with code 3 so a wrapper loop restarts the run (it resumes).
    """
    def watch():
        while True:
            time.sleep(300)
            repo, since = _upload["repo"], _upload["since"]
            if repo is None:
                continue
            try:
                last = api.list_repo_commits(repo, repo_type="dataset")[0].created_at.timestamp()
            except Exception:
                last = 0.0  # can't reach Hugging Face either; count from the upload start
            if time.time() - max(last, since) > STALL_MINUTES * 60:
                print(f"No commit for {STALL_MINUTES} min, the upload looks hung. Exiting to restart.", flush=True)
                os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def prices_folder(market, symbol):
    return "prices_1" if market in SPLIT_MARKETS and symbol[0].upper() >= "L" else "prices"


def folders(market):
    """(repo folder, local folder, columns, symbol filter) for every folder of one market."""
    market_dir = os.path.join(OUTPUT_DIR, market)
    tops = ["prices", "prices_1"] if market in SPLIT_MARKETS else ["prices"]
    for top in tops:
        for iv in PRICE_INTERVALS:
            yield (f"{top}/{iv}", os.path.join(market_dir, iv), PRICE_COLUMNS,
                   lambda s, top=top: prices_folder(market, s) == top)
    yield "trends", os.path.join(market_dir, "trends", TRENDS_INTERVAL), TREND_COLUMNS, lambda s: True


def configs_for(market):
    """Dataset-card configs [(name, [patterns])] and an example file path, from the local folders."""
    configs, example = {}, None
    for repo_dir, local_dir, _, wanted in folders(market):
        if not os.path.isdir(local_dir):
            continue
        symbols = [f[: -len(".csv.gz")] for f in sorted(os.listdir(local_dir))
                   if f.endswith(".csv.gz") and not f.startswith("._")]
        symbols = [s for s in symbols if wanted(s)]
        if symbols:
            name = repo_dir.replace("prices_1/", "").replace("prices/", "")
            if repo_dir.startswith("prices_1/") and name in SPLIT_CONFIGS:
                name += "_L-Z"
            configs.setdefault(name, []).append(f"{repo_dir}/*.csv")
            example = example or f"{repo_dir}/{symbols[0]}.csv"
    order = [*PRICE_INTERVALS, "trends"]  # each _L-Z half right after its A-K half
    return sorted(configs.items(), key=lambda c: (order.index(c[0].split("_")[0]), c[0])), example


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(src, dst, symbol, columns):
    df = pd.read_csv(src, index_col=0)
    df = df.reindex(columns=columns).rename_axis("datetime").reset_index()
    df.insert(0, "symbol", symbol)
    df.to_csv(dst, index=False, float_format="%.7g")


def dataset_card(market, repo_id, configs, n_symbols, has_trends, example):
    config_yaml = "\n".join(
        f"- config_name: {name}\n  data_files: {patterns}" + ("\n  default: true" if i == 0 else "")
        for i, (name, patterns) in enumerate(configs)
    )
    info_yaml = "\n".join(
        f"- config_name: {name}\n  features:\n" + "\n".join(
            f"  - name: {col}\n    dtype: {dtype}"
            for col, dtype in (TREND_FEATURES if name == "trends" else PRICE_FEATURES).items())
        for name, _ in configs
    )
    split = f"""
Hugging Face allows at most 10,000 files per folder, so symbols starting with **A–K**
are in `prices/` and those starting with **L–Z** are in `prices_1/`, with the same
interval folders in both.

The {", ".join(f"`{c}`" for c in sorted(SPLIT_CONFIGS))} subsets are too large for the
dataset viewer in one piece, so each holds the A–K symbols and a matching `_L-Z` subset
(e.g. `1d_L-Z`) holds the rest. Load both for every symbol:

```python
from datasets import concatenate_datasets, load_dataset
daily = concatenate_datasets([load_dataset("{repo_id}", c, split="train") for c in ["1d", "1d_L-Z"]])
```
""" if market in SPLIT_MARKETS else ""
    trends = """
## Search trends (`trends/<SYMBOL>.csv`)

| Interval | History | Stocks |
|---|---|---|
| weekly (weeks start on Sunday) | 2004 to now | the 100 most traded (by dollar volume) |

Google Trends interest in each company's Google topic, searched from the stock's own
country. Google only returns weekly data for windows of up to 5 years, each scaled 0-100
on its own, so the windows are rescaled onto Google's monthly series for 2004-now.
Columns: `value` is interest on one scale for the whole file (all-time monthly peak =
100; weekly spikes can exceed it), `raw` is Google's 0-100 number within its request
window, `window` is the start of that window. `search_topics.csv` lists which Google
topic each stock was matched to.
""" if has_trends else ""
    title = f"{market} stocks and search trends" if has_trends else f"{market} market data"
    return f"""---
license: other
pretty_name: {title}
tags:
- finance
- stocks
- time-series
configs:
{config_yaml}
dataset_info:
{info_yaml}
---

# {title}

{n_symbols} symbols, one plain CSV per symbol and resolution.

```python
import pandas as pd
df = pd.read_csv("hf://datasets/{repo_id}/{example}")

from datasets import load_dataset          # every symbol of one resolution as one table
ds = load_dataset("{repo_id}", "{configs[0][0]}")
```

## Prices (`prices/<interval>/<SYMBOL>.csv`)

| Interval | Bars | History |
|---|---|---|
| `1d` | daily | full history (back to 1962 for the oldest) |
| `1h` | hourly | ~2 years |
| `5m`, `2m` | 5- and 2-minute | ~60 days |
| `1m` | 1-minute | ~30 days |
{split}
Columns: `symbol`, `datetime` (exchange-local time with UTC offset), `Open`, `High`,
`Low`, `Close`, `Adj Close`, `Volume`, `Dividends`, `Stock Splits`, `Capital Gains`
(funds only). Prices are raw (not split-adjusted); use `Adj Close` or the
`Dividends`/`Stock Splits` columns to adjust. Values have 7 significant digits, the
precision Yahoo serves them at. `symbols.csv` lists every symbol with its name.
{trends}
## Data cleaning and known issues

Removed from Yahoo's raw data (logged in `fixes.csv`): blank bars Yahoo pads onto dividend
and split dates, zero-price placeholder bars, zero-volume bars 5x off both neighbours
(feed glitches), and daily bars whose Close was split-adjusted while Open/High/Low were not.

Still present, so filter if they matter to you:
- **`Adj Close` can be zero or negative** on very old bars of a few long-lived stocks:
  Yahoo adjusts backwards for dividends and spin-offs, which overshoots. Raw prices are correct.
- **High/Low can miss Open/Close by under 0.5%** on some bars (rounding), and old commodity
  futures' settlement Close can sit outside the day's range.
- **One-bar price spikes with real volume** in thinly traded stocks are left in. The
  previous and next bars are both 5x away. Yahoo also records some reverse splits one bar late.
- **Intraday files are missing** for some recently listed or thinly traded symbols: Yahoo had no bars.

## Source and terms

Prices come from Yahoo Finance via the `yfinance` library; search interest from Google
Trends via `pytrends`. Yahoo Finance data is provided for personal use under Yahoo's terms,
so check those terms before redistributing this data.
"""


def upload_market(api, market, repo_id, public, dry_run, staging, card_only=False):
    universe = pd.read_csv(os.path.join(OUTPUT_DIR, "universe.csv"))
    symbols = universe[universe["market"] == market]
    if not card_only:
        upload_data(api, market, repo_id, public, dry_run, staging)

    configs, example = configs_for(market)
    extras_dir = os.path.join(staging, market + "_extras")
    os.makedirs(extras_dir, exist_ok=True)
    extras = {name: os.path.join(extras_dir, name) for name in ["symbols.csv", "README.md"]}
    fixes = os.path.join(OUTPUT_DIR, "fixes.csv")
    if os.path.exists(fixes):
        rows = pd.read_csv(fixes)
        rows = rows[rows["market"] == market]
        if len(rows):
            extras["fixes.csv"] = os.path.join(extras_dir, "fixes.csv")
            rows.to_csv(extras["fixes.csv"], index=False)
    symbols.drop(columns="market").to_csv(extras["symbols.csv"], index=False)
    topics = os.path.join(OUTPUT_DIR, market, "trends", "topics.csv")
    if os.path.exists(topics):
        extras["search_topics.csv"] = topics
    has_trends = any(name == "trends" for name, _ in configs)
    with open(extras["README.md"], "w") as f:
        f.write(dataset_card(market, repo_id, configs, len(symbols), has_trends, example))
    if not dry_run:
        api.create_commit(repo_id, repo_type="dataset", commit_message="Update dataset card and symbol lists",
                          operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=local)
                                      for p, local in extras.items()])
    print(f"{market}: {'built locally in ' + staging if dry_run else 'https://huggingface.co/datasets/' + repo_id}")


def upload_data(api, market, repo_id, public, dry_run, staging):
    """Convert and upload every folder of one market, skipping files already uploaded."""
    remote = {}
    if not dry_run:
        api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
        remote = {f.path: (f.lfs.sha256 if getattr(f, "lfs", None) else f"size:{f.size}")
                  for f in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
                  if getattr(f, "size", None) is not None}
        # Small files otherwise travel inside the commit request, which times out on slow links
        attributes = open(api.hf_hub_download(repo_id, ".gitattributes", repo_type="dataset")).read()
        if CSV_RULE not in attributes.splitlines():
            api.upload_file(repo_id=repo_id, repo_type="dataset", path_in_repo=".gitattributes",
                            path_or_fileobj=(attributes.rstrip("\n") + "\n" + CSV_RULE + "\n").encode(),
                            commit_message="Store CSV files as large files")

    chunk = os.path.join(staging, market)
    for repo_dir, local_dir, columns, wanted in folders(market):
        if not os.path.isdir(local_dir):
            continue
        if not dry_run:
            shutil.rmtree(chunk, ignore_errors=True)
        os.makedirs(os.path.join(chunk, repo_dir), exist_ok=True)
        names = sorted(f for f in os.listdir(local_dir) if f.endswith(".csv.gz") and not f.startswith("._"))
        total = new = new_bytes = 0
        for name in names:
            symbol = name[: -len(".csv.gz")]
            if not wanted(symbol):
                continue
            repo_path = f"{repo_dir}/{symbol}.csv"
            dst = os.path.join(chunk, repo_path)
            try:
                convert(os.path.join(local_dir, name), dst, symbol, columns)
            except Exception as e:
                print(f"    skipped {symbol}: {e}")
                continue
            total += 1
            size = os.path.getsize(dst)
            if remote.get(repo_path) == sha256_of(dst):
                os.remove(dst)
            else:
                new += 1
                new_bytes += size
        if repo_dir == "trends" and not dry_run:
            # Stocks dropped from trends (e.g. no Google data) are removed from the repo too
            local = {f"trends/{n[: -len('.csv.gz')]}.csv" for n in names}
            gone = [p for p in remote if p.startswith("trends/") and p.count("/") == 1 and p not in local]
            if gone:
                api.create_commit(repo_id, repo_type="dataset", commit_message="Remove dropped trends",
                                  operations=[CommitOperationDelete(path_in_repo=p) for p in gone])
                print(f"  trends: removed {len(gone)} dropped files: {', '.join(gone)}")
        if not total:
            continue
        print(f"  {repo_dir}: {total} symbols, {new} new or changed ({new_bytes / 1e6:.0f} MB)", flush=True)
        if new and not dry_run:
            _upload.update(repo=repo_id, since=time.time())
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=chunk,
                              commit_message=f"Upload {repo_dir}")
            _upload["repo"] = None
            print(f"    uploaded at {dt.datetime.now():%H:%M}", flush=True)
    if not dry_run:
        shutil.rmtree(chunk, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markets", default="Macro,India,US")
    parser.add_argument("--user", help="Hugging Face user or org (default: the logged-in user)")
    parser.add_argument("--public", action="store_true", help="make new repos public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="convert locally, upload nothing")
    parser.add_argument("--card-only", action="store_true", help="only update the dataset cards and symbol lists")
    parser.add_argument("--staging", help="folder for converted CSVs (default: a temp folder)")
    args = parser.parse_args()

    api = HfApi()
    if not args.dry_run:
        start_watchdog(api)
    user = args.user or (None if args.dry_run else api.whoami()["name"]) or "you"
    staging = args.staging or os.path.join(tempfile.gettempdir(), "hf_upload_staging")
    if not args.staging:
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
    for market in args.markets.split(","):
        repo_id = f"{user}/{REPOS[market]}"
        print(f"{market} -> {repo_id}", flush=True)
        upload_market(api, market, repo_id, args.public, args.dry_run, staging, args.card_only)
    if not args.dry_run and not args.staging:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
