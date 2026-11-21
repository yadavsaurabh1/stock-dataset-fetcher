# Stock dataset fetcher

Builds a multi-resolution price dataset for every listed US and Indian (NSE) stock,
plus Google search trends for the most traded ones, and publishes it to Hugging Face
as plain CSV.

Published datasets:

- [yadavsaurabh/us-stocks-and-search-trends](https://huggingface.co/datasets/yadavsaurabh/us-stocks-and-search-trends)
- [yadavsaurabh/india-stocks-and-search-trends](https://huggingface.co/datasets/yadavsaurabh/india-stocks-and-search-trends)
- [yadavsaurabh/macro-markets](https://huggingface.co/datasets/yadavsaurabh/macro-markets)

## What it collects

| Data | Source | Coverage |
|---|---|---|
| Daily prices | Yahoo Finance | full history (1962+ for the oldest) |
| Hourly prices | Yahoo Finance | last ~2 years |
| 5-, 2- and 1-minute prices | Yahoo Finance | last 60 / 60 / 30 days |
| Weekly search interest | Google Trends | 2004 to now, top 100 stocks per market |

Universe: ~5,900 US stocks, ~5,700 US ETFs, ~2,600 NSE stocks and ~90 macro series
(indices, VIX, yields, currencies, commodity futures, crypto).

Yahoo only serves a rolling window of intraday data. Re-running the builder merges new
bars into the existing files, so the intraday history keeps growing as long as it runs
at least every ~3 weeks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login          # only needed for hf_upload.py
```

## Scripts

| Script | What it does |
|---|---|
| `dataset_builder.py` | Downloads prices into `~/stock_dataset/<Market>/<interval>/<SYMBOL>.csv.gz`. Resumable. |
| `trends_builder.py` | Adds Google Trends interest for the most traded stocks. Paces itself to Google's rate limit. |
| `audit.py` | Read-only check of every price file (ordering, blank rows, zero prices, glitch spikes, bad bars). |
| `fix_data.py` | Applies the fixes `audit.py` reports, logging every dropped row to `fixes.csv`. |
| `hf_upload.py` | Converts to plain CSV and uploads one dataset repo per market, with a generated dataset card. Only changed files are sent. |
| `run_trends.sh` | Runs `trends_builder.py` after any running price build finishes. |
| `run_hf_upload.sh` | Runs `hf_upload.py`, restarting it if an upload hangs. |

Typical run:

```bash
python3 dataset_builder.py            # a few hours for everything
python3 trends_builder.py             # a day or more; Google rate limits hard
python3 audit.py && python3 fix_data.py
python3 hf_upload.py --public
```

Each script's `--help` lists its options (markets, sample sizes, output folder).

## Notes

- Yahoo Finance data is provided for personal use under Yahoo's terms. Check them before
  redistributing anything built with this.
- Google Trends returns 0-100 relative to the peak in each request window, so weekly data
  is fetched in 5-year windows and rescaled onto the monthly series to get one scale per stock.
- Hugging Face allows at most 10,000 files per folder, so US prices are split into
  `prices/` (symbols A-K) and `prices_1/` (L-Z).

## License

[GPL-3.0](LICENSE)
