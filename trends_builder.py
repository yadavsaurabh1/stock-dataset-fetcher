"""
Google search-interest dataset to go with the prices from dataset_builder.py.

For each stock (ETFs and macro series are skipped) it fetches Google Trends interest
in the company's Google "topic" (the company entity, so "Apple" the company and not
the fruit), searched from the stock's own country (US or India).

Layout, next to the price folders:
    OUTPUT_DIR/<MARKET>/trends/topics.csv            symbol -> Google topic (edit to fix a bad match)
    OUTPUT_DIR/<MARKET>/trends/1mo/<SYMBOL>.csv.gz   monthly, 2004 -> now
    OUTPUT_DIR/<MARKET>/trends/1wk/<SYMBOL>.csv.gz   weekly,  2004 -> now
    OUTPUT_DIR/<MARKET>/trends/1d/<SYMBOL>.csv.gz    daily,   2004 -> now
    OUTPUT_DIR/<MARKET>/trends/1h/<SYMBOL>.csv.gz    hourly,  last ~730 days (grows on re-runs)
All timestamps are UTC.

Google only returns 0-100 relative to the peak inside each request window, and long
ranges come back coarser (daily needs <=269-day windows, hourly 7-day windows). Each
finer window is rescaled onto the next coarser series, so every file has one scale:
`value` is interest relative to the all-time monthly peak (= 100; daily and hourly
spikes can go above it). `raw` and `window` keep Google's original numbers.

Stocks are processed in order of average dollar volume over the last year (from the
price files), one resolution at a time, so an interrupted run still leaves the most
traded stocks with the coarse series. Re-running only fetches what is new.

Usage:
    python3 trends_builder.py                              # top 100 US + top 100 India, monthly + weekly
    python3 trends_builder.py --resolutions 1mo,1wk,1d,1h  # add daily and hourly (slow)
    python3 trends_builder.py --top 0 --resolutions 1mo    # every stock, monthly only
    python3 trends_builder.py --symbols AAPL,TCS.NS        # just these
"""
import argparse
import datetime as dt
import os
import random
import re
import time
import warnings

import pandas as pd
import requests
from pytrends.exceptions import ResponseError, TooManyRequestsError
from pytrends.request import TrendReq

from dataset_builder import OUTPUT_DIR, load_universe

warnings.simplefilter("ignore", FutureWarning)  # pytrends uses deprecated pandas calls

GEO = {"US": "US", "India": "IN"}
START = dt.datetime(2004, 1, 1)  # Google Trends starts here

# Window length per resolution; Google switches to a coarser step above these
WINDOW = {"1wk": dt.timedelta(days=1820), "1d": dt.timedelta(days=269), "1h": dt.timedelta(days=7)}
COARSER = {"1wk": "1mo", "1d": "1wk", "1h": "1d"}
# Skip a file refreshed more recently than this
REFRESH = {"1mo": dt.timedelta(days=7), "1wk": dt.timedelta(days=7), "1d": dt.timedelta(days=1),
           "1h": dt.timedelta(hours=12)}
# Windows ending later than this before now may still hold partial data, so they are
# always re-fetched as the "tail" instead of stored as finished grid windows
SETTLE = dt.timedelta(days=3)

COMPANY_TYPE = re.compile(
    r"company|corporation|business|bank|manufacturer|conglomerate|firm|retailer|airline|insur|maker|"
    r"broadcaster|undertaking|exchange|industry|carrier|"
    r"producer|provider|operator|developer|publisher|chain|group|holding|trust|utility|pharma|biotech|"
    r"semiconductor|restaurant|hotel|telecom|energy|mining|oil|software|enterprise|organi[sz]ation|brand",
    re.I,
)
PLACE_TYPE = re.compile(r"\bin [A-Z]")  # "Mining company in Delhi, India": a site, not the company
NOT_COMPANY_TYPE = re.compile(r"book|author|novel|band|song|album|film|movie|episode|station|earnings|^topic$", re.I)
LEGAL_SUFFIX = re.compile(
    r"[,.]?\s+(inc|incorporated|ltd|limited|llc|l\.l\.c|plc|corp|corporation|co|company|holdings?|group|"
    r"n\.?v|s\.?a|ag|se)\b\.?", re.I)
NAME_NOISE = re.compile(
    r"\s+(-\s+)?(class [a-z]\b.*|common stock.*|ordinary shares.*|common shares.*|american depositary.*|"
    r"depositary shares.*|ads\b.*|units?\b.*|shares of beneficial.*)$",
    re.I,
)


class Trends:
    """
    One Google Trends session, paced to Google's limit. Google blocks an IP for ~7 min
    after ~40 quick requests, and for an hour or more after repeated blocks, so each
    block slows the pace a little and a long run of successes speeds it up again.
    """

    def __init__(self, delay):
        self.min_delay = self.delay = delay
        self.session = None
        self.streak = 0

    def _call(self, fn):
        for attempt in range(24):  # up to ~10 h of blocks before giving up
            time.sleep(self.delay * random.uniform(0.8, 1.2))
            try:
                if self.session is None:
                    self.session = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
                result = fn()
            except TooManyRequestsError:
                self.session, self.streak = None, 0
                self.delay = min(self.delay * 1.25, 60)
                pause = min(120 * 2**attempt, 1800)
                print(f"  Google rate limit: pausing {pause // 60} min, then 1 request per {self.delay:.0f}s")
                time.sleep(pause)
                continue
            except requests.RequestException as e:
                self.session = None
                print(f"  network error ({type(e).__name__}), retrying in 1 min")
                time.sleep(60)
                continue
            self.streak += 1
            if self.streak % 50 == 0:
                self.delay = max(self.delay * 0.9, self.min_delay)
            return result
        raise SystemExit("Google keeps rate limiting this IP. Progress is saved; re-run later.")

    def suggestions(self, text):
        try:
            return self._call(lambda: self.session.suggestions(text))
        except ResponseError:
            return []

    def interest(self, query, geo, start, end, hourly=False):
        fmt = "%Y-%m-%dT%H" if hourly else "%Y-%m-%d"
        timeframe = f"{start.strftime(fmt)} {end.strftime(fmt)}"

        def fetch():
            self.session.build_payload([query], timeframe=timeframe, geo=geo)
            return self.session.interest_over_time()

        df = self._call(fetch)
        if df.empty:
            return pd.Series(dtype=float)
        df = df[~df["isPartial"].astype(bool)]
        return df[query].astype(float)


def clean_name(name):
    name = name.split(" - ")[0]
    name = NAME_NOISE.sub("", name)
    return name.strip(" ,.")


def short_name(name):
    """'Vertiv Holdings, LLC' -> 'Vertiv'."""
    return LEGAL_SUFFIX.sub("", name).strip(" ,.") or name


def name_words(text):
    return set(re.findall(r"[a-z0-9]+", short_name(text).lower())) - {"and", "of", "the"}


def same_name(name, title):
    """'Tesla, Inc' ~ 'Tesla'; 'International Business Machines' ~ 'IBM' (initials)."""
    words, title_words = name_words(name), name_words(title)
    initials = "".join(w[0] for w in re.findall(r"[a-z0-9]+", short_name(name).lower())
                       if w not in ("and", "of", "the"))
    return words == title_words or (len(title_words) == 1 and initials == next(iter(title_words)))


def pick_topic(name, suggestions):
    """
    Best Google topic for a company, or None. Only company-type suggestions count (not
    "Ethnic group", "Book", or local sites like "Mining company in Delhi, India"). An exact
    name match wins, then an established entity (mid /m/...) over a lesser namesake
    ("TESLA, Inc." the software company has a /g/ mid), then word overlap. A company with
    another name (IndiGo for "InterGlobe Aviation") is only taken as Google's top hit.
    """
    words = name_words(name)
    candidates = []
    for i, s in enumerate(suggestions):
        kind = s.get("type", "")
        if PLACE_TYPE.search(kind) or NOT_COMPANY_TYPE.search(kind) or not COMPANY_TYPE.search(kind):
            continue
        exact = same_name(name, s["title"])
        overlap = len(words & name_words(s["title"])) / len(words | name_words(s["title"]))
        if exact or overlap or i == 0:
            candidates.append(((exact, s["mid"].startswith("/m/"), overlap, -i), s))
    if not candidates:
        return None
    s = max(candidates, key=lambda c: c[0])[1]
    return s["mid"], s["title"], s.get("type", "")


def resolve_topic(name, trends):
    """Topic for a company name; retries with the short name, then falls back to the name as a search term."""
    pick = pick_topic(name, trends.suggestions(name))
    if (pick is None or not pick[0].startswith("/m/")) and short_name(name) != name:
        pick = pick_topic(name, trends.suggestions(short_name(name))) or pick
    return pick or (name, name, "search term")


def load_topics(path, stocks, trends):
    """symbol -> query, resolving (and saving) any stock that has no entry yet."""
    topics = pd.read_csv(path, keep_default_na=False) if os.path.exists(path) else pd.DataFrame(
        columns=["symbol", "name", "query", "title", "type", "status"])
    known = set(topics["symbol"])
    missing = stocks[~stocks["symbol"].isin(known)]
    if len(missing):
        print(f"  resolving Google topics for {len(missing)} companies")
    new_rows = []
    for i, (symbol, name) in enumerate(zip(missing["symbol"], missing["name"]), 1):
        name = clean_name(name)
        query, title, kind = resolve_topic(name, trends)
        new_rows.append({"symbol": symbol, "name": name, "query": query, "title": title, "type": kind,
                         "status": ""})
        if i % 25 == 0 or i == len(missing):
            topics = pd.concat([topics, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows = []
            topics.to_csv(path, index=False)
            print(f"    {i}/{len(missing)}")
    return topics


def grid(resolution, now, oldest):
    """Fixed windows from 2004 on; the part not covered by a settled window is the tail."""
    step = WINDOW[resolution]
    windows = []
    start = START
    while start + step <= now - SETTLE:
        if start + step > oldest:
            windows.append((start, start + step - dt.timedelta(hours=1 if resolution == "1h" else 24)))
        start += step
    tail = (now - step + dt.timedelta(hours=1 if resolution == "1h" else 24), now)
    return windows, tail


def rescale(fine, coarse):
    """Put each window of `fine` on the scale of `coarse` (one factor per window)."""
    fine = fine.reset_index()  # dates repeat where the tail overlaps a grid window
    pos = coarse.index.searchsorted(fine["date"], side="right") - 1
    fine["period"] = coarse.index[pos.clip(0)].where(pos >= 0)
    fine["value"] = 0.0
    for _, rows in fine.groupby("window"):
        means = rows.groupby("period")["raw"].mean()
        target = coarse.reindex(means.index).dropna()
        denom = means[target.index].sum()
        fine.loc[rows.index, "value"] = rows["raw"] * (target.sum() / denom if denom > 0 else 0.0)
    return fine.drop(columns="period").set_index("date")


def read_series(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)
    return df


def update_resolution(trends, query, geo, resolution, path, coarse_path, now, hourly_days):
    """Fetch what is missing for one stock at one resolution. Returns False if Google has no data."""
    if resolution == "1mo":
        raw = trends.interest(query, geo, START, now)
        if raw.empty or raw.max() == 0:
            return False
        pd.DataFrame({"value": raw, "raw": raw, "window": "all"}).rename_axis("date").to_csv(path)
        return True

    coarse = read_series(coarse_path)["value"]
    oldest = now - dt.timedelta(days=hourly_days) if resolution == "1h" else START
    windows, tail = grid(resolution, now, oldest)
    stored = read_series(path) if os.path.exists(path) else pd.DataFrame(columns=["value", "raw", "window"])
    stored = stored[stored["window"].astype(str) != "tail"]  # the tail is always re-fetched
    done = set(stored["window"].astype(str))

    todo = [w for w in windows if w[0].isoformat() not in done] + [tail]
    parts = [stored[["raw", "window"]]]
    for i, (start, end) in enumerate(todo, 1):
        raw = trends.interest(query, geo, start, end, hourly=resolution == "1h")
        label = "tail" if (start, end) == tail else start.isoformat()
        parts.append(pd.DataFrame({"raw": raw, "window": label}))
        if i % 10 == 0 or i == len(todo):
            fine = pd.concat([p for p in parts if len(p)]).rename_axis("date")
            fine = rescale(fine, coarse)
            fine = fine[~fine.index.duplicated(keep="first")].sort_index()  # grid rows win over the tail
            fine[["value", "raw", "window"]].to_csv(path)
    return True


def dollar_volume(output_dir, market, symbol):
    path = os.path.join(output_dir, market, "1d", f"{symbol}.csv.gz")
    if not os.path.exists(path):
        return 0.0
    df = pd.read_csv(path, usecols=["Close", "Volume"]).tail(252)
    return float((df["Close"] * df["Volume"]).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--markets", default="us,in", help="comma list of: us, in")
    parser.add_argument("--top", type=int, default=100, help="most traded N stocks per market (0 = all)")
    parser.add_argument("--symbols", help="comma list of symbols to fetch instead of --top, e.g. AAPL,TCS.NS")
    parser.add_argument("--resolutions", default="1mo,1wk", help="comma list of: 1mo, 1wk, 1d, 1h")
    parser.add_argument("--hourly-days", type=int, default=730)
    parser.add_argument("--delay", type=float, default=15.0, help="starting seconds between Google requests")
    args = parser.parse_args()

    resolutions = [r for r in ["1mo", "1wk", "1d", "1h"] if r in args.resolutions.split(",")]
    trends = Trends(args.delay)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)

    universe = load_universe(args.markets.split(","), include_etfs=False)
    markets = {}
    for market, stocks in universe.groupby("market", sort=False):
        if args.symbols and not stocks["symbol"].isin(args.symbols.split(",")).any():
            continue
        base = os.path.join(args.output, market, "trends")
        for res in resolutions:
            os.makedirs(os.path.join(base, res), exist_ok=True)
        print(f"{market}: ranking {len(stocks)} stocks by dollar volume")
        stocks = stocks.assign(dv=[dollar_volume(args.output, market, s) for s in stocks["symbol"]])
        stocks = stocks.sort_values("dv", ascending=False)
        if args.symbols:
            stocks = stocks[stocks["symbol"].isin(args.symbols.split(","))]
        elif args.top:
            stocks = stocks.head(args.top)
        topics_path = os.path.join(base, "topics.csv")
        markets[market] = (base, topics_path, load_topics(topics_path, stocks, trends).set_index("symbol"),
                           list(stocks["symbol"]))

    # One resolution at a time across both markets, coarse first (finer ones are scaled onto it)
    for res in resolutions:
        jobs = [(m, s) for m, (_, _, _, symbols) in markets.items() for s in symbols]
        print(f"{res}: {len(jobs)} stocks")
        started, fetched = time.time(), 0
        for i, (market, symbol) in enumerate(jobs, 1):
            base, topics_path, topics, _ = markets[market]
            if topics.at[symbol, "status"] == "no data":
                continue
            path = os.path.join(base, res, f"{symbol}.csv.gz")
            coarse_path = os.path.join(base, COARSER.get(res, res), f"{symbol}.csv.gz")
            if os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH[res].total_seconds():
                continue
            if res != "1mo" and not os.path.exists(coarse_path):
                continue
            try:
                found = update_resolution(trends, topics.at[symbol, "query"], GEO[market], res, path,
                                          coarse_path, now, args.hourly_days)
            except ResponseError as e:
                print(f"  ERROR {symbol}: {e}")
                continue
            if not found:
                topics.at[symbol, "status"] = "no data"
                topics.reset_index().to_csv(topics_path, index=False)
            fetched += 1
            if fetched % 10 == 0:
                elapsed = time.time() - started
                print(f"  [{i}/{len(jobs)}] {symbol}  elapsed {elapsed / 60:.0f} min, "
                      f"ETA {elapsed / fetched * (len(jobs) - i) / 60:.0f} min")
    print("Done.")


if __name__ == "__main__":
    main()
