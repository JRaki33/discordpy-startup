"""
Tweet Stock Price Alert Monitor
Monitors specific Twitter/X accounts via Nitter RSS feeds and sends desktop
notifications when a tweet is likely to affect stock prices.

No API key needed — uses keyword matching for stock impact detection.

Usage:
    python tweet_monitor.py
"""

import asyncio
import json
import logging
import re
import sys
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple

import aiohttp
import feedparser
from dotenv import load_dotenv
from plyer import notification

# ---------------------------------------------------------------------------
# Nitter public instances (tried in order, with fallback)
# ---------------------------------------------------------------------------

NITTER_INSTANCES = [
    "nitter.poast.org",
    "nitter.privacyredirect.com",
    "nitter.nixnet.services",
    "nitter.1d4.us",
    "nitter.fdn.fr",
    "nitter.unixfox.eu",
    "lightbrd.com",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            RotatingFileHandler(
                'tweet_monitor.log',
                maxBytes=1_000_000,
                backupCount=3,
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s %(levelname)s %(message)s'
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str = 'config.json') -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'accounts' not in config or not config['accounts']:
        raise ValueError("config.json must contain a non-empty 'accounts' list")

    config.setdefault('check_interval_seconds', 10)
    config.setdefault('max_tweets_per_check', 5)
    config.setdefault('dedup_store_path', 'processed_tweets.json')
    config.setdefault('dedup_max_ids', 10000)
    config.setdefault('nitter_instances', NITTER_INSTANCES)
    config.setdefault('fetch_timeout_seconds', 15)
    return config


# ---------------------------------------------------------------------------
# Deduplication store
# ---------------------------------------------------------------------------

def load_processed_ids(path: str, maxlen: int) -> deque:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ids = data.get('processed_ids', [])
        logging.info("Loaded %d processed tweet IDs from %s", len(ids), path)
        return deque(ids, maxlen=maxlen)
    except FileNotFoundError:
        logging.info("No dedup store found at %s, starting fresh", path)
        return deque(maxlen=maxlen)
    except json.JSONDecodeError:
        logging.warning("Corrupted dedup store at %s, starting fresh", path)
        return deque(maxlen=maxlen)


def save_processed_ids(ids: deque, path: str) -> None:
    data = {
        'processed_ids': list(ids),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Nitter RSS fetching
# ---------------------------------------------------------------------------

async def fetch_rss(
    session: aiohttp.ClientSession,
    username: str,
    instances: List[str],
    timeout: int
) -> Optional[feedparser.FeedParserDict]:
    """Try each Nitter instance in order and return the first successful RSS feed."""
    for instance in instances:
        url = f"https://{instance}/{username}/rss"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": "TweetStockMonitor/1.0 (RSS reader)"}
            ) as resp:
                if resp.status != 200:
                    logging.debug(
                        "Instance %s returned HTTP %d for @%s",
                        instance, resp.status, username
                    )
                    continue
                body = await resp.text()
                feed = feedparser.parse(body)
                if feed.bozo and not feed.entries:
                    logging.debug(
                        "Instance %s returned unparseable feed for @%s",
                        instance, username
                    )
                    continue
                logging.debug("Fetched RSS for @%s from %s", username, instance)
                return feed
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.debug("Instance %s failed for @%s: %s", instance, username, e)

    logging.warning("All Nitter instances failed for @%s", username)
    return None


def extract_tweet_id(entry_id: str) -> str:
    """Extract numeric tweet ID from a Nitter RSS entry id URL."""
    # entry.id is typically "https://nitter.instance/username/status/1234567890#m"
    parts = entry_id.rstrip('#m').split('/')
    for part in reversed(parts):
        if part.isdigit():
            return part
    return entry_id  # fallback: use raw id


def extract_tweet_text(entry: feedparser.FeedParserDict) -> str:
    """Extract plain tweet text from an RSS entry."""
    # feedparser puts content in entry.title or entry.summary
    text = entry.get('title', '') or entry.get('summary', '')
    # Remove leading "R to @user: " or "RT @user: " prefix
    if text.startswith('RT @'):
        return ''  # skip retweets
    return text.strip()


# ---------------------------------------------------------------------------
# Keyword-based stock impact detection (no API needed)
# ---------------------------------------------------------------------------

# High-confidence keywords that strongly indicate stock price impact
STOCK_KEYWORDS = [
    # Earnings & financials
    r'\bearnings\b', r'\brevenue\b', r'\bprofit\b', r'\bloss\b',
    r'\bguidance\b', r'\bforecast\b', r'\boutlook\b', r'\bEPS\b',
    r'\bbeat\b', r'\bmiss\b', r'\bquarterly\b', r'\bannual results\b',
    # M&A
    r'\bacquisition\b', r'\bmerger\b', r'\bbuyout\b', r'\btakeover\b',
    r'\bacquires?\b', r'\bmerges?\b',
    # Market events
    r'\bIPO\b', r'\boffering\b', r'\bdilution\b', r'\bbuyback\b',
    r'\bdividend\b', r'\bsplit\b',
    # Negative events
    r'\bbankruptcy\b', r'\bdefault\b', r'\blayoff\b', r'\blayoffs\b',
    r'\brecall\b', r'\bshutdown\b', r'\bfraud\b', r'\bscandal\b',
    # Regulatory / legal
    r'\bSEC\b', r'\blawsuit\b', r'\bsettlement\b', r'\bfine\b',
    r'\bpenalty\b', r'\bregulat\w+\b', r'\bapproval\b', r'\bban\b',
    # Leadership
    r'\bCEO\b', r'\bCFO\b', r'\bresign\w*\b', r'\bfired\b',
    r'\bappointed\b', r'\bsteps down\b',
    # Macro / Fed
    r'\bFed\b', r'\brate hike\b', r'\brate cut\b', r'\binterest rate\b',
    r'\binflation\b', r'\bCPI\b', r'\bGDP\b', r'\brecession\b',
    # Trade / geopolitics affecting markets
    r'\btariff\b', r'\bsanction\w*\b', r'\btrade war\b', r'\bembargo\b',
    r'\boil\b', r'\bcrude\b', r'\bOPEC\b',
    # Stock tickers (e.g. $TSLA)
    r'\$[A-Z]{1,5}\b',
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in STOCK_KEYWORDS]


def analyze_tweet(tweet_text: str) -> Tuple[bool, str]:
    """
    Returns (affects_stock, reason).
    No API call needed — pure keyword matching.
    """
    matched = [p.pattern for p in _COMPILED if p.search(tweet_text)]
    if matched:
        # Clean up the pattern for display
        keywords = ', '.join(
            re.sub(r'\\b|\\w\+|\(\?i\)', '', m).strip('()').replace('\\', '')
            for m in matched[:3]
        )
        return True, f"Contains market keywords: {keywords}"
    return False, "No stock-related keywords found"


# ---------------------------------------------------------------------------
# Desktop notification
# ---------------------------------------------------------------------------

def send_desktop_notification(tweet_text: str, username: str, reason: str) -> None:
    title = f"Stock Alert: @{username}"
    message = f"{reason}\n\n{tweet_text[:200]}"

    try:
        notification.notify(
            title=title,
            message=message,
            app_name="TweetStockMonitor",
            timeout=15
        )
        logging.info("Desktop notification sent for @%s", username)
    except Exception as e:
        logging.warning("Desktop notification failed: %s", e)


# ---------------------------------------------------------------------------
# Account checking
# ---------------------------------------------------------------------------

async def check_account(
    session: aiohttp.ClientSession,
    account: Dict,
    processed_ids: deque,
    config: Dict
) -> List[str]:
    username = account['username']
    display_name = account.get('display_name', username)
    new_ids: List[str] = []

    feed = await fetch_rss(
        session,
        username,
        config['nitter_instances'],
        config['fetch_timeout_seconds']
    )
    if feed is None:
        return new_ids

    entries = feed.entries[:config['max_tweets_per_check']]

    for entry in entries:
        tweet_id = extract_tweet_id(entry.get('id', ''))

        if tweet_id in processed_ids:
            continue

        new_ids.append(tweet_id)
        tweet_text = extract_tweet_text(entry)

        if not tweet_text:
            logging.debug("Skipping RT or empty tweet %s from @%s", tweet_id, username)
            continue

        logging.info(
            "Analyzing tweet %s from @%s: %s",
            tweet_id, username, tweet_text[:80]
        )

        affects_stock, reason = analyze_tweet(tweet_text)

        if affects_stock:
            logging.info("Stock-impacting tweet from @%s: %s", username, reason)
            send_desktop_notification(tweet_text, username, reason)
        else:
            logging.debug("Tweet %s from @%s: no stock impact", tweet_id, username)

    return new_ids


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_monitor_loop(config: Dict) -> None:
    load_dotenv()

    dedup_path = config['dedup_store_path']
    processed_ids = load_processed_ids(dedup_path, config['dedup_max_ids'])

    accounts: List[Dict] = config['accounts']
    interval: int = config['check_interval_seconds']

    logging.info(
        "Monitoring started: %d accounts, interval=%ds, instances=%s",
        len(accounts),
        interval,
        config['nitter_instances']
    )

    backoff = interval

    connector = aiohttp.TCPConnector(ssl=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                for account in accounts:
                    new_ids = await check_account(
                        session,
                        account,
                        processed_ids,
                        config
                    )
                    for tid in new_ids:
                        processed_ids.append(tid)

                save_processed_ids(processed_ids, dedup_path)
                backoff = interval

            except Exception as e:
                logging.error("Unexpected error in monitor loop: %s", e)
                backoff = min(backoff * 2, 1800)

            await asyncio.sleep(backoff)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    setup_logging()
    try:
        config = load_config('config.json')
    except (FileNotFoundError, ValueError) as e:
        logging.error("Failed to load config: %s", e)
        sys.exit(1)

    try:
        asyncio.run(run_monitor_loop(config))
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user")
    except EnvironmentError as e:
        logging.error("%s", e)
        sys.exit(1)
