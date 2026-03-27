"""
Tweet Stock Price Alert Monitor
Monitors specific Twitter/X accounts via Nitter RSS feeds and sends desktop
notifications when a tweet is likely to affect stock prices.

No Twitter account or API key needed — uses public Nitter instances.

Usage:
    python tweet_monitor.py

Required environment variables (set in .env):
    ANTHROPIC_API_KEY - Anthropic API key
"""

import asyncio
import json
import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import aiohttp
import anthropic
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
    config.setdefault('claude_model', 'claude-haiku-4-5-20251001')
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
# Claude analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst assistant specialized in identifying market-moving information.

Your task: analyze a tweet and determine whether it is likely to affect the stock price of any publicly traded company.

Respond ONLY with a valid JSON object in this exact format:
{
  "affects_stock": true or false,
  "confidence": "high", "medium", or "low",
  "ticker_symbols": ["TSLA", "AAPL"],
  "reason": "One sentence explanation"
}

Guidelines:
- "affects_stock" is true if the tweet contains: product announcements, earnings hints, regulatory news, leadership changes, major contracts, acquisitions, mergers, lawsuits, financial results, or strong positive/negative sentiment about a company's business.
- "affects_stock" is false for: personal opinions unrelated to business, sports, food, general politics with no specific company impact, jokes, or routine social posts.
- Only include ticker symbols you are highly confident about.
- Keep "reason" to one concise sentence."""


async def analyze_tweet(
    client: anthropic.AsyncAnthropic,
    tweet_text: str,
    username: str,
    display_name: str,
    model: str
) -> Optional[Dict]:
    user_message = (
        f"Twitter account: @{username} ({display_name})\n"
        f"Tweet text: {tweet_text[:1000]}"
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": user_message}]
        )

        text = next(
            (b.text for b in response.content if b.type == "text"),
            None
        )
        if text is None:
            logging.warning("Claude returned no text for tweet from @%s", username)
            return None

        return json.loads(text)

    except json.JSONDecodeError as e:
        logging.warning("Claude returned invalid JSON for @%s tweet: %s", username, e)
        return None
    except anthropic.RateLimitError:
        logging.warning("Anthropic rate limit hit, skipping tweet from @%s", username)
        return None
    except Exception as e:
        logging.warning("Claude analysis failed for @%s: %s", username, e)
        return None


# ---------------------------------------------------------------------------
# Desktop notification
# ---------------------------------------------------------------------------

def send_desktop_notification(tweet_text: str, username: str, analysis: Dict) -> None:
    tickers = analysis.get('ticker_symbols', [])
    ticker_str = ', '.join(tickers) if tickers else 'ticker unknown'
    confidence = analysis.get('confidence', '')
    reason = analysis.get('reason', '')

    title = f"Stock Alert: @{username} [{ticker_str}]"
    if confidence:
        title += f" ({confidence})"

    message = f"{reason}\n\n{tweet_text[:200]}"

    try:
        notification.notify(
            title=title,
            message=message,
            app_name="TweetStockMonitor",
            timeout=15
        )
        logging.info("Desktop notification sent for @%s (%s)", username, ticker_str)
    except Exception as e:
        logging.warning("Desktop notification failed: %s", e)


# ---------------------------------------------------------------------------
# Account checking
# ---------------------------------------------------------------------------

async def check_account(
    session: aiohttp.ClientSession,
    anthropic_client: anthropic.AsyncAnthropic,
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

        analysis = await analyze_tweet(
            anthropic_client,
            tweet_text,
            username,
            display_name,
            config['claude_model']
        )

        if analysis is None:
            continue

        if analysis.get('affects_stock'):
            logging.info(
                "Stock-impacting tweet from @%s: %s (confidence: %s)",
                username,
                analysis.get('reason', ''),
                analysis.get('confidence', '')
            )
            send_desktop_notification(tweet_text, username, analysis)
        else:
            logging.debug(
                "Tweet %s from @%s: no stock impact (%s)",
                tweet_id, username, analysis.get('reason', '')
            )

    return new_ids


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_monitor_loop(config: Dict) -> None:
    load_dotenv()

    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not anthropic_api_key:
        raise EnvironmentError("Missing ANTHROPIC_API_KEY. Set it in .env")

    anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

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
                        anthropic_client,
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
