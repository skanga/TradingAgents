"""Background poller + WebSocket fanout for real-time prices and news.

How it works:

- A single ``Broadcaster`` instance (module-level) keeps:
    - one ``asyncio.Queue`` per (channel, ticker) subscription
    - a set of "active tickers" derived from current subscriptions and
      explicit warm registrations

- A background task wakes up every N seconds, pulls fresh data from
  yfinance for every active ticker, diffs against the last snapshot,
  and pushes deltas onto each subscriber's queue.

- WebSocket endpoints (in ``routers/streaming.py``) subscribe a queue
  and forward events to the client until the socket closes.

This is single-process, single-host. For multi-process scaling you'd
swap the Queue fanout for Redis pub/sub — the API surface stays the same.

Frequencies (during US market hours; we poll regardless of hours and
let the client decide what to do with stale weekend data):
- prices: every 30s
- news:   every 5 minutes
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

PRICE_INTERVAL = 30.0      # seconds between price polls
NEWS_INTERVAL = 300.0      # seconds between news polls
PRICE_HISTORY_LEN = 120    # in-memory recent ticks per ticker (~1 hour)


@dataclass
class TickerState:
    """In-memory state for one watched ticker."""
    ticker: str
    last_price: float | None = None
    last_change: float | None = None
    last_change_pct: float | None = None
    last_volume: int | None = None
    last_polled: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)  # {ts, price}
    last_news_titles: set[str] = field(default_factory=set)
    last_news_polled: float | None = None


class Broadcaster:
    """Singleton-ish: subscribers + per-ticker state + the polling loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()
        # Per-channel subscribers: {channel: {ticker: [queue, ...]}}.
        # ``channel`` is "price" or "news"; ``ticker`` is uppercase symbol.
        self._subs: dict[str, dict[str, list[asyncio.Queue[dict[str, Any]]]]] = {
            "price": {}, "news": {},
        }
        # Price tickers to poll even before a browser opens a WebSocket.
        # Values are source tokens, so independent features can warm and
        # unwarm the same ticker without stepping on each other.
        self._warm_price_tickers: dict[str, set[str]] = {}
        self._state: dict[str, TickerState] = {}
        self._task: asyncio.Task | None = None
        self._stop = False

    # ---- Public lifecycle ------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._task is not None and not self._task.done():
            return
        self._loop = loop
        self._task = loop.create_task(self._poll_forever(), name="broadcaster")

    async def stop(self) -> None:
        self._stop = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    # ---- Subscription API ------------------------------------------

    async def subscribe(self, channel: str, ticker: str
                       ) -> asyncio.Queue[dict[str, Any]]:
        ticker = ticker.upper()
        if channel not in self._subs:
            raise ValueError(f"unknown channel {channel!r}")
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subs[channel].setdefault(ticker, []).append(q)
            self._state.setdefault(ticker, TickerState(ticker=ticker))
        # Replay last snapshot if we have one, so a fresh client doesn't
        # wait up to 30s for the next poll.
        await self._send_initial_snapshot(channel, ticker, q)
        return q

    async def unsubscribe(self, channel: str, ticker: str,
                          q: asyncio.Queue[dict[str, Any]]) -> None:
        ticker = ticker.upper()
        async with self._lock:
            queues = self._subs.get(channel, {}).get(ticker)
            if queues:
                with contextlib.suppress(ValueError):
                    queues.remove(q)
                if not queues:
                    del self._subs[channel][ticker]

    async def warm_ticker(self, ticker: str, *, source: str = "manual") -> None:
        """Poll a price ticker without allocating a subscriber queue."""
        ticker = ticker.upper()
        async with self._lock:
            self._warm_price_tickers.setdefault(ticker, set()).add(source)
            self._state.setdefault(ticker, TickerState(ticker=ticker))

    async def unwarm_ticker(self, ticker: str, *, source: str = "manual") -> None:
        ticker = ticker.upper()
        async with self._lock:
            sources = self._warm_price_tickers.get(ticker)
            if not sources:
                return
            sources.discard(source)
            if not sources:
                del self._warm_price_tickers[ticker]

    async def _send_initial_snapshot(self, channel: str, ticker: str,
                                     q: asyncio.Queue[dict[str, Any]]) -> None:
        st = self._state.get(ticker)
        if not st:
            return
        if channel == "price" and st.last_price is not None:
            await q.put(self._price_event(st))
        # No initial news snapshot — clients will get the next batch.

    def _price_event(self, st: TickerState) -> dict[str, Any]:
        return {
            "type": "price",
            "ticker": st.ticker,
            "price": st.last_price,
            "change": st.last_change,
            "change_pct": st.last_change_pct,
            "volume": st.last_volume,
            "polled_at": st.last_polled,
            "history": list(st.history),
        }

    # ---- Polling loop ---------------------------------------------

    def _active_tickers(self, channel: str) -> set[str]:
        tickers = set(self._subs.get(channel, {}).keys())
        if channel == "price":
            tickers.update(self._warm_price_tickers.keys())
        return tickers

    async def _poll_forever(self) -> None:
        last_news_run = 0.0
        while not self._stop:
            # Prices: every PRICE_INTERVAL.
            tickers = self._active_tickers("price") | self._active_tickers("news")
            if tickers:
                await self._poll_prices(self._active_tickers("price"))

            now = time.time()
            if now - last_news_run >= NEWS_INTERVAL:
                await self._poll_news(self._active_tickers("news"))
                last_news_run = now

            await asyncio.sleep(PRICE_INTERVAL)

    async def _poll_prices(self, tickers: set[str]) -> None:
        if not tickers:
            return
        for ticker in tickers:
            try:
                price, change, change_pct, volume = await asyncio.to_thread(
                    _fetch_price, ticker
                )
            except Exception as e:
                logger.warning("price fetch failed for %s: %s", ticker, e)
                continue
            if price is None:
                continue
            now = time.time()
            st = self._state.get(ticker) or TickerState(ticker=ticker)
            st.last_price = price
            st.last_change = change
            st.last_change_pct = change_pct
            st.last_volume = volume
            st.last_polled = now
            st.history.append({"ts": now, "price": price})
            if len(st.history) > PRICE_HISTORY_LEN:
                st.history = st.history[-PRICE_HISTORY_LEN:]
            self._state[ticker] = st

            event = self._price_event(st)
            for q in list(self._subs.get("price", {}).get(ticker, [])):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)

    async def _poll_news(self, tickers: set[str]) -> None:
        if not tickers:
            return
        for ticker in tickers:
            try:
                articles = await asyncio.to_thread(_fetch_news, ticker)
            except Exception as e:
                logger.warning("news fetch failed for %s: %s", ticker, e)
                continue
            st = self._state.get(ticker) or TickerState(ticker=ticker)
            new = []
            for a in articles:
                key = a.get("title") or a.get("link") or ""
                if key and key not in st.last_news_titles:
                    st.last_news_titles.add(key)
                    new.append(a)
            # Keep last_news_titles bounded.
            if len(st.last_news_titles) > 200:
                st.last_news_titles = set(list(st.last_news_titles)[-200:])
            st.last_news_polled = time.time()
            self._state[ticker] = st

            for article in new:
                event = {"type": "news", "ticker": ticker, **article}
                for q in list(self._subs.get("news", {}).get(ticker, [])):
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(event)


# ---- Sync helpers (called via asyncio.to_thread) ---------------------

def _fetch_price(ticker: str) -> tuple[float | None, float | None, float | None, int | None]:
    """Pull last/prev close from yfinance. Synchronous — call via to_thread."""
    t = yf.Ticker(ticker)
    fast = t.fast_info if hasattr(t, "fast_info") else None
    if fast:
        last = getattr(fast, "last_price", None) or fast.get("last_price")  # type: ignore[arg-type]
        prev = getattr(fast, "previous_close", None) or fast.get("previous_close")  # type: ignore[arg-type]
        vol = getattr(fast, "last_volume", None) or fast.get("last_volume")  # type: ignore[arg-type]
        if last is not None:
            change = (last - prev) if prev is not None else None
            pct = ((last - prev) / prev * 100) if prev else None
            return float(last), (float(change) if change is not None else None), \
                   (float(pct) if pct is not None else None), \
                   (int(vol) if vol is not None else None)
    # Fallback: 1-minute history's last row.
    df = t.history(period="1d", interval="1m", auto_adjust=False)
    if df.empty:
        return None, None, None, None
    last_row = df.iloc[-1]
    last = float(last_row["Close"])
    prev = float(df.iloc[0]["Open"])
    change = last - prev
    pct = (change / prev * 100) if prev else None
    vol = int(last_row.get("Volume", 0) or 0)
    return last, change, pct, vol


def _fetch_news(ticker: str) -> list[dict[str, Any]]:
    """Pull recent news articles. Synchronous."""
    t = yf.Ticker(ticker)
    raw = t.get_news(count=15) if hasattr(t, "get_news") else (t.news or [])
    out: list[dict[str, Any]] = []
    for a in raw:
        # yfinance switches between flat and nested envelopes.
        if isinstance(a, dict) and "content" in a:
            content = a["content"]
            title = content.get("title") or ""
            summary = content.get("summary") or ""
            provider = (content.get("provider") or {}).get("displayName") or "Unknown"
            url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            link = url_obj.get("url") if isinstance(url_obj, dict) else ""
            pub = content.get("pubDate")
            ts = pub
        else:
            title = a.get("title", "")
            summary = ""
            provider = a.get("publisher", "Unknown")
            link = a.get("link", "")
            ts = a.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if title:
            out.append({
                "title": title,
                "summary": summary,
                "publisher": provider,
                "link": link,
                "published_at": ts,
            })
    return out


broadcaster = Broadcaster()
