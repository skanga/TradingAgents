from __future__ import annotations

import pytest

from service.streaming import Broadcaster


@pytest.mark.asyncio
async def test_warm_ticker_marks_price_active_without_subscription_queue():
    broadcaster = Broadcaster()

    await broadcaster.warm_ticker("nvda")

    assert "NVDA" in broadcaster._state
    assert broadcaster._active_tickers("price") == {"NVDA"}
    assert broadcaster._subs["price"] == {}


@pytest.mark.asyncio
async def test_unwarm_ticker_stops_price_polling_when_no_subscribers_remain():
    broadcaster = Broadcaster()

    await broadcaster.warm_ticker("NVDA")
    await broadcaster.unwarm_ticker("NVDA")

    assert broadcaster._active_tickers("price") == set()


@pytest.mark.asyncio
async def test_unwarm_ticker_preserves_real_subscriptions():
    broadcaster = Broadcaster()

    queue = await broadcaster.subscribe("price", "NVDA")
    await broadcaster.warm_ticker("NVDA")
    await broadcaster.unwarm_ticker("NVDA")

    assert broadcaster._active_tickers("price") == {"NVDA"}

    await broadcaster.unsubscribe("price", "NVDA", queue)
    assert broadcaster._active_tickers("price") == set()
