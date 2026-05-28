from cli.models import AnalystType, AssetType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.graph.propagation import Propagator


def test_detects_crypto_pair_symbols():
    assert detect_asset_type("BTC-USD") == AssetType.CRYPTO
    assert detect_asset_type("eth-usdt") == AssetType.CRYPTO


def test_defaults_non_crypto_symbols_to_stock():
    assert detect_asset_type("AAPL") == AssetType.STOCK
    assert detect_asset_type("CNC.TO") == AssetType.STOCK


def test_filters_out_fundamentals_analyst_for_crypto():
    analysts = [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]

    assert filter_analysts_for_asset_type(analysts, AssetType.CRYPTO) == [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
    ]


def test_keeps_all_analysts_for_stock():
    analysts = [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]

    assert filter_analysts_for_asset_type(analysts, AssetType.STOCK) == analysts


def test_propagator_includes_asset_type_in_initial_state():
    state = Propagator().create_initial_state(
        "BTC-USD", "2026-04-18", asset_type=AssetType.CRYPTO.value
    )

    assert state["asset_type"] == "crypto"


def test_crypto_instrument_context_warns_not_to_assume_company_fundamentals():
    context = build_instrument_context("BTC-USD", asset_type=AssetType.CRYPTO.value)

    assert "crypto asset" in context
    assert "Do not assume company fundamentals are available" in context
