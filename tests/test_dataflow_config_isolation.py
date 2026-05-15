from tradingagents.dataflows.config import get_config, reset_config, set_config, use_config


def test_config_context_isolation():
    base = get_config()
    token = use_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})
    try:
        assert get_config()["data_vendors"]["core_stock_apis"] == "alpha_vantage"
    finally:
        reset_config(token)

    assert (
        get_config()["data_vendors"].get("core_stock_apis")
        == base["data_vendors"]["core_stock_apis"]
    )


def test_nested_config_context_restores_outer():
    outer = use_config({"output_language": "Spanish"})
    try:
        inner = use_config({"output_language": "Japanese"})
        try:
            assert get_config()["output_language"] == "Japanese"
        finally:
            reset_config(inner)
        assert get_config()["output_language"] == "Spanish"
    finally:
        reset_config(outer)


def test_use_config_merges_from_defaults_while_set_config_merges_current_context():
    token = use_config({"output_language": "Spanish"})
    try:
        set_config({"llm_provider": "openai"})
        assert get_config()["output_language"] == "Spanish"
        assert get_config()["llm_provider"] == "openai"

        inner = use_config({"llm_provider": "anthropic"})
        try:
            assert get_config()["llm_provider"] == "anthropic"
            assert get_config()["output_language"] != "Spanish"
        finally:
            reset_config(inner)
    finally:
        reset_config(token)
