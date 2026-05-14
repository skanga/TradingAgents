from tradingagents.formatting import format_number, format_percent, format_quantity


def test_shared_formatting_helpers_match_batch_and_allocation_output():
    assert format_number(None) == ""
    assert format_number(12.345) == "12.35"
    assert format_percent(None) == ""
    assert format_percent(0.1234) == "12.34%"
    assert format_quantity(None) == ""
    assert format_quantity(5.0) == "5"
    assert format_quantity(5.125) == "5.1250"
