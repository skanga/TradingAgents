from datetime import datetime
import inspect
import logging

import pandas as pd

from tradingagents.dataflows.utils import get_next_weekday, save_output


def test_get_next_weekday_returns_datetime_for_string_weekday():
    result = get_next_weekday("2026-05-05")

    assert result == datetime(2026, 5, 5)
    assert isinstance(result, datetime)


def test_get_next_weekday_moves_string_weekend_to_monday():
    result = get_next_weekday("2026-05-09")

    assert result == datetime(2026, 5, 11)


def test_get_next_weekday_contract_is_annotated_as_datetime_return():
    signature = inspect.signature(get_next_weekday)

    assert signature.return_annotation is datetime


def test_save_output_logs_saved_path(tmp_path, caplog):
    output_path = tmp_path / "out.csv"
    caplog.set_level(logging.INFO)

    save_output(pd.DataFrame({"a": [1]}), "sample", str(output_path))

    assert output_path.exists()
    assert "sample saved to" in caplog.text
