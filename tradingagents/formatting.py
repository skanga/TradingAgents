"""Shared formatting helpers for reports and batch outputs."""


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def format_percent(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}%"


def format_quantity(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}"
