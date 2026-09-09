"""Release metadata and framework boundaries; no provider or market requests."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_release_metadata_and_ci_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.5.0"
    assert any(dep.startswith("streamlit") for dep in project["optional-dependencies"]["service"])
    assert "## [0.5.0]" in Path("CHANGELOG.md").read_text(encoding="utf-8")
    assert any(dep.startswith("tomli") and "3.11" in dep for dep in project["optional-dependencies"]["dev"])
    assert 'pip install -e ".[dev,service]"' in Path(".github/workflows/ci.yml").read_text(encoding="utf-8")


@pytest.mark.parametrize("query,expected", [("", ["SPY", "QQQ"]), ("&benchmarks=DIA&benchmarks=IWM", ["DIA", "IWM"])])
def test_chart_query_defaults_and_repeated_values(monkeypatch, query, expected):
    from service.routers import charts

    build = Mock(return_value=None)
    monkeypatch.setattr(charts.charts_mod, "build_comparison_frame", build)
    monkeypatch.setattr(charts.charts_mod, "realised_returns_table", Mock(return_value=None))
    app = FastAPI()
    app.include_router(charts.router)
    with TestClient(app) as client:
        response = client.get("/charts/comparison?ticker=AAPL&trade_date=2024-01-15" + query)
    assert response.status_code == 200
    assert response.json()["benchmarks"] == expected
    assert build.call_args.kwargs["benchmarks"] == tuple(expected)
