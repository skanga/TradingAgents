import tomllib
from pathlib import Path

from gui import export


def test_render_pdf_uses_matplotlib_pdf_backend():
    state = {
        "market_report": "Market details",
        "final_trade_decision": "BUY",
    }
    meta = {
        "ticker": "SPY",
        "trade_date": "2026-05-07",
        "decision": "BUY",
        "provider": "openai",
        "deep_model": "gpt-5.4",
        "quick_model": "gpt-5.4-mini",
        "run_id": "run-1",
    }

    pdf = export.render_pdf(state, meta)

    assert pdf.startswith(b"%PDF")
    assert export.has_pdf_support()


def test_gui_pdf_export_does_not_depend_on_xhtml2pdf():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert "xhtml2pdf>=0.2.13" not in optional_dependencies["gui"]
    assert "xhtml2pdf>=0.2.13" not in optional_dependencies["service"]


def test_html_export_escapes_raw_script_tags():
    html = export.render_html(
        {"market_report": "<script>alert(1)</script>"},
        {"ticker": "SPY", "trade_date": "2026-05-07"},
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_html_export_escapes_raw_html_event_handlers():
    html = export.render_html(
        {"market_report": '<img src=x onerror="alert(1)">'},
        {"ticker": "SPY", "trade_date": "2026-05-07"},
    )

    assert "<img" not in html
    assert '<img src=x onerror="alert(1)">' not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html


def test_html_export_still_renders_markdown_tables():
    html = export.render_html(
        {"market_report": "| Metric | Value |\n| --- | ---: |\n| Price | 100 |"},
        {"ticker": "SPY", "trade_date": "2026-05-07"},
    )

    assert "<table>" in html
    assert "<td style=\"text-align:right\">100</td>" in html
