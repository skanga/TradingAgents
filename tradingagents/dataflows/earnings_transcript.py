"""Earnings call transcript fetch + LLM-scored sentiment.

Source: The Motley Fool transcripts archive
(``https://www.fool.com/quote/{exchange}/{ticker}/transcripts/``).
The listing page returns the most recent transcripts for a ticker; the
first link is the latest one we attempt to fetch and parse.

Sentiment scoring uses the project's configured ``quick_thinking_llm``
rather than a local FinBERT model. This avoids ~2 GB of ``transformers``
+ ``torch`` and lets users score with whichever provider they already use.

The function returns a Markdown string summarising:
- Management Prepared Remarks: sentiment label + one-paragraph rationale
- Q&A section: sentiment label + one-paragraph rationale
- Hedge-word frequency per 1k words for each section
- Q&A deflection rate (frequency of non-answer phrases like "we don't break
  that out", "refer you to our guidance")

Caching: per-ticker reports cached for 7 days, since transcripts are
quarterly events.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_SOURCE = "earnings_transcript"
_LISTING_URL = "https://www.fool.com/quote/{exchange}/{ticker}/transcripts/"
_EXCHANGES_TO_TRY = ("nasdaq", "nyse")
_USER_AGENT = (
    "Mozilla/5.0 (compatible; TradingAgents/0.2; +https://github.com/TauricResearch/TradingAgents)"
)
_TIMEOUT = 30
_MIN_TRANSCRIPT_CHARS = 2000

# Hedging language — case-insensitive match. Phrases first so they're not
# eaten by the single-word patterns.
_HEDGE_PATTERNS = (
    r"remain[s]? to be seen",
    r"hard to say",
    r"too early to (?:tell|call|say)",
    r"\buncertain(?:ty|ties)?\b",
    r"\bchallenging\b",
    r"\bheadwind[s]?\b",
    r"\bcautious(?:ly)?\b",
    r"\bsoftness\b",
    r"\bweakness\b",
    r"\bdifficult\b",
    r"\bvolatil(?:e|ity)\b",
    r"\bpressure(?:d|s)?\b",
    r"\bcompressed\b",
    r"\bdeceleration\b",
)

# Q&A deflection / non-answer phrases — same matching style.
_DEFLECTION_PATTERNS = (
    r"we don'?t (?:break|disclose|comment|share|provide|guide)",
    r"we won'?t (?:comment|share|disclose|guide)",
    r"refer you (?:back )?to (?:our )?(?:prior|previous )?guidance",
    r"not (?:something we|a number we) (?:disclose|share|comment)",
    r"not in a position to (?:share|comment|discuss)",
    r"we'?re not (?:going to|providing|breaking|sharing|disclosing)",
    r"that'?s not something (?:we|that we)",
    r"we'?ll (?:provide|share) more (?:detail|color) (?:later|in the future|next quarter)",
    r"more to come",
)

_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)
_DEFLECTION_RE = re.compile("|".join(_DEFLECTION_PATTERNS), re.IGNORECASE)


# --- Public API -------------------------------------------------------------


def get_earnings_transcript_sentiment(ticker: str) -> str:
    cache_key = {
        "kind": "report",
        "ticker": ticker.upper(),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=7 * 24 * 3600)
    if cached is not None:
        return cached

    try:
        transcript_url = _find_latest_transcript_url(ticker)
        if transcript_url is None:
            return f"[Earnings transcript unavailable: no transcript found for {ticker} on Motley Fool. Proceed with available data.]"

        text = _fetch_transcript_text(transcript_url)
        if not text or len(text) < _MIN_TRANSCRIPT_CHARS:
            return f"[Earnings transcript unavailable: fetched page for {ticker} too short ({len(text or '')} chars). Proceed with available data.]"

        title = _extract_title(text) or _slug_from_url(transcript_url)
        prepared, qa = _split_sections(text)

        prepared_hedge = _per_1k(prepared, _HEDGE_RE)
        qa_hedge = _per_1k(qa, _HEDGE_RE)
        qa_deflection = _per_1k(qa, _DEFLECTION_RE)

        sentiment = _score_sentiment_with_llm(prepared, qa)
        report = _format_report(
            ticker=ticker,
            title=title,
            transcript_url=transcript_url,
            prepared_words=_word_count(prepared),
            qa_words=_word_count(qa),
            prepared_hedge=prepared_hedge,
            qa_hedge=qa_hedge,
            qa_deflection=qa_deflection,
            sentiment=sentiment,
        )
        cache_put(_SOURCE, cache_key, report)
        return report
    except Exception as e:
        logger.exception("earnings transcript pipeline failed for %s", ticker)
        return f"[Earnings transcript unavailable: {e}. Proceed with available data.]"


# --- Fetch + parse ---------------------------------------------------------


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def _find_latest_transcript_url(ticker: str) -> Optional[str]:
    """Hit the Motley Fool ticker listing page on each candidate exchange."""
    sess = _session()
    for exchange in _EXCHANGES_TO_TRY:
        url = _LISTING_URL.format(exchange=exchange, ticker=ticker.lower())
        try:
            resp = sess.get(url, timeout=_TIMEOUT)
        except requests.RequestException:
            continue
        if resp.status_code != 200 or not resp.text:
            continue
        link = _extract_first_transcript_link(resp.text)
        if link:
            return link if link.startswith("http") else f"https://www.fool.com{link}"
    return None


def _extract_first_transcript_link(html: str) -> Optional[str]:
    """Pick the first /earnings/call-transcripts/ href that looks like a transcript."""
    # Use a regex rather than bs4 here because the listing page markup is
    # extremely JS-heavy; the hrefs are present in the source as plain links.
    matches = re.findall(
        r'href="(/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[a-z0-9\-/]+)"',
        html,
        re.IGNORECASE,
    )
    return matches[0] if matches else None


def _fetch_transcript_text(url: str) -> str:
    """Fetch the transcript page and return body text via BeautifulSoup."""
    sess = _session()
    resp = sess.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    return _extract_article_text(resp.text)


def _extract_article_text(html: str) -> str:
    """Pull the article body, stripping nav/ads/scripts."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.body
    if article is None:
        return ""
    text = article.get_text("\n", strip=True)
    # Collapse runs of blank lines so the section splitter heuristics work.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_title(text: str) -> Optional[str]:
    # The Motley Fool transcripts open with the article H1 in line 1 or 2.
    for line in text.splitlines()[:5]:
        line = line.strip()
        if "Earnings Call" in line or "earnings call" in line:
            return line
    return None


def _slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].replace(".aspx", "")


def _split_sections(text: str) -> tuple[str, str]:
    """Split the transcript into (Prepared Remarks, Q&A).

    Falls back to a 65/35 split if no recognisable section header is found.
    """
    qa_markers = (
        r"questions\s*(?:and|&)\s*answers",
        r"question[-\s]and[-\s]answer\s+session",
        r"q\s*&\s*a\s+session",
        r"\boperator\s*:\s*we will now begin",  # common Q&A intro
    )
    pattern = re.compile("(" + "|".join(qa_markers) + ")", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        idx = match.start()
        return text[:idx].strip(), text[idx:].strip()
    # Fallback: assume the prepared remarks take the front 65% of the text.
    cut = int(len(text) * 0.65)
    return text[:cut].strip(), text[cut:].strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def _per_1k(text: str, pattern: re.Pattern) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    matches = len(pattern.findall(text or ""))
    return matches / words * 1000.0


# --- LLM scoring -----------------------------------------------------------


def _score_sentiment_with_llm(prepared: str, qa: str) -> dict:
    """Return ``{prepared_sentiment, prepared_reason, qa_sentiment, qa_reason}``.

    Falls back to a neutral stub if the LLM call fails or the response can
    not be parsed; the caller will still produce a useful (degraded) report.
    """
    fallback = {
        "prepared_sentiment": "neutral",
        "prepared_reason": "LLM scoring unavailable; sentiment defaulted to neutral.",
        "qa_sentiment": "neutral",
        "qa_reason": "LLM scoring unavailable; sentiment defaulted to neutral.",
    }
    try:
        llm = _build_quick_llm()
    except Exception as e:
        logger.warning("could not construct quick LLM for transcript scoring: %s", e)
        return fallback

    # Cap each section so we stay well inside small models' context windows.
    prepared_excerpt = (prepared or "")[:8000]
    qa_excerpt = (qa or "")[:8000]
    prompt = (
        "You are scoring the sentiment of an earnings call transcript on behalf of "
        "an equity research team. Respond ONLY with a single JSON object using the "
        "exact keys shown — no prose, no fences.\n\n"
        '{\n'
        '  "prepared_sentiment": "positive" | "neutral" | "negative",\n'
        '  "prepared_reason": "one or two sentences citing concrete language from the section",\n'
        '  "qa_sentiment": "positive" | "neutral" | "negative",\n'
        '  "qa_reason": "one or two sentences citing concrete language from the section"\n'
        '}\n\n'
        "Be precise. Use 'neutral' when management language is balanced or routine; "
        "use 'negative' when there is meaningful hedging, missed targets, lowered "
        "guidance, or evasive Q&A; use 'positive' when there is clear acceleration, "
        "raised guidance, or unambiguous confidence. Quote two or three short phrases "
        "(<= 8 words) as evidence in the reason fields.\n\n"
        f"=== Management Prepared Remarks ===\n{prepared_excerpt}\n\n"
        f"=== Q&A ===\n{qa_excerpt}\n"
    )

    try:
        result = llm.invoke(prompt)
        raw = getattr(result, "content", None) or str(result)
    except Exception as e:
        logger.warning("transcript LLM call failed: %s", e)
        return fallback

    parsed = _parse_json_object(raw)
    if not parsed:
        return fallback

    out = dict(fallback)
    for key in ("prepared_sentiment", "prepared_reason", "qa_sentiment", "qa_reason"):
        v = parsed.get(key)
        if isinstance(v, str) and v.strip():
            out[key] = v.strip()
    # Normalise sentiment labels
    for key in ("prepared_sentiment", "qa_sentiment"):
        out[key] = _normalise_sentiment(out[key])
    return out


def _build_quick_llm():
    """Construct the configured quick_thinking_llm for one-shot scoring."""
    from tradingagents.llm_clients import create_llm_client
    config = get_config()
    client = create_llm_client(
        provider=config.get("llm_provider", "openai"),
        model=config.get("quick_think_llm") or config.get("deep_think_llm"),
        base_url=config.get("backend_url"),
    )
    return client.get_llm()


def _parse_json_object(raw: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response."""
    if not raw:
        return None
    # Try plain parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to locate the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalise_sentiment(label: str) -> str:
    s = (label or "").strip().lower()
    if s in ("positive", "bullish", "constructive"):
        return "positive"
    if s in ("negative", "bearish", "cautious", "weak"):
        return "negative"
    return "neutral"


# --- Reporting --------------------------------------------------------------


def _format_report(
    *,
    ticker: str,
    title: str,
    transcript_url: str,
    prepared_words: int,
    qa_words: int,
    prepared_hedge: float,
    qa_hedge: float,
    qa_deflection: float,
    sentiment: dict,
) -> str:
    risk_flags: list[str] = []
    if qa_hedge >= 8.0 and sentiment["prepared_sentiment"] == "positive":
        risk_flags.append(
            f"Prepared remarks read POSITIVE but Q&A hedge density is high "
            f"({qa_hedge:.1f}/1k words) — management may be more guarded under "
            "questioning than in the script."
        )
    if qa_deflection >= 4.0:
        risk_flags.append(
            f"Q&A deflection rate is elevated ({qa_deflection:.1f}/1k words) — "
            "management is steering analysts away from specifics."
        )
    if sentiment["qa_sentiment"] == "negative" and sentiment["prepared_sentiment"] != "negative":
        risk_flags.append(
            "Q&A sentiment is more negative than the prepared remarks — the "
            "scripted message and the unscripted answers are diverging."
        )

    lines = [
        f"## Earnings Call Sentiment for {ticker}",
        "",
        f"**Source**: {title}",
        f"<{transcript_url}>",
        "",
        "| Section | Words | Sentiment | Hedge / 1k | Notes |",
        "|---|---|---|---|---|",
        f"| Prepared Remarks | {prepared_words:,} | "
        f"{sentiment['prepared_sentiment'].upper()} | {prepared_hedge:.1f} | — |",
        f"| Q&A | {qa_words:,} | "
        f"{sentiment['qa_sentiment'].upper()} | {qa_hedge:.1f} | "
        f"deflection {qa_deflection:.1f}/1k |",
        "",
        "**Prepared remarks reasoning**: " + sentiment["prepared_reason"],
        "",
        "**Q&A reasoning**: " + sentiment["qa_reason"],
    ]
    if risk_flags:
        lines.append("")
        lines.append("**Risk flags**:")
        for flag in risk_flags:
            lines.append(f"- {flag}")
    lines.append("")
    lines.append(
        "_Note: prior-quarter comparison is not yet implemented — sentiment shift "
        "vs the previous call is shown as the hedge-word frequency only._"
    )
    return "\n".join(lines)
