"""Deterministic sample controls, not bot detection or population estimates."""

import re
import unicodedata
from collections import Counter


MAX_POSTS_PER_AUTHOR = 3


def bounded_sample_limit(value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Social sample limit must be a positive integer")
    return min(value, maximum)


def _normalized(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


# Conservative lexical screening, not semantic equivalence or bot detection.
# Never compare truncated prefixes, and never extend clusters through duplicates.
_MAX_NEAR_CANDIDATES = 300
_PROTECTED_WORDS = set("no not never without cannot can't don't doesn't didn't isn't aren't wasn't weren't won't wouldn't shouldn't couldn't buy sell long short bullish bearish up down rise fall rises falls raised lowered increase decrease growth decline beat miss beats misses revenue earnings profit loss demand supply".split())


def _near_features(content):
    if len(content) > 4000:
        return None
    tokens = re.findall(r"\$?\w+(?:'\w+)?(?:\.\d+)?%?", content.replace("’", "'"))
    if not 8 <= len(tokens) <= 128:
        return None
    protected = (
        tuple(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*%?", content)),
        tuple(t for t in tokens if any(character.isdigit() for character in t)),
        tuple(re.findall(r"\$[a-z][a-z0-9.-]*", content)),
        tuple(sorted(Counter(t for t in tokens if t in _PROTECTED_WORDS).items())),
    )
    return set(tokens), set(zip(tokens, tokens[1:], strict=False)), protected


def _near_match(first, second):
    words, pairs, protected = first
    other_words, other_pairs, other_protected = second
    return (protected == other_protected
            and len(words & other_words) / len(words | other_words) >= 0.85
            and len(pairs & other_pairs) / len(pairs | other_pairs) >= 0.75)


class SocialSample:
    """Keep first occurrences and cap known authors across all supplied batches.

    Callers apply publication-window filtering before selection. Missing authors
    are not grouped into a fictitious single account or asserted independent.
    """

    def __init__(self):
        self.ids = set()
        self.texts = set()
        self.near_candidates = []
        self.near_duplicates = self.near_omissions = 0
        self.observed_authors = Counter()
        self.retained_authors = Counter()
        self.observed = self.retained = self.missing_author = 0
        self.duplicates = self.author_exclusions = self.limit_exclusions = self.empty = 0

    def select(self, posts, limit, *, identity, author, text):
        kept = []
        for post in posts:
            self.observed += 1
            user = _normalized(author(post))
            if user in {"", "?", "unknown", "[deleted]", "[removed]"}:
                user = None
                self.missing_author += 1
            else:
                self.observed_authors[user] += 1
            key = identity(post)
            key = str(key) if key is not None else ""
            content = _normalized(text(post))
            if not content:
                self.empty += 1
                continue
            duplicate = (key and key in self.ids) or content in self.texts
            if not duplicate:
                features = _near_features(content)
                if features is None:
                    self.near_omissions += 1
                elif any(_near_match(features, candidate) for candidate in self.near_candidates):
                    duplicate = True
                    self.near_duplicates += 1
                elif len(self.near_candidates) < _MAX_NEAR_CANDIDATES:
                    self.near_candidates.append(features)
                else:
                    self.near_omissions += 1
            if key:
                self.ids.add(key)
            self.texts.add(content)
            if duplicate:
                self.duplicates += 1
            elif user and self.retained_authors[user] >= MAX_POSTS_PER_AUTHOR:
                self.author_exclusions += 1
            elif len(kept) >= limit:
                self.limit_exclusions += 1
            else:
                kept.append(post)
                self.retained += 1
                if user:
                    self.retained_authors[user] += 1
        return kept

    def summary(self) -> str:
        largest = max(self.observed_authors.values(), default=0)
        retained_largest = max(self.retained_authors.values(), default=0)
        return (
            f"Sample quality (in-window candidates): observed: {self.observed}; retained: {self.retained}; "
            f"duplicates removed: {self.duplicates}; near-duplicates removed: {self.near_duplicates}; "
            f"near-duplicate screen/index limits: {self.near_omissions}; "
            f"author-cap exclusions: {self.author_exclusions}; "
            f"limit exclusions: {self.limit_exclusions}; empty text: {self.empty}; "
            f"missing author: {self.missing_author}.\n"
            f"Known authors observed: {len(self.observed_authors)}; "
            f"largest known-author share: {largest}/{self.observed} observed, "
            f"{retained_largest}/{self.retained} retained (counts, not population estimates). "
            f"Per-author cap: {MAX_POSTS_PER_AUTHOR}.\n"
            "This is not a representative sample; recency/search ranking and sample controls affect selection. "
            "Missing authors prevent full concentration checks; account independence and authenticity "
            "are unverified. Account age and bot-quality metrics have not been assessed. "
            "Repeated text is not independent corroboration. "
            "Near-duplicate screening uses conservative word/bigram overlap, protecting explicit numbers, "
            "cashtags and selected polarity/negation terms. It may miss paraphrases or merge distinct claims. "
            "Screening is limited to 8-128 tokens, 4000 characters and 300 canonical candidates; "
            "short/oversized text and index overflow retain exact-ID/text controls only.\n"
            "Coverage is limited to the retrieved recent feed, not a historical archive; "
            "missing matches do not establish an absence of discussion."
        )
