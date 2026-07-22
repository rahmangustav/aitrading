"""Tests for the pure scoring/aggregation logic in sentiment_analyzer.py.

Covers only network-free code paths (label thresholds, symbol mapping,
weighted aggregation). The HTTP-backed source fetchers (_analyze_news,
_analyze_reddit, _analyze_twitter, _analyze_cryptopanic) are out of scope --
they need live feeds/API keys not available in this sandbox.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sentiment_analyzer import SentimentAnalyzer  # noqa: E402


@pytest.fixture
def analyzer():
    # __init__ tries to import vaderSentiment and falls back to
    # self._vader = None on ImportError, so this works with or without
    # the dependency installed.
    return SentimentAnalyzer()


# ----------------------------------------------------------------------
# _score_to_label
# ----------------------------------------------------------------------

class TestScoreToLabel:
    def test_very_bullish_at_and_above_threshold(self):
        assert SentimentAnalyzer._score_to_label(0.3) == "very_bullish"
        assert SentimentAnalyzer._score_to_label(0.9) == "very_bullish"

    def test_bullish_band(self):
        assert SentimentAnalyzer._score_to_label(0.1) == "bullish"
        assert SentimentAnalyzer._score_to_label(0.29) == "bullish"

    def test_neutral_band(self):
        assert SentimentAnalyzer._score_to_label(0.0) == "neutral"
        assert SentimentAnalyzer._score_to_label(0.09) == "neutral"
        assert SentimentAnalyzer._score_to_label(-0.09) == "neutral"

    def test_bearish_band(self):
        assert SentimentAnalyzer._score_to_label(-0.1) == "bearish"
        assert SentimentAnalyzer._score_to_label(-0.29) == "bearish"

    def test_very_bearish_at_and_below_threshold(self):
        assert SentimentAnalyzer._score_to_label(-0.3) == "very_bearish"
        assert SentimentAnalyzer._score_to_label(-0.9) == "very_bearish"


# ----------------------------------------------------------------------
# _get_symbol_names
# ----------------------------------------------------------------------

class TestGetSymbolNames:
    def test_known_symbol_returns_aliases(self):
        names = SentimentAnalyzer._get_symbol_names("BTC")
        assert names == ["Bitcoin", "BTC", "#Bitcoin"]

    def test_case_insensitive(self):
        assert SentimentAnalyzer._get_symbol_names("btc") == SentimentAnalyzer._get_symbol_names("BTC")

    def test_strips_quote_asset(self):
        assert SentimentAnalyzer._get_symbol_names("ETH/USDT") == SentimentAnalyzer._get_symbol_names("ETH")

    def test_unknown_symbol_falls_back_to_itself(self):
        assert SentimentAnalyzer._get_symbol_names("SHIBA") == ["SHIBA"]

    def test_unknown_symbol_with_quote_asset_still_falls_back(self):
        assert SentimentAnalyzer._get_symbol_names("shiba/usdt") == ["SHIBA"]


# ----------------------------------------------------------------------
# _aggregate_scores
# ----------------------------------------------------------------------

class TestAggregateScores:
    def test_no_sources_have_data_returns_neutral_zero(self):
        sources = {
            "news": {"count": 0, "avg_score": 0},
            "cryptopanic": {"error": "CRYPTOPANIC_API_KEY not set", "count": 0, "avg_score": 0},
        }
        agg = SentimentAnalyzer._aggregate_scores(sources)
        assert agg == {
            "score": 0.0,
            "label": "neutral",
            "total_items_analyzed": 0,
            "confidence": "low",
        }

    def test_single_source_returns_its_own_score(self):
        sources = {"news": {"count": 5, "avg_score": 0.5}}
        agg = SentimentAnalyzer._aggregate_scores(sources)
        assert agg["score"] == 0.5
        assert agg["label"] == "very_bullish"
        assert agg["total_items_analyzed"] == 5

    def test_weighted_average_across_sources_with_data(self):
        # news weight 0.30, cryptopanic weight 0.25; reddit/twitter absent.
        sources = {
            "news": {"count": 10, "avg_score": 1.0},
            "cryptopanic": {"count": 10, "avg_score": 0.0},
        }
        agg = SentimentAnalyzer._aggregate_scores(sources)
        expected = (1.0 * 0.30 + 0.0 * 0.25) / (0.30 + 0.25)
        assert agg["score"] == pytest.approx(round(expected, 4))

    def test_zero_count_source_excluded_from_weighting(self):
        # Present in sources dict but with count=0 (e.g. no articles found) --
        # must not silently pull the aggregate toward its avg_score.
        sources = {
            "news": {"count": 0, "avg_score": -1.0},
            "cryptopanic": {"count": 10, "avg_score": 1.0},
        }
        agg = SentimentAnalyzer._aggregate_scores(sources)
        assert agg["score"] == 1.0

    def test_error_source_without_avg_score_key_ignored(self):
        sources = {
            "reddit": {"error": "Reddit API credentials not set", "count": 0, "avg_score": 0},
            "twitter": {"count": 8, "avg_score": 0.2},
        }
        agg = SentimentAnalyzer._aggregate_scores(sources)
        assert agg["score"] == 0.2
        assert agg["total_items_analyzed"] == 8

    def test_confidence_bands(self):
        low = SentimentAnalyzer._aggregate_scores({"news": {"count": 5, "avg_score": 0.0}})
        medium = SentimentAnalyzer._aggregate_scores({"news": {"count": 6, "avg_score": 0.0}})
        high = SentimentAnalyzer._aggregate_scores({"news": {"count": 21, "avg_score": 0.0}})
        assert low["confidence"] == "low"
        assert medium["confidence"] == "medium"
        assert high["confidence"] == "high"

    def test_unknown_source_name_gets_default_weight(self):
        # A source key not in the {news, cryptopanic, reddit, twitter} weight
        # map should still count (default weight 0.1), not be silently dropped.
        sources = {"mystery_source": {"count": 3, "avg_score": 0.5}}
        agg = SentimentAnalyzer._aggregate_scores(sources)
        assert agg["score"] == 0.5
        assert agg["total_items_analyzed"] == 3


# ----------------------------------------------------------------------
# _score_text (VADER unavailable in this sandbox -> must degrade safely)
# ----------------------------------------------------------------------

class TestScoreTextWithoutVader:
    def test_returns_zero_when_vader_not_initialized(self, analyzer):
        analyzer._vader = None
        assert analyzer._score_text("Bitcoin is mooning, huge rally today!") == 0.0


# ----------------------------------------------------------------------
# get_quick_sentiment (exercises _analyze_news's feedparser-missing path)
# ----------------------------------------------------------------------

class TestGetQuickSentiment:
    def test_defaults_when_no_articles_found(self, analyzer, monkeypatch):
        monkeypatch.setattr(
            analyzer, "_analyze_news",
            lambda symbol: {"count": 0, "avg_score": 0, "label": "neutral", "articles": []},
        )
        result = analyzer.get_quick_sentiment("BTC")
        assert result == {
            "symbol": "BTC",
            "score": 0,
            "label": "neutral",
            "articles_analyzed": 0,
        }
