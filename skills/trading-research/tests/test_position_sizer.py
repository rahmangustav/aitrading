"""Tests for position_sizer.py -- risk-based position sizing math.

SKILL.md tells every research session to "Use position_sizer.py for every
trade," so this is load-bearing risk-management logic even though the repo
is currently gated off live trading. Nothing here previously had coverage.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from position_sizer import (
    calculate_position_size,
    calculate_take_profit_levels,
    calculate_ladder_strategy,
)


class TestCalculatePositionSize:
    def test_long_position_sizing(self):
        # Risking 1% of a 10,000 balance = 100 USD. Stop is 5,000 away from
        # entry, so units = 100 / 5000 = 0.02 BTC.
        result = calculate_position_size(10000.0, 1.0, 100000.0, 95000.0)
        assert result["risk_amount"] == pytest.approx(100.0)
        assert result["position_size_units"] == pytest.approx(0.02)
        assert result["position_value"] == pytest.approx(2000.0)
        assert result["position_percent"] == pytest.approx(20.0)
        assert result["stop_loss_percent"] == pytest.approx(5.0)

    def test_short_position_uses_absolute_price_difference(self):
        # Stop above entry (short): sizing math must use |entry - stop|,
        # not a signed difference that would flip the risk amount negative.
        result = calculate_position_size(10000.0, 1.0, 95000.0, 100000.0)
        assert result["position_size_units"] == pytest.approx(0.02)
        assert result["risk_amount"] > 0
        assert result["position_value"] > 0

    def test_rejects_non_positive_entry_price(self):
        with pytest.raises(ValueError):
            calculate_position_size(10000.0, 1.0, 0.0, 95000.0)

    def test_rejects_non_positive_stop_loss_price(self):
        with pytest.raises(ValueError):
            calculate_position_size(10000.0, 1.0, 100000.0, -1.0)

    def test_rejects_non_positive_account_balance(self):
        with pytest.raises(ValueError):
            calculate_position_size(0.0, 1.0, 100000.0, 95000.0)

    def test_rejects_risk_percent_out_of_range(self):
        with pytest.raises(ValueError):
            calculate_position_size(10000.0, 0.0, 100000.0, 95000.0)
        with pytest.raises(ValueError):
            calculate_position_size(10000.0, 101.0, 100000.0, 95000.0)

    def test_rejects_entry_equal_to_stop_loss(self):
        with pytest.raises(ValueError):
            calculate_position_size(10000.0, 1.0, 100000.0, 100000.0)

    def test_risk_percent_boundary_100_is_allowed(self):
        result = calculate_position_size(10000.0, 100.0, 100000.0, 95000.0)
        assert result["risk_amount"] == pytest.approx(10000.0)


class TestCalculateTakeProfitLevels:
    def test_long_direction_detected_when_stop_below_entry(self):
        levels, is_long = calculate_take_profit_levels(100000.0, 95000.0, [1, 2, 3])
        assert is_long is True
        assert [lvl["ratio"] for lvl in levels] == [1, 2, 3]
        # 1R above entry = entry + (5000 * 1) = 105000
        assert levels[0]["price"] == pytest.approx(105000.0)
        assert levels[0]["profit_percent"] == pytest.approx(5.0)

    def test_short_direction_detected_when_stop_above_entry(self):
        levels, is_long = calculate_take_profit_levels(95000.0, 100000.0, [1, 2])
        assert is_long is False
        # 1R below entry = entry - (5000 * 1) = 90000
        assert levels[0]["price"] == pytest.approx(90000.0)
        assert levels[0]["profit_percent"] < 0
        assert levels[1]["price"] == pytest.approx(85000.0)

    def test_default_risk_reward_ratios_are_1_2_3(self):
        levels, _ = calculate_take_profit_levels(100000.0, 95000.0)
        assert [lvl["ratio"] for lvl in levels] == [1, 2, 3]

    def test_higher_ratio_moves_price_further_from_entry(self):
        levels, _ = calculate_take_profit_levels(100000.0, 95000.0, [1, 2, 3])
        prices = [lvl["price"] for lvl in levels]
        assert prices == sorted(prices)


class TestCalculateLadderStrategy:
    def test_splits_position_evenly_across_levels(self):
        levels = calculate_ladder_strategy(1.0, num_levels=4)
        assert len(levels) == 4
        assert sum(lvl["size"] for lvl in levels) == pytest.approx(1.0)
        assert levels[0]["level"] == 1
        assert levels[-1]["level"] == 4

    def test_cumulative_size_reaches_total_at_last_level(self):
        levels = calculate_ladder_strategy(0.5, num_levels=3)
        assert levels[-1]["cumulative"] == pytest.approx(0.5)

    def test_percent_per_level_sums_to_100(self):
        levels = calculate_ladder_strategy(2.0, num_levels=5)
        assert sum(lvl["percent"] for lvl in levels) == pytest.approx(100.0)

    def test_default_num_levels_is_3(self):
        levels = calculate_ladder_strategy(3.0)
        assert len(levels) == 3

    def test_zero_num_levels_raises_instead_of_zerodivisionerror(self):
        with pytest.raises(ValueError, match="levels"):
            calculate_ladder_strategy(1.0, num_levels=0)

    def test_negative_num_levels_raises(self):
        with pytest.raises(ValueError, match="levels"):
            calculate_ladder_strategy(1.0, num_levels=-2)

    def test_zero_position_size_raises_instead_of_zerodivisionerror(self):
        with pytest.raises(ValueError, match="Position size"):
            calculate_ladder_strategy(0.0, num_levels=3)
