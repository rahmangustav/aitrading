"""Tests for dca_calculator.py -- DCA scheduling and simulation math.

Pure calculation logic with no network/exchange calls, previously without
any test coverage at all (no test file existed under trading-research).
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dca_calculator import (
    calculate_dca_schedule,
    calculate_lump_sum_comparison,
    simulate_dca_returns,
)


class TestCalculateDcaSchedule:
    def test_weekly_schedule_splits_amount_evenly(self):
        start = datetime(2026, 1, 1)
        schedule, num_purchases, amount_per_purchase = calculate_dca_schedule(
            700.0, "weekly", 28, start_date=start
        )
        assert num_purchases == 4
        assert amount_per_purchase == pytest.approx(175.0)
        assert len(schedule) == 4
        assert schedule[0]["date"] == "2026-01-01"
        assert schedule[-1]["cumulative"] == pytest.approx(700.0)

    def test_dates_increment_by_interval(self):
        start = datetime(2026, 1, 1)
        schedule, _, _ = calculate_dca_schedule(300.0, "weekly", 21, start_date=start)
        assert [entry["date"] for entry in schedule] == [
            "2026-01-01",
            "2026-01-08",
            "2026-01-15",
        ]

    def test_frequency_is_case_insensitive(self):
        schedule, num_purchases, _ = calculate_dca_schedule(
            100.0, "WEEKLY", 14, start_date=datetime(2026, 1, 1)
        )
        assert num_purchases == 2

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError):
            calculate_dca_schedule(100.0, "yearly", 30, start_date=datetime(2026, 1, 1))

    def test_duration_shorter_than_interval_forces_one_purchase(self):
        schedule, num_purchases, amount_per_purchase = calculate_dca_schedule(
            500.0, "monthly", 3, start_date=datetime(2026, 1, 1)
        )
        assert num_purchases == 1
        assert amount_per_purchase == pytest.approx(500.0)
        assert len(schedule) == 1

    def test_cumulative_tracks_running_total(self):
        schedule, _, amount_per_purchase = calculate_dca_schedule(
            400.0, "daily", 4, start_date=datetime(2026, 1, 1)
        )
        cumulative = [entry["cumulative"] for entry in schedule]
        assert cumulative == pytest.approx(
            [amount_per_purchase * i for i in range(1, 5)]
        )


class TestSimulateDcaReturns:
    def test_matches_exact_dates(self):
        schedule = [
            {"date": "2026-01-01", "amount": 100.0},
            {"date": "2026-01-08", "amount": 100.0},
        ]
        price_history = {"2026-01-01": 50.0, "2026-01-08": 100.0}
        purchases, total_invested, total_units = simulate_dca_returns(
            schedule, price_history
        )
        assert total_invested == pytest.approx(200.0)
        assert total_units == pytest.approx(2.0 + 1.0)
        assert purchases[0]["avg_price"] == pytest.approx(50.0)
        # Running average after both buys: 200 invested / 3 units.
        assert purchases[-1]["avg_price"] == pytest.approx(200.0 / 3.0)

    def test_falls_back_to_next_available_date(self):
        schedule = [{"date": "2026-01-02", "amount": 100.0}]
        # No price for the 2nd; the next available date (the 5th) is used.
        price_history = {"2026-01-01": 40.0, "2026-01-05": 80.0}
        purchases, total_invested, total_units = simulate_dca_returns(
            schedule, price_history
        )
        assert len(purchases) == 1
        assert purchases[0]["price"] == pytest.approx(80.0)
        assert total_invested == pytest.approx(100.0)
        assert total_units == pytest.approx(1.25)

    def test_skips_entry_with_no_price_available_at_or_after_date(self):
        schedule = [
            {"date": "2026-02-01", "amount": 100.0},
            {"date": "2026-01-01", "amount": 50.0},
        ]
        # Only a price before the first entry's date exists, and none at/after.
        price_history = {"2026-01-01": 40.0}
        purchases, total_invested, total_units = simulate_dca_returns(
            schedule, price_history
        )
        # First entry (2026-02-01) finds no date >= itself -> skipped.
        # Second entry (2026-01-01) matches exactly -> kept.
        assert len(purchases) == 1
        assert purchases[0]["date"] == "2026-01-01"
        assert total_invested == pytest.approx(50.0)
        assert total_units == pytest.approx(1.25)

    def test_empty_schedule_returns_zero_totals(self):
        purchases, total_invested, total_units = simulate_dca_returns([], {})
        assert purchases == []
        assert total_invested == 0
        assert total_units == 0


class TestCalculateLumpSumComparison:
    def test_positive_return_when_price_rises(self):
        result = calculate_lump_sum_comparison(1000.0, 100.0, 150.0)
        assert result["units"] == pytest.approx(10.0)
        assert result["final_value"] == pytest.approx(1500.0)
        assert result["return_pct"] == pytest.approx(50.0)

    def test_negative_return_when_price_falls(self):
        result = calculate_lump_sum_comparison(1000.0, 100.0, 60.0)
        assert result["final_value"] == pytest.approx(600.0)
        assert result["return_pct"] == pytest.approx(-40.0)

    def test_flat_price_gives_zero_return(self):
        result = calculate_lump_sum_comparison(500.0, 20.0, 20.0)
        assert result["return_pct"] == pytest.approx(0.0)
