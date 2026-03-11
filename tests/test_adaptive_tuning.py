"""Tests for analytics/adaptive_tuning.py"""
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_portfolio(trade_log=None):
    mock = MagicMock()
    mock.trade_log = trade_log or []
    mock.balance = 500.0
    mock.open_trades = []
    mock.is_dead = False
    return mock


def _make_greeks_trade(exit_reason="theta_burn_tightened", pnl=-5.0, pnl_pct=-0.10):
    return {
        "trade_id": f"test_{id(exit_reason)}_{pnl}",
        "sim_id": "SIM05",
        "exit_reason": exit_reason,
        "realized_pnl_dollars": pnl,
        "realized_pnl_pct": pnl_pct,
        "entry_price": 2.0,
        "exit_price": 2.0 + pnl / 100,
        "exit_time": "2026-03-10T14:00:00-04:00",
    }


class TestAdaptiveTuning:

    @patch("analytics.adaptive_tuning.SimPortfolio")
    def test_evaluate_sufficient_saved(self, mock_sp_class):
        from analytics.adaptive_tuning import evaluate_greeks_effectiveness

        # 7 theta exits, all losing (saved money)
        trades = [_make_greeks_trade(pnl=-3.0) for _ in range(7)]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        result = evaluate_greeks_effectiveness("SIM05", {"theta_burn_enabled": True})
        assert result["theta_burn"]["count"] == 7
        assert result["theta_burn"]["saved_pct"] == 100.0
        assert result["theta_burn"]["status"] == "sufficient"

    @patch("analytics.adaptive_tuning.SimPortfolio")
    def test_evaluate_cut_winners(self, mock_sp_class):
        from analytics.adaptive_tuning import evaluate_greeks_effectiveness

        # 6 theta exits: 4 were profitable (cut winners), 2 losing
        trades = [
            _make_greeks_trade(pnl=5.0) for _ in range(4)
        ] + [
            _make_greeks_trade(pnl=-3.0) for _ in range(2)
        ]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        result = evaluate_greeks_effectiveness("SIM05", {"theta_burn_enabled": True})
        assert result["theta_burn"]["count"] == 6
        assert result["theta_burn"]["cut_winner_count"] == 4
        assert result["theta_burn"]["status"] == "sufficient"

    @patch("analytics.adaptive_tuning.SimPortfolio")
    def test_evaluate_insufficient_data(self, mock_sp_class):
        from analytics.adaptive_tuning import evaluate_greeks_effectiveness

        trades = [_make_greeks_trade() for _ in range(3)]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        result = evaluate_greeks_effectiveness("SIM05", {"theta_burn_enabled": True})
        assert result["theta_burn"]["status"] == "insufficient"

    @patch("analytics.adaptive_tuning._save_overrides")
    @patch("analytics.adaptive_tuning._load_overrides")
    @patch("analytics.adaptive_tuning.evaluate_greeks_effectiveness")
    def test_tuning_tighten(self, mock_eval, mock_load_ov, mock_save_ov):
        from analytics.adaptive_tuning import run_adaptive_tuning

        mock_eval.return_value = {
            "theta_burn": {
                "count": 10,
                "enabled": True,
                "saved_count": 8,
                "cut_winner_count": 2,
                "saved_pct": 80.0,
                "avg_pnl": -5.0,
                "near_stop_count": 1,
                "status": "sufficient",
            },
        }
        mock_load_ov.return_value = {}

        profile = {
            "theta_burn_enabled": True,
            "theta_burn_stop_tighten_pct": 0.50,
            "adaptive_tuning_enabled": True,
        }

        with patch("analytics.adaptive_tuning._log_change"):
            changes = run_adaptive_tuning("SIM05", profile)

        assert len(changes) == 1
        assert changes[0]["trigger"] == "theta_burn"
        assert changes[0]["new"] == 0.45  # tightened from 0.50

    @patch("analytics.adaptive_tuning._save_overrides")
    @patch("analytics.adaptive_tuning._load_overrides")
    @patch("analytics.adaptive_tuning.evaluate_greeks_effectiveness")
    def test_tuning_loosen(self, mock_eval, mock_load_ov, mock_save_ov):
        from analytics.adaptive_tuning import run_adaptive_tuning

        mock_eval.return_value = {
            "theta_burn": {
                "count": 10,
                "enabled": True,
                "saved_count": 3,
                "cut_winner_count": 7,
                "saved_pct": 30.0,
                "avg_pnl": 2.0,
                "near_stop_count": 0,
                "status": "sufficient",
            },
        }
        mock_load_ov.return_value = {}

        profile = {
            "theta_burn_enabled": True,
            "theta_burn_stop_tighten_pct": 0.50,
            "adaptive_tuning_enabled": True,
        }

        with patch("analytics.adaptive_tuning._log_change"):
            changes = run_adaptive_tuning("SIM05", profile)

        assert len(changes) == 1
        assert changes[0]["new"] == 0.55  # loosened from 0.50

    def test_sim00_never_tuned(self):
        from analytics.adaptive_tuning import run_adaptive_tuning

        changes = run_adaptive_tuning("SIM00", {"adaptive_tuning_enabled": True})
        assert changes == []

    @patch("analytics.adaptive_tuning._save_overrides")
    @patch("analytics.adaptive_tuning._load_overrides")
    @patch("analytics.adaptive_tuning.evaluate_greeks_effectiveness")
    def test_hard_floor_ceiling(self, mock_eval, mock_load_ov, mock_save_ov):
        from analytics.adaptive_tuning import run_adaptive_tuning, THETA_BURN_MIN

        mock_eval.return_value = {
            "theta_burn": {
                "count": 10,
                "enabled": True,
                "saved_count": 8,
                "cut_winner_count": 2,
                "saved_pct": 80.0,
                "avg_pnl": -5.0,
                "near_stop_count": 1,
                "status": "sufficient",
            },
        }
        mock_load_ov.return_value = {}

        # Already at minimum
        profile = {
            "theta_burn_enabled": True,
            "theta_burn_stop_tighten_pct": THETA_BURN_MIN,
            "adaptive_tuning_enabled": True,
        }

        with patch("analytics.adaptive_tuning._log_change"):
            changes = run_adaptive_tuning("SIM05", profile)

        # No change because we're already at the floor
        assert len(changes) == 0

    def test_get_effective_threshold_with_override(self):
        from analytics.adaptive_tuning import get_effective_threshold

        with patch("analytics.adaptive_tuning._load_overrides") as mock_ov:
            mock_ov.return_value = {
                "SIM05": {"thresholds": {"theta_burn_stop_tighten_pct": 0.35}}
            }
            val = get_effective_threshold("SIM05", {"theta_burn_stop_tighten_pct": 0.50}, "theta_burn_stop_tighten_pct", 0.50)
            assert val == 0.35

    def test_get_effective_threshold_no_override(self):
        from analytics.adaptive_tuning import get_effective_threshold

        with patch("analytics.adaptive_tuning._load_overrides") as mock_ov:
            mock_ov.return_value = {}
            val = get_effective_threshold("SIM05", {"theta_burn_stop_tighten_pct": 0.45}, "theta_burn_stop_tighten_pct", 0.50)
            assert val == 0.45
