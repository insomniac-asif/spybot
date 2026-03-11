"""Tests for analytics/trade_journal.py"""
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_portfolio(trade_log=None, open_trades=None, balance=500.0, is_dead=False):
    """Create a mock SimPortfolio."""
    mock = MagicMock()
    mock.trade_log = trade_log or []
    mock.open_trades = open_trades or []
    mock.balance = balance
    mock.peak_balance = balance
    mock.is_dead = is_dead
    return mock


def _make_trade(sim_id="SIM01", pnl=10.0, pnl_pct=0.1, exit_reason="profit_target",
                direction="BULLISH", entry_time="2026-03-10T10:00:00-04:00",
                exit_time="2026-03-10T10:30:00-04:00", **kwargs):
    trade = {
        "trade_id": f"{sim_id}__{id(kwargs)}",
        "sim_id": sim_id,
        "direction": direction,
        "entry_price": 2.0,
        "exit_price": 2.0 + pnl / 100,
        "realized_pnl_dollars": pnl,
        "realized_pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "regime_at_entry": "TREND",
        "signal_mode": "TREND_PULLBACK",
        "strike": 595,
        "dte_bucket": "1",
        "contract_type": "CALL",
    }
    trade.update(kwargs)
    return trade


class TestTradeJournal:
    """Tests for trade journal generation."""

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_journal_with_trades(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import generate_daily_journal

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        trades = [
            _make_trade(pnl=10.0, pnl_pct=0.05),
            _make_trade(pnl=-5.0, pnl_pct=-0.025, exit_reason="stop_loss"),
        ]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        journal = generate_daily_journal("2026-03-10")
        assert "Trade Journal" in journal
        assert "SIM01" in journal
        assert "Entries" in journal
        assert "Exits" in journal

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_journal_zero_trades(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import generate_daily_journal

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        mock_sp = _mock_portfolio(trade_log=[])
        mock_sp_class.return_value = mock_sp

        journal = generate_daily_journal("2026-03-10")
        assert "No entries today" in journal
        assert "No exits today" in journal

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_journal_greeks_exits(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import generate_daily_journal

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        trades = [
            _make_trade(pnl=-3.0, pnl_pct=-0.15, exit_reason="theta_burn_0dte"),
            _make_trade(pnl=-2.0, pnl_pct=-0.10, exit_reason="iv_crush_exit"),
            _make_trade(pnl=5.0, pnl_pct=0.05, exit_reason="profit_target"),
        ]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        journal = generate_daily_journal("2026-03-10")
        assert "Greeks Exit Detail" in journal
        assert "Theta Burn" in journal
        assert "IV Crush" in journal

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_journal_no_greeks_exits(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import generate_daily_journal

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        trades = [_make_trade(pnl=5.0, exit_reason="profit_target")]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        journal = generate_daily_journal("2026-03-10")
        assert "Greeks Exit Detail" not in journal

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_build_journal_summary(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import build_journal_summary

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        trades = [
            _make_trade(pnl=10.0, pnl_pct=0.05),
            _make_trade(pnl=-5.0, pnl_pct=-0.025, exit_reason="theta_burn"),
        ]
        mock_sp = _mock_portfolio(trade_log=trades)
        mock_sp_class.return_value = mock_sp

        summary = build_journal_summary("2026-03-10")
        assert summary["exits"] == 2
        assert summary["net_pnl"] == 5.0
        assert "theta" in summary["greeks_exits"]

    @patch("analytics.trade_journal._load_profiles")
    @patch("analytics.trade_journal.SimPortfolio")
    def test_weekly_digest(self, mock_sp_class, mock_profiles):
        from analytics.trade_journal import generate_weekly_digest

        mock_profiles.return_value = {"SIM01": {"balance_start": 500}}
        mock_sp = _mock_portfolio(trade_log=[])
        mock_sp_class.return_value = mock_sp

        digest = generate_weekly_digest("2026-03-10")
        assert "Weekly Trade Digest" in digest
        assert "Daily Breakdown" in digest
