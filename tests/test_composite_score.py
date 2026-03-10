"""tests/test_composite_score.py — unit tests for composite scoring."""
import pytest
from unittest.mock import patch, MagicMock

from analytics.composite_score import (
    compute_composite_score,
    _score_profitability,
    _score_win_rate,
    _score_risk_adjusted,
    _score_consistency,
    _score_drawdown,
    _compute_green_day_pct,
    MIN_TRADES_FOR_RANKING,
)


# ── Pure-function scorer tests ────────────────────────────────────────────────

def test_score_profitability_midpoint():
    assert _score_profitability(0.0) == pytest.approx(50.0)

def test_score_profitability_max():
    assert _score_profitability(0.20) == pytest.approx(100.0)

def test_score_profitability_min():
    assert _score_profitability(-0.20) == pytest.approx(0.0)

def test_score_win_rate_zero():
    assert _score_win_rate(0.40) == pytest.approx(0.0)

def test_score_win_rate_full():
    assert _score_win_rate(0.60) == pytest.approx(100.0)

def test_score_win_rate_clamped():
    assert _score_win_rate(0.00) == 0.0
    assert _score_win_rate(1.00) == 100.0

def test_score_risk_adjusted_pf1():
    assert _score_risk_adjusted(1.0) == pytest.approx(30.0)

def test_score_risk_adjusted_pf2():
    assert _score_risk_adjusted(2.0) == pytest.approx(100.0)

def test_score_risk_adjusted_pf_zero():
    assert _score_risk_adjusted(0.50) == pytest.approx(0.0)

def test_score_consistency_low():
    assert _score_consistency(0.30) == pytest.approx(0.0)

def test_score_consistency_full():
    assert _score_consistency(0.70) == pytest.approx(100.0)

def test_score_drawdown_zero():
    assert _score_drawdown(0.0) == pytest.approx(100.0)

def test_score_drawdown_full():
    assert _score_drawdown(0.50) == pytest.approx(0.0)

def test_green_day_pct_basic():
    log = [
        {"realized_pnl_dollars": 10.0, "exit_time": "2026-03-01T12:00:00"},
        {"realized_pnl_dollars": -5.0, "exit_time": "2026-03-02T12:00:00"},
        {"realized_pnl_dollars":  5.0, "exit_time": "2026-03-03T12:00:00"},
    ]
    assert _compute_green_day_pct(log) == pytest.approx(2 / 3)

def test_green_day_pct_same_day():
    log = [
        {"realized_pnl_dollars": -10.0, "exit_time": "2026-03-01T10:00:00"},
        {"realized_pnl_dollars":   3.0, "exit_time": "2026-03-01T14:00:00"},
    ]
    # Net day pnl = -7 → 0 green days out of 1
    assert _compute_green_day_pct(log) == 0.0

def test_green_day_pct_empty():
    assert _compute_green_day_pct([]) == 0.0


# ── Integration tests using mock SimPortfolio ─────────────────────────────────

def _make_sim(trades: list, balance: float = 500.0, peak_balance: float = 500.0):
    sim = MagicMock()
    sim.trade_log = trades
    sim.balance = balance
    sim.peak_balance = peak_balance
    return sim


def _make_trade(pnl: float, day: str = "2026-03-01", direction: str = "BULLISH") -> dict:
    return {
        "realized_pnl_dollars": pnl,
        "exit_time": f"{day}T14:00:00",
        "direction": direction,
    }


@patch("simulation.sim_portfolio.SimPortfolio")
def test_compute_strong_sim(MockSim):
    """High PnL, high WR, low DD → expect A / A+."""
    trades = (
        [_make_trade(25.0, f"2026-03-{i:02d}") for i in range(1, 12)]   # 11 wins
        + [_make_trade(-5.0, f"2026-03-{i:02d}") for i in range(12, 14)]  # 2 losses
    )
    MockSim.return_value = _make_sim(trades, balance=570.0, peak_balance=580.0)
    profile = {"balance_start": 500, "signal_mode": "TREND_PULLBACK"}

    result = compute_composite_score("SIM_TEST", profile)

    assert result["unranked"] is False
    assert result["composite_score"] >= 70, f"Expected A-range, got {result['composite_score']}"
    assert result["grade"] in ("A+", "A", "B")


@patch("simulation.sim_portfolio.SimPortfolio")
def test_compute_struggling_sim(MockSim):
    """Negative PnL, low WR, large DD → expect D / F."""
    trades = (
        [_make_trade(-30.0, f"2026-03-{i:02d}") for i in range(1, 9)]   # 8 losses
        + [_make_trade(5.0,  f"2026-03-{i:02d}") for i in range(9, 12)]  # 3 wins
    )
    MockSim.return_value = _make_sim(trades, balance=275.0, peak_balance=500.0)
    profile = {"balance_start": 500, "signal_mode": "MEAN_REVERSION"}

    result = compute_composite_score("SIM_TEST", profile)

    assert result["unranked"] is False
    assert result["composite_score"] < 55, f"Expected low score, got {result['composite_score']}"
    assert result["grade"] in ("D", "F", "C")


@patch("simulation.sim_portfolio.SimPortfolio")
def test_compute_unranked_insufficient_trades(MockSim):
    """Fewer than MIN_TRADES_FOR_RANKING → UNRANKED."""
    trades = [_make_trade(10.0, "2026-03-01") for _ in range(MIN_TRADES_FOR_RANKING - 1)]
    MockSim.return_value = _make_sim(trades)
    profile = {"balance_start": 500}

    result = compute_composite_score("SIM_TEST", profile)

    assert result["unranked"] is True
    assert "insufficient_trades" in result["unranked_reason"]
    assert result["composite_score"] is None


@patch("simulation.sim_portfolio.SimPortfolio")
def test_compute_zero_trades(MockSim):
    """Zero trades → UNRANKED."""
    MockSim.return_value = _make_sim([])
    profile = {"balance_start": 500}

    result = compute_composite_score("SIM_TEST", profile)

    assert result["unranked"] is True
    assert result["total_trades"] == 0


@patch("simulation.sim_portfolio.SimPortfolio")
def test_components_structure(MockSim):
    """Result dict has all required fields and components."""
    trades = [_make_trade(10.0, f"2026-03-{i:02d}") for i in range(1, 15)]
    MockSim.return_value = _make_sim(trades)
    profile = {"balance_start": 500}

    result = compute_composite_score("SIM_TEST", profile)

    assert result["unranked"] is False
    for key in ("profitability", "win_rate", "risk_adjusted", "consistency", "drawdown"):
        assert key in result["components"]
        comp = result["components"][key]
        assert "raw" in comp and "score" in comp and "weight" in comp
        assert 0.0 <= comp["score"] <= 100.0
