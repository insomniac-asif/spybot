# Sim Trade Optimizer — Discovery Inventory

## _trade_grade Location and Signature

Defined in FOUR places (all identical logic, copies for import convenience):

| File | Line | Signature |
|------|------|-----------|
| `simulation/sim_engine.py` | 418 | `def _trade_grade(tr) -> float | None` |
| `simulation/sim_entry_helpers.py` | 26 | `def _trade_grade(tr) -> "float | None"` |
| `simulation/sim_exit_helpers.py` | 14 | `def _trade_grade(tr) -> "float | None"` |
| `simulation/sim_live_helpers.py` | 26 | `def _trade_grade(tr) -> float | None` |

**Logic**: Checks keys `("edge_prob", "prediction_confidence", "confidence", "ml_probability")` on the
trade dict, collects numeric values, returns `max(candidates)` or `None`.

**Canonical import**: `from simulation.sim_engine import _trade_grade`

---

## Complete Closed Trade Dict Schema

All keys found in closed trades (SIM01–SIM03 sample):

```
account_phase, balance_after_trade, cash_adjusted, confidence, contract_type,
conviction_score, delta_at_entry, direction, direction_prob, dte_bucket, edge_prob,
entry_context, entry_notional, entry_price, entry_price_source, entry_time,
exit_context, exit_price, exit_price_source, exit_quote_model, exit_reason,
exit_time, expiry, follow_through, gamma_at_entry, hold_max_seconds,
hold_min_seconds, horizon, impulse, iv_at_entry, mae, mae_pct, mfe, mfe_pct,
ml_probability, option_symbol, otm_pct, peak_balance_after_trade, peak_price,
predicted_direction, prediction_confidence, qty, realized_pnl_dollars,
realized_pnl_pct, regime, regime_at_entry, setup, signal_mode, sim_id,
spread_guard_bypassed, strategy_family, strike, style, symbol, theta_at_entry,
time_in_trade_seconds, time_of_day_bucket, trade_id, trailing_stop_activated,
trailing_stop_high, vega_at_entry, volatility
```

**Closed trade indicator**: `exit_price is not None` OR `realized_pnl_dollars is not None`

---

## Analytics / Scoring Functions

| File | Function | Purpose |
|------|----------|---------|
| `analytics/composite_score.py` | `compute_composite_score(sim_id, profile)` | Multi-factor sim-level composite score |
| `analytics/composite_score.py` | `_score_profitability`, `_score_win_rate`, etc. | Sub-scorers |
| `analytics/composite_score.py` | `_letter_grade(score)` | Score → letter grade |
| `analytics/grader.py` | `grade_trade(trade)` | Grades a single prediction trade (win/loss/neutral) |
| `analytics/grader.py` | `update_edge_stats(graded)` | Updates conviction_expectancy.csv |
| `analytics/strategy_performance.py` | `get_score(strategy, regime, time_bucket)` | Per-strategy-regime-bucket perf lookup |
| `simulation/sim_opportunity_ranker.py` | `_historical_score`, `_regime_score`, `_risk_reward_score` | Sub-scores for opportunity ranking |
| `simulation/sim_evaluation.py` | `evaluate_sims(start_dt, end_dt, ...)` | Full batch evaluation |

---

## Fields Available for Analysis

| Dimension | Field(s) | Notes |
|-----------|----------|-------|
| MAE | `mae_pct` (float, negative = adverse) | Populated by `update_open_trade_excursion` |
| MFE | `mfe_pct` (float, positive = favorable) | Populated by `update_open_trade_excursion` |
| ML Confidence | `edge_prob`, `prediction_confidence`, `confidence`, `ml_probability` | Use `_trade_grade()` to get best |
| Regime | `regime_at_entry` | Values: TREND, RANGE, VOLATILE, SIDEWAYS, NO_DATA |
| Time bucket | `time_of_day_bucket` | Values: MORNING, MIDDAY, AFTERNOON, CLOSE, PREMARKET, etc. |
| Direction | `direction` | BULLISH or BEARISH |
| Signal mode | `signal_mode` | TREND_PULLBACK, MEAN_REVERSION, BREAKOUT, etc. |
| PnL | `realized_pnl_dollars`, `realized_pnl_pct` | Positive = win |
| Entry price | `entry_price` | Option contract price |
| Hold time | `time_in_trade_seconds` | Actual hold duration |
| Hold limits | `hold_min_seconds`, `hold_max_seconds` | From profile, stored on trade |
| Spread | `spread_guard_bypassed` | bool — no raw spread value stored |
| OTM | `otm_pct` | OTM percentage at entry |
| Conviction | `conviction_score` | Float |

---

## Key Observations

- `spread_at_entry` is NOT stored on the trade dict; `spread_guard_bypassed` (bool) is.
- MAE is stored as a negative fraction of entry price; MFE as positive fraction.
- `_trade_grade()` returns the best available ML confidence from the four candidate keys.
- sim_config.yaml has `regime_filter` (list or string) per sim — already used in `run_sim_entries`.
- Entry runner is in `simulation/sim_entry_runner.py`; the quality gate should be inserted after line ~517 (after regime_filter check, before `execution_mode == "live"` branch).
