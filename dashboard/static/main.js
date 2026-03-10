/* ═══════════════════════════════════════════════════════ SpyBot Dashboard */
'use strict';

// ─────────────────────────────────────────────── STRATEGY DESCRIPTIONS
const STRATEGY_DESCRIPTIONS = {
  TREND_PULLBACK: `Trades in the direction of the dominant short-term trend by entering exactly when price pulls back to the EMA9 dynamic support/resistance line. The signal fires when EMA9 is above EMA20 (uptrend) and the current close is within ±0.1% of EMA9 — meaning price has retraced to the fast moving average but the trend structure is still intact. The edge is that EMA9 reliably absorbs selling pressure in trending conditions and launches the next leg higher; entry right at the line minimises risk relative to chasing the move. In downtrends the same logic applies in reverse: EMA9 < EMA20 and price bouncing up to the EMA9 level from below. Best in trending, directional sessions. Be cautious when EMA9 and EMA20 are converging — that is a sign of trend exhaustion and the pullback may turn into a full reversal.`,

  MEAN_REVERSION: `A contrarian strategy that fades short-term price extremes. Entry is triggered when RSI14 drops below 30 AND price is below VWAP (buy call), or RSI14 exceeds 70 AND price is above VWAP (buy put). Both conditions must be true simultaneously — RSI alone is too noisy intraday, and VWAP displacement alone can persist in trending markets. The combination identifies moments where price has deviated significantly from both its statistical momentum baseline and the intraday volume-weighted fair value. The edge comes from the mathematical tendency of RSI to revert toward 50 and price toward VWAP after extreme readings. Highest win rate in range-bound, choppy sessions (e.g. consolidation days, low-news afternoons). In strong trending environments, RSI can stay overbought/oversold for 30+ minutes, so this signal will fire early and require a wide stop.`,

  BREAKOUT: `A momentum breakout strategy that enters when price closes beyond the highest high or lowest low of the previous 20 bars. A close above the 20-bar rolling high signals a BULLISH breakout (buy call); a close below the 20-bar low signals a BEARISH breakdown (buy put). The 20-bar window captures roughly 20 minutes of price action, so this is an intraday momentum signal. The edge: a genuine close beyond the recent range signals that supply or demand has been decisively overcome and momentum should carry the move further. The biggest risk is the false breakout — price pierces the level momentarily then reverses, trapping late buyers. This is especially common in the first 30 minutes of the session. Works best on volume-confirmed moves (large candles, not just a thin poke above the high).`,

  SWING_TREND: `A trend-following signal that requires both price and the EMA20 to be moving in the same direction simultaneously. Bullish: EMA20 is rising (current value > value 10 bars ago) AND price closes above EMA20. Bearish: EMA20 is falling AND price closes below EMA20. This is intentionally less precise than TREND_PULLBACK — it fires whenever the market is in a confirmed swing trend, not just at pullback entry points. This means more trades but later entries (you're already above EMA20, not touching it). Best used to capture extended trending days when the market makes a clear directional move from open to close. The 10-bar slope check prevents entries into exhausted moves where EMA20 has already flattened out.`,

  OPPORTUNITY: `An adaptive multi-factor conviction model that combines momentum, VWAP positioning, RSI, and market structure into a numeric conviction score. Based on the score and current volatility state, it dynamically selects the trade horizon: DAYTRADE (conviction ≤ 3 or high volatility), SWING (moderate conviction, normal vol), or WEEKLY (conviction ≥ 6 or low volatility). Each horizon maps to different DTE ranges and prediction windows. After 50 completed trades, an ML prediction filter activates: if the bot's own price forecast disagrees with the signal direction at ≥60% confidence, the trade is blocked. This self-filtering means early trades are unfiltered (maximum learning), and later trades are increasingly selective. The most adaptive signal in the system — it changes what it looks for based on market conditions and its own historical performance.`,

  ORB_BREAKOUT: `Opening Range Breakout using the first 30-minute high/low as the reference range. A close above the opening range high triggers a BULLISH entry (calls); a close below the opening range low triggers a BEARISH entry (puts). This strategy is grounded in the institutional behavior at the open: the first 30 minutes often defines the day's directional bias, and a clean break of that range tends to follow through for several percent. Two configurable filters sharpen entries: (1) vol_z threshold — only trade when current volume is statistically elevated versus recent average, confirming institutional participation rather than thin-volume noise; (2) require_trend_bias — only take bullish ORB breaks when EMA9 > EMA20 (trend confirmation), reducing counter-trend breakout failures. Best on high-conviction open days (economic data, earnings reactions) and avoid on choppy, mean-reverting sessions.`,

  VWAP_REVERSION: `Fades overextended VWAP deviations, optionally using machine-learning feature snapshots when features_enabled is true. In basic mode, fires when RSI plus raw VWAP distance exceed thresholds. With features enabled, uses the vwap_z score (number of standard deviations from intraday VWAP) plus RSI: bullish when VWAP-Z is deeply negative and RSI is oversold, bearish when VWAP-Z is deeply positive and RSI is overbought. The VWAP is the institutional benchmark for intraday value; large deviations are unsustainable because market makers and algorithms actively work price back toward it throughout the session. Edge deteriorates in strong trending sessions where VWAP itself is sloping aggressively — in those cases VWAP-Z can stay elevated for the entire afternoon.`,

  ZSCORE_BOUNCE: `Pure statistical mean reversion using a rolling z-score of close prices. Calculates how many standard deviations the current price is from its recent rolling mean, using feature_snapshot data when available. A z-score below -2.0 triggers a BULLISH entry (statistically cheap); above +2.0 triggers a BEARISH entry (statistically expensive). RSI is used as a secondary confirmation filter. Blocked if EMA9/EMA20 spread is too large, since strong trends can sustain extreme z-scores indefinitely. The edge is purely quantitative — no subjectivity required. Works best in oscillating, low-trend markets. In trending sessions this signal fires against the prevailing momentum and suffers from being persistently early; sizing down and widening stops is essential on trend days.`,

  FAILED_BREAKOUT_REVERSAL: `Identifies false breakout traps and trades the reversal. Uses a 20-bar reference window to establish the range. A bearish failed breakout fires when: the previous bar's high exceeded the 20-bar reference high (attempted breakout) but its close fell back below that level (rejection), AND the current bar confirms downward continuation. The bullish version mirrors this for false breakdowns. A four-factor structure score captures quality: (1) wick rejection ≥50% of bar range, (2) break magnitude > 0.1%, (3) current volume above recent average, (4) strong current-bar confirmation close. The edge: trapped bulls at the breakout level become forced sellers as their stops trigger, fueling the reversal. This is one of the most reliable patterns because it exploits predictable stop-loss placement by retail traders who chased the breakout.`,

  VWAP_CONTINUATION: `A trend-following entry that combines EMA alignment with a VWAP bounce. Requires: EMA9 > EMA20 (uptrend), current price just above VWAP (within a configurable 0.5% band by default), AND price touched VWAP or came within 0.2% of it in the last 5 bars. The three-condition requirement makes this more selective than raw trend signals: the EMA alignment confirms the macro trend, the proximity to VWAP ensures you're not entering too extended, and the recent VWAP touch confirms that VWAP is acting as active support (not just a distant reference). The edge: every time price bounces off VWAP with EMA alignment, the path of least resistance is continuation. Entries right off VWAP also provide a natural, tight stop — a close back through VWAP would invalidate the thesis.`,

  OPENING_DRIVE: `Captures sustained directional momentum from the open. Measures the percentage move from the first bar's open price over a 15-bar window. Fires BULLISH when that move exceeds 0.3% (configurable) upward. An exhaustion filter blocks the signal if the last three bars are moving counter to the drive direction (three consecutive lower closes during an upward drive), since that indicates the momentum is fading. Bonus structure score if a mid-session pullback occurred (bars 4-12 dipped and then price resumed) — this pullback-and-resume pattern is the hallmark of a genuine Trend Day rather than an early spike that quickly reverses. Best for capturing the classic strong open → brief consolidation → continuation pattern that occurs on high-conviction directional days.`,

  AFTERNOON_BREAKOUT: `Volatility compression-to-expansion breakout. Uses the last 12 bars to establish a compressed range and calculate the average bar range during that compression. Fires BULLISH when current bar: (1) closes above the compression period high AND (2) has a bar range ≥1.5× the average compression bar range, indicating genuine expansion rather than a slow drift above the range. ATR comparison provides a secondary expansion confirmation. This strategy is specifically designed for afternoon consolidation breakouts — the common pattern where the market chops tightly for 30-60 minutes midday then breaks out sharply. The compression period acts as a coiled spring. The expansion ratio filter (1.5× default) prevents false signals where price barely creeps above the range with no real momentum.`,

  TREND_RECLAIM: `A three-bar EMA9 cross reclaim pattern. Specifically looks for: bar[-3] was on the wrong side of EMA9 (failed), bar[-2] crossed back over and reclaimed EMA9, bar[-1] (current) is holding the reclaim. Additionally requires EMA9 to be on the correct side of EMA20 — this ensures you're reclaiming into a trend, not just bouncing in a range. Optional VWAP alignment filter adds a third confirmation layer. The edge: a clean EMA9 reclaim after a brief failure is a high-probability continuation signal because it traps both the stops from the initial EMA9 break AND the momentum from the new cross. The stop-hunting from the temporary dip below EMA9 clears out weak hands before the real move. Works best when the EMA9-EMA20 spread is meaningful (confirmed trend) rather than flat (choppy).`,

  EXTREME_EXTENSION_FADE: `Requires features_enabled: true. Fades price at statistically extreme VWAP deviations using a two-filter approach. The primary filter is VWAP-Z score (standard deviations from intraday VWAP): bearish fade fires when VWAP-Z ≥ 2.5, bullish fade fires when VWAP-Z ≤ -2.5. The secondary filter is RSI: bearish requires RSI ≥ 76, bullish requires RSI ≤ 24. Both must be true. A trend environment blocker prevents entries when EMA spread exceeds 0.5% — in strongly trending markets even extreme VWAP-Z readings can persist. This is the most statistically selective fade signal in the system. The edge: a VWAP-Z of 2.5 combined with overbought RSI represents a confluence of two independent statistical extremes simultaneously — the probability of mean reversion at that point is significantly higher than either filter alone.`,

  FVG_4H: `Smart Money Concept — Fair Value Gap on 4-hour bars. Aggregates 1-minute candles into 240-minute bars and scans for price imbalances (FVGs). A bullish FVG exists when bar[i].high < bar[i+2].low — a gap in price where bar i and bar i+2 have no overlapping range, meaning bar[i+1] moved so fast it left an unfilled imbalance. Entry fires when current price re-enters (retests) an unfilled FVG zone. The 4-hour timeframe gives institutional context: large players move size quickly, leaving behind these gaps. Institutional algorithms and smart money tend to return to fill these imbalances, making them high-probability entry zones. The zone is tracked until filled; once price passes through the entire zone, it's discarded. 60-bar cooldown prevents re-entering the same zone repeatedly. Best during trending sessions when institutional order flow is directional.`,

  FVG_5M: `Fair Value Gap on 5-minute aggregated bars — a shorter-timeframe, more tactical version of FVG_4H. 1-minute bars are aggregated into 5-minute bars and the same three-candle gap logic applies. Entry fires when current price revisits an unfilled 5-minute FVG. An important filter: only FVG zones formed on bars with volume ≥ 1× the 20-bar volume average are considered, ensuring the gap was created with meaningful participation rather than thin-air movement during dead periods. 10-bar cooldown allows more frequent intraday entries. Because 5-minute FVGs are more numerous and less structurally significant than 4-hour ones, this signal fires more often and works best in volatile, liquid sessions where price action is directional. In choppy conditions, 5-minute FVGs fill and reverse frequently, so win rate drops.`,

  LIQUIDITY_SWEEP: `Smart Money Concept — Stop Hunt Reversal. Scans the last 60 bars for structural swing highs and lows (5-bar left/right look-around). A buyside sweep (BEARISH signal) fires when price spikes above a swing high, then closes back below it within 3 bars on a full-bodied candle (body ≥ 50% of bar range). This represents stop-hunting of retail long positions sitting above the swing high; the reversal entry fades the institutional players who just collected that liquidity. The minimum break threshold (0.01% above the swing level) ensures the sweep was genuine and not just noise. A sellside sweep (BULLISH signal) is the mirror: spike below swing low, recovery close above. 15-bar cooldown. The edge: liquidity sweeps are one of the most reliable reversal signals in SMC theory because they represent documented institutional accumulation/distribution activity — the smart money taking the other side of retail stops.`,

  FVG_SWEEP_COMBO: `A high-conviction confluence signal requiring two independent SMC conditions to align simultaneously. First, a liquidity sweep must be detected (see LIQUIDITY_SWEEP). Then, within that same directional bias, price must be inside an unfilled 5-minute FVG zone. Both the sweep and the FVG must point in the same direction. The logic: a sweep provides the directional catalyst (institutional order flow reversal), while a nearby FVG in the same direction acts as a magnetic target that price is likely to fill. Entry at the intersection of both conditions creates a layered edge — you have confirmation from both price structure (the sweep) and order flow imbalance (the FVG). 20-bar cooldown to prevent over-trading. This fires significantly less often than individual SMC signals but has a higher expected win quality when it does trigger.`,

  FLOW_DIVERGENCE: `A flow-divergence reversal signal that uses TREND_PULLBACK as a directional baseline and then looks for contradicting evidence from three independent flow measures. Fires BEARISH when the EMA-based baseline says BULLISH but: (1) price momentum over 10 bars is negative beyond -0.5 ATR (price is actually falling despite EMA alignment), (2) volume is spiking ≥1.5× the 20-bar average (high-volume selling disguised within an uptrend), and (3) VWAP slope is declining over the last 10 bars. All three must confirm the divergence. This signal specifically hunts institutional distribution: the market looks constructive on EMAs (attracting late buyers) but the actual flow tells a different story. 30-bar cooldown. The bullish version catches institutional accumulation hidden within a downward EMA structure. Because this is a counter-trend signal, it requires higher quality (all three flow factors) to fire.`,

  MULTI_TF_CONFIRM: `A multi-timeframe trend confirmation signal that requires EMA alignment on both the 5-minute and 15-minute timeframes simultaneously. 1-minute bars are aggregated into 5m and 15m bars on the fly; both must show EMA9 > EMA20 for a bullish signal (and EMA9 < EMA20 on both for bearish). A third filter requires the current close to be above VWAP, ensuring the entry is on the correct side of intraday fair value. The 150-bar minimum (covering 2.5 hours of 1m data) ensures enough history to build reliable 15m bars. The edge: a false signal on a single timeframe rarely survives cross-timeframe scrutiny. When the 5m and 15m trend structures agree and price is above VWAP, the probability of directional follow-through is materially higher than any single-TF signal. Fires less frequently than single-timeframe signals — quality over quantity.`,

  GAP_FADE: `Fades the opening gap by trading back toward the pre-gap close. Detects a gap by comparing the session open price (30 bars back) to the prior close; if the gap exceeds 0.3%, the signal is active. Entry fires when price has not yet fully returned to the pre-gap level — meaning the gap is still open. Bullish fade: gap-down that hasn't filled (buy calls expecting price to rise back to fill). Bearish fade: gap-up that hasn't filled (buy puts expecting reversion). The edge: intraday gaps on index ETFs fill a high percentage of the time, especially in the first two hours of the session, because the gap often reflects overnight futures moves or pre-market overreaction rather than a genuine fundamental shift. Works best on moderate gaps (0.3–1.5%) in liquid, mean-reverting market conditions; large gap-and-go days will stop out this signal.`,

  VPOC_REVERSION: `Volume Point of Control reversion — trades back toward the price level where the most volume transacted over the last 30 bars. Builds a volume profile using $0.50 price buckets and identifies the VPOC (highest-volume bucket). Entry fires when current price is >0.5% away from the VPOC and RSI diverges from the displacement: bullish when price is below VPOC and RSI < 45 (both confirming the dislocation), bearish when price is above VPOC and RSI > 55. The VPOC represents the market's consensus fair value for the session window — participants transacted the most there, so it acts as a gravitational anchor. Price tends to return to the VPOC throughout the session. Works best in range-bound, high-liquidity conditions where volume profile is well-developed. In trending sessions where the VPOC itself is migrating, reversion signals will fire early.`,

  OPENING_RANGE_RECLAIM: `Trades re-entries into the opening range after a breakout fails and price reclaims the range boundary. The opening range is defined as the high and low of the first 6 bars (6 minutes). A bullish reclaim fires when the previous bar closed below the OR low (breakout attempt failed), the current bar closes back above the OR low (reclaim), price is above VWAP, and RSI is below 60 (not overbought). The bearish version mirrors this at the OR high. The edge: failed breakouts trap momentum traders who chased the break; their forced stop-outs fuel the reversal back into the range, and the reclaim signals the rejection is complete. VWAP and RSI filters remove low-quality reclaims that happen late in the day or in extended conditions. A more selective and higher-quality signal than a simple ORB because it waits for confirmation of the failure.`,

  VOL_COMPRESSION_BREAKOUT: `A volatility regime transition signal that enters when the market shifts from compression to expansion. Measures ATR (Average True Range) over a rolling window; compression is defined as the previous bar's ATR being below 60% of the 20-bar ATR mean. Expansion fires when the current bar's ATR exceeds 90% of the 20-bar mean — the market has snapped out of its tight range. Direction is set by whether the current close is above or below EMA9. The edge: after a compression period, the first expansion bar often marks the start of a sustained directional move because trapped positions from the range are unwound simultaneously. The dual ATR threshold (prev < 60%, curr > 90%) ensures the transition is sharp rather than gradual — slow drifts out of range are ignored. Works well on consolidation breakouts after news catalysts or morning ranges.`,

  VOL_SPIKE_FADE: `Fades volatility spikes by entering against the direction of an extreme ATR expansion. Entry requires: current ATR is >1.5× the 20-bar ATR mean (genuine spike, not just normal movement) AND the absolute distance between close and VWAP exceeds 1.5× the ATR mean (price has moved far from intraday fair value). Direction is determined by RSI: bearish fade when RSI > 70 (overbought spike), bullish fade when RSI < 30 (oversold spike). The edge: extreme ATR spikes in index ETFs commonly represent capitulation or short-squeeze events that exhaust momentum quickly. The VWAP displacement filter ensures you're fading a real extension, not just normal high-volatility range. Works best on news-driven spikes that occur without a fundamental change in market structure. Avoid on trend initiation days where the spike is the start of a new regime rather than an exhaustion point.`,
};

const POLL_INTERVAL = 30000; // 30s
let symbolCharts = {};   // { SPY: ApexChartsInstance, ... }
let _focusedSym = null;
const FOCUSED_CHART_H = 300;
let perfChart = null;
let simsCache = [];
let currentSimId = null;
let _symbolRegistryCache = null;

// ─────────────────────────────────────────────── CLASSROOM SOUNDS

let _soundEnabled  = true;
let _audioCtx      = null;
let _openBellDate  = null;   // date string when open bell last played
let _closeBellDate = null;   // date string when close bell last played

function _ctx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}

function _tone(freq, dur, type = 'triangle', vol = 0.22, delay = 0) {
  try {
    const ctx = _ctx();
    const osc = ctx.createOscillator();
    const g   = ctx.createGain();
    osc.connect(g); g.connect(ctx.destination);
    osc.type = type;
    osc.frequency.value = freq;
    const t = ctx.currentTime + delay;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.start(t); osc.stop(t + dur + 0.01);
  } catch (_) {}
}

function playClassroomSound(type) {
  if (!_soundEnabled) return;
  switch (type) {
    case 'entry':
      // Ascending two-note chime  C5 → E5
      _tone(523, 0.18, 'triangle', 0.20, 0.00);
      _tone(659, 0.18, 'triangle', 0.20, 0.14);
      break;
    case 'profit':
      // Bright cash-register arpeggio  C5 → E5 → G5
      _tone(523, 0.12, 'triangle', 0.22, 0.00);
      _tone(659, 0.12, 'triangle', 0.22, 0.10);
      _tone(784, 0.20, 'triangle', 0.22, 0.20);
      break;
    case 'loss':
      // Soft descending droop  E4 → A3
      _tone(330, 0.18, 'triangle', 0.18, 0.00);
      _tone(220, 0.30, 'sine',     0.14, 0.12);
      break;
    case 'open_bell':
      // Two quick school bell dings
      _tone(800, 0.28, 'square',   0.15, 0.00);
      _tone(800, 0.28, 'square',   0.15, 0.34);
      break;
    case 'close_bell':
      // Single deeper, longer ding
      _tone(600, 0.55, 'triangle', 0.20, 0.00);
      break;
  }
}

function toggleSound() {
  _soundEnabled = !_soundEnabled;
  const btn = document.getElementById('sound-toggle');
  if (!btn) return;
  btn.classList.toggle('sound-on', _soundEnabled);
  btn.textContent = _soundEnabled ? '🔊' : '🔇';
  btn.title = _soundEnabled ? 'Sound ON — click to mute' : 'Sound OFF — click to enable';
  if (_soundEnabled) {
    try { _ctx().resume(); } catch (_) {}
    _tone(660, 0.10, 'triangle', 0.15);
  }
}

// ─────────────────────────────────────────────── CHALKBOARD ANNOUNCEMENTS

let _prevSimState    = null;   // null = first load, don't announce
const _announceQueue = [];
let _announceActive  = false;

function _buildSimSnapshot(sims) {
  const snap = {};
  for (const s of sims) {
    snap[s.sim_id] = {
      open_count:    s.open_count    || 0,
      pnl_dollars:   s.pnl_dollars   || 0,
      open_trade_key: s.open_trade
        ? (s.open_trade.trade_id || s.open_trade.symbol || s.open_trade.direction || '')
        : '',
    };
  }
  return snap;
}

function detectTradeEvents(sims) {
  if (!_prevSimState) { _prevSimState = _buildSimSnapshot(sims); return; }

  for (const sim of sims) {
    const prev = _prevSimState[sim.sim_id];
    if (!prev) continue;

    const hadTrade = prev.open_count > 0;
    const hasTrade = (sim.open_count || 0) > 0;

    if (!hadTrade && hasTrade) {
      // ── New entry ──
      const dir = (sim.open_trade && sim.open_trade.direction)
        ? sim.open_trade.direction.toUpperCase() : 'LONG';
      _enqueueAnnounce(`${sim.sim_id} went ${dir}`);
      playClassroomSound('entry');

    } else if (hadTrade && !hasTrade) {
      // ── Exit ──
      const pnlDelta = (sim.pnl_dollars || 0) - prev.pnl_dollars;
      let msg;
      if (Math.abs(pnlDelta) > 0.01) {
        const sign = pnlDelta >= 0 ? '+' : '-';
        msg = `${sim.sim_id} closed ${sign}$${Math.abs(pnlDelta).toFixed(2)}`;
      } else {
        msg = `${sim.sim_id} closed a trade`;
      }
      _enqueueAnnounce(msg);
      playClassroomSound(pnlDelta >= 0 ? 'profit' : 'loss');
    }
  }

  _prevSimState = _buildSimSnapshot(sims);
}

function _enqueueAnnounce(text) {
  if (_announceQueue.length >= 5) _announceQueue.shift(); // cap queue
  _announceQueue.push(text);
  if (!_announceActive) _runAnnounceQueue();
}

function _runAnnounceQueue() {
  if (!_announceQueue.length) { _announceActive = false; return; }
  _announceActive = true;
  _playChalkMsg(_announceQueue.shift(), _runAnnounceQueue);
}

function _playChalkMsg(text, onDone) {
  const el = document.getElementById('chalk-announce');
  if (!el) { onDone && onDone(); return; }

  // Reset
  el.style.transition = '';
  el.style.opacity    = '1';
  el.textContent      = '';
  el.classList.add('visible');

  // Typewriter
  let i = 0;
  function typeNext() {
    if (i < text.length) {
      el.textContent += text[i++];
      setTimeout(typeNext, 48);
    } else {
      // Hold 3s → fade 500ms → next
      setTimeout(() => {
        el.style.transition = 'opacity 0.5s ease';
        el.style.opacity    = '0';
        setTimeout(() => {
          el.classList.remove('visible');
          el.textContent = '';
          el.style.transition = '';
          onDone && onDone();
        }, 520);
      }, 3000);
    }
  }
  typeNext();
}

// ─────────────────────────────────────────────── SUBNAV
function navTo(section, btn) {
  const targets = {
    charts: '#section-charts',
    trades: '#section-trades',
    roster: '#section-roster',
  };
  const el = document.querySelector(targets[section]);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  document.querySelectorAll('.subnav-tab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

// ─────────────────────────────────────────────── LEADERBOARD
function renderLeaderboard(sims) {
  const strip = document.getElementById('leaderboard-strip');
  if (!strip) return;
  const qualified = [...sims]
    .filter(s => (s.total_trades || 0) >= 10)
    .sort((a, b) => {
      const wDiff = (b.win_rate || 0) - (a.win_rate || 0);
      return wDiff !== 0 ? wDiff : (b.total_trades || 0) - (a.total_trades || 0);
    })
    .slice(0, 3);
  if (!qualified.length) { strip.innerHTML = ''; return; }
  const MEDALS = ['🥇', '🥈', '🥉'];
  const cards = qualified.map((s, i) => {
    const wr = s.win_rate != null ? s.win_rate.toFixed(0) + '%' : '—';
    const pnl = s.pnl_dollars != null ? (s.pnl_dollars >= 0 ? '+$' : '-$') + Math.abs(s.pnl_dollars).toFixed(0) : '';
    return `<div class="lb-card" onclick="openDrawer('${s.sim_id}')">
      <div class="lb-card-top">
        <span class="lb-card-rank">${MEDALS[i]}</span>
        <span class="lb-card-id">${s.sim_id}</span>
        <span class="lb-card-wr">${wr}</span>
      </div>
      <div class="lb-card-meta">${shortName(s.signal_mode || '')} · ${s.total_trades || 0}t${pnl ? ' · ' + pnl : ''}</div>
    </div>`;
  }).join('');
  strip.innerHTML = `<div class="lb-title">Top Performers</div><div class="lb-grid">${cards}</div>`;
}

// ─────────────────────────────────────────────── WALL POSTER
function updateWallPoster(sims) {
  const el = document.getElementById('poster-text');
  if (!el) return;
  const totalPnl    = sims.reduce((s, x) => s + (x.pnl_dollars || 0), 0);
  const activeCount = sims.filter(x => x.open_count > 0).length;
  const sign = totalPnl >= 0 ? '+' : '';
  if (activeCount > 0) {
    el.innerHTML = `${activeCount} OPEN<br>${sign}$${Math.abs(totalPnl).toFixed(0)}`;
  } else {
    el.innerHTML = `P&amp;L<br>${sign}$${Math.abs(totalPnl).toFixed(0)}`;
  }
}

// ─────────────────────────────────────────────── INIT
document.addEventListener('DOMContentLoaded', async () => {
  startClock();
  await initSymbolCharts();
  refreshAll();
  setInterval(refreshAll, POLL_INTERVAL);
});

async function refreshAll() {
  await Promise.all([
    fetchStatus(),
    fetchSims(),
    fetchChartAndPredictions(),
    fetchRecentTrades(),
  ]);
  updateRefreshTime();
}

function updateRefreshTime() {
  const el = document.getElementById('last-refresh');
  const now = new Date();
  el.textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ─────────────────────────────────────────────── STATUS
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    const dot = document.getElementById('bot-dot');
    const txt = document.getElementById('bot-status-text');
    const mkt = document.getElementById('market-status');

    dot.className = 'bot-dot ' + (d.alive ? 'alive' : 'dead');
    if (d.alive) {
      txt.textContent = `Bot alive · ${d.age_seconds}s ago`;
    } else if (d.last_heartbeat) {
      txt.textContent = `Bot offline · ${d.age_seconds}s ago`;
    } else {
      txt.textContent = 'Bot offline';
    }

    if (d.market_open) {
      mkt.textContent = `🟢 Market Open · ${d.market_time}`;
      mkt.className = 'market-open';
    } else {
      mkt.textContent = `⭕ Market Closed · ${d.market_time}`;
      mkt.className = 'market-close';
    }
  } catch (e) {
    document.getElementById('bot-status-text').textContent = 'Fetch error';
  }
}

// ─────────────────────────────────────────────── SIMS GRID
async function fetchSims() {
  try {
    const r = await fetch('/api/sims');
    simsCache = await r.json();
    detectTradeEvents(simsCache);
    renderDesks(simsCache);
    renderTeacherDesk(simsCache);
    renderTrophyShelf(simsCache);
    renderLeaderboard(simsCache);
    updateWallPoster(simsCache);
  } catch (e) {
    document.getElementById('desks-grid').innerHTML =
      '<div class="loading-msg text-red">Failed to load sims</div>';
  }
}

// ─────────────────────────────────────────────── ET CLOCK
function startClock() {
  function tick() {
    const el = document.getElementById('et-clock');
    if (!el) return;
    const now = new Date();
    const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const h = et.getHours(), m = et.getMinutes(), s = et.getSeconds();
    const pad = n => String(n).padStart(2, '0');
    const timeStr = `${pad(h)}:${pad(m)}:${pad(s)}`;
    const totalMin = h * 60 + m;
    let session;
    if (totalMin < 9 * 60 + 30)       session = 'Pre-Market';
    else if (totalMin < 16 * 60)       session = 'Market Open';
    else if (totalMin < 20 * 60)       session = 'After-Hours';
    else                               session = 'Closed';
    el.textContent = `${timeStr} ET · ${session}`;

    // Market bells — play once per calendar day
    const dateStr = et.toDateString();
    if (h === 9 && m === 30 && s < 3 && _openBellDate !== dateStr) {
      _openBellDate = dateStr;
      playClassroomSound('open_bell');
    }
    if (h === 16 && m === 0 && s < 3 && _closeBellDate !== dateStr) {
      _closeBellDate = dateStr;
      playClassroomSound('close_bell');
    }
  }
  tick();
  setInterval(tick, 1000);
}

// ─────────────────────────────────────────────── TEACHER DESK
function renderTeacherDesk(sims) {
  const panel = document.getElementById('teacher-panel');
  if (!panel || !sims.length) return;

  const totalPnl    = sims.reduce((s, x) => s + (x.pnl_dollars || 0), 0);
  const totalBal    = sims.reduce((s, x) => s + (x.balance || 0), 0);
  const totalTrades = sims.reduce((s, x) => s + (x.total_trades || 0), 0);
  const activeCount = sims.filter(x => x.open_count > 0).length;
  const wrs         = sims.map(x => x.win_rate).filter(v => v != null && v !== '' && !isNaN(v));
  const avgWr       = wrs.length ? (wrs.reduce((a, b) => a + parseFloat(b), 0) / wrs.length) : null;

  const pnlSign  = totalPnl >= 0 ? '+' : '';
  const pnlClass = totalPnl > 0 ? 'profit' : totalPnl < 0 ? 'loss' : '';

  panel.innerHTML = `
    <div class="teacher-desk-wrap">
      <div class="teacher-desk-top">
        <div class="teacher-nameplate">INSTRUCTOR DESK · ACCOUNT SUMMARY</div>
        <div class="teacher-stats-row">
          ${tdStat('Active Trades', activeCount + ' / ' + sims.length)}
          ${tdStat('Total Trades', totalTrades)}
          ${tdStat('Avg Win Rate', avgWr != null ? avgWr.toFixed(1) + '%' : '—')}
        </div>
        <div class="att-banner" style="margin-top:6px;display:flex;gap:6px">
          <span class="att-pill active-pill">${activeCount} active</span>
          <span class="att-pill">${sims.length - activeCount} idle</span>
        </div>
      </div>
    </div>
  `;
}

function tdStat(label, value, cls = '') {
  return `<div class="td-stat">
    <div class="td-stat-label">${label}</div>
    <div class="td-stat-value ${cls}">${value}</div>
  </div>`;
}

// ─────────────────────────────────────────────── RECENT TRADES PANEL
let _rtAllTrades = [];
let _rtSims = [];
let _rtSymbols = [];
let _rtExpanded = false;
const RT_DEFAULT_LIMIT = 10;

let _rtPanelOpen = false;

async function fetchRecentTrades() {
  const panel = document.getElementById('recent-trades-panel');
  if (!panel) return;
  try {
    const r = await fetch('/api/trades/recent?limit=200');
    if (!r.ok) return;
    const data = await r.json();
    _rtAllTrades = data.trades || [];
    _rtSims      = data.sims    || [];
    _rtSymbols   = data.symbols || [];
    // Update toggle header meta
    const openCnt   = _rtAllTrades.filter(t => t.status === 'open').length;
    const closedCnt = _rtAllTrades.filter(t => t.status === 'closed').length;
    const metaEl = document.getElementById('rt-toggle-meta');
    if (metaEl) metaEl.textContent = `${openCnt} open · ${closedCnt} closed`;
    if (_rtPanelOpen) renderRecentTrades(panel);
  } catch(e) {}
}

function toggleRtPanel() {
  _rtPanelOpen = !_rtPanelOpen;
  const body    = document.getElementById('rt-panel-body');
  const chevron = document.getElementById('rt-toggle-chevron');
  if (body)    body.classList.toggle('open', _rtPanelOpen);
  if (chevron) chevron.textContent = _rtPanelOpen ? '▼' : '▶';
  if (_rtPanelOpen) renderRecentTrades(document.getElementById('recent-trades-panel'));
}

function _rtFilteredTrades() {
  const simF   = (document.getElementById('rt-f-sim')   || {}).value || '';
  const symF   = (document.getElementById('rt-f-sym')   || {}).value || '';
  const _maxEPRaw = ((document.getElementById('rt-f-maxep') || {}).value || '').trim();
  const maxEP = _maxEPRaw !== '' && !isNaN(parseFloat(_maxEPRaw)) ? parseFloat(_maxEPRaw) : null;
  const _minEPRaw = ((document.getElementById('rt-f-minep') || {}).value || '').trim();
  const minEP = _minEPRaw !== '' && !isNaN(parseFloat(_minEPRaw)) ? parseFloat(_minEPRaw) : null;
  return _rtAllTrades.filter(t => {
    if (simF && t.sim_id !== simF) return false;
    if (symF && t.symbol !== symF) return false;
    if (maxEP != null && t.entry_price != null && parseFloat(t.entry_price) > maxEP) return false;
    if (minEP != null && t.entry_price != null && parseFloat(t.entry_price) < minEP) return false;
    return true;
  });
}

function applyRtFilter() {
  _rtExpanded = false;
  renderRecentTrades(document.getElementById('recent-trades-panel'));
}

function toggleRtShowMore() {
  _rtExpanded = !_rtExpanded;
  renderRecentTrades(document.getElementById('recent-trades-panel'));
}

function renderRecentTrades(panel) {
  if (!panel) return;
  const filtered = _rtFilteredTrades();
  const limit    = _rtExpanded ? filtered.length : RT_DEFAULT_LIMIT;
  const visible  = filtered.slice(0, limit);
  const openCnt  = filtered.filter(t => t.status === 'open').length;

  if (!filtered.length) {
    panel.innerHTML = `<div class="rt-panel">
      <div class="rt-header-row">
        <div class="rt-title">RECENT TRADES <span class="rt-subtitle">no results</span></div>
        ${_rtFilterBar()}
      </div></div>`;
    return;
  }

  const rows = visible.map((t, idx) => {
    const pnl     = t.pnl != null ? parseFloat(t.pnl) : null;
    const pnlPct  = t.pnl_pct != null ? (parseFloat(t.pnl_pct) * 100).toFixed(1) : null;
    const pnlCls  = pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : '';
    const pnlSign = pnl > 0 ? '+' : '';
    const pnlStr  = pnl != null ? pnlSign + '$' + fmt2(Math.abs(pnl)) : '—';
    const dir     = (t.direction || '').toUpperCase();
    const ct      = (t.contract_type || '').toUpperCase();
    const ctShort = ct === 'CALL' ? 'C' : ct === 'PUT' ? 'P' : ct.charAt(0) || '?';
    const strike  = t.strike != null ? '$' + parseFloat(t.strike).toFixed(0) : '—';
    const expiry  = t.expiry ? t.expiry.slice(5) : '—';
    const entryP  = t.entry_price != null ? '$' + fmt4(t.entry_price) : '—';
    const exitP   = t.exit_price  != null ? '$' + fmt4(t.exit_price)  : '—';
    const slStr   = t.sl_price != null ? '$' + fmt4(t.sl_price) : '—';
    const tpStr   = t.tp_price != null ? '$' + fmt4(t.tp_price) : '—';
    const isOpen  = t.status === 'open';
    const exitTime  = t.exit_time  ? fmtTime(t.exit_time)  : isOpen ? '<span class="rt-open-badge">OPEN</span>' : '—';
    const entryTime = t.entry_time ? fmtTime(t.entry_time) : '—';
    const holdStr = (() => {
      if (!t.entry_time) return null;
      const end = t.exit_time ? new Date(t.exit_time) : new Date();
      const secs = Math.max(0, Math.round((end - new Date(t.entry_time)) / 1000));
      if (secs < 60) return `${secs}s`;
      const m = Math.floor(secs / 60), h = Math.floor(m / 60), d = Math.floor(h / 24);
      if (d > 0) return `${d}d ${h % 24}h`;
      if (h > 0) return `${h}h ${m % 60}m`;
      return `${m}m`;
    })();
    const expandId  = `rt-expand-${idx}`;
    const rowCls    = isOpen ? 'rt-row rt-row-open' : `rt-row ${pnlCls}`;
    return `
    <tr class="${rowCls}" onclick="toggleRtExpand('${expandId}', this)">
      <td class="rt-sim">${t.sim_id}</td>
      <td><span class="sym-badge sym-${t.symbol}">${t.symbol}</span></td>
      <td><span class="dir-badge ${dir}">${dir || '—'}</span></td>
      <td class="rt-contract">${strike} ${ctShort} ${expiry}</td>
      <td class="rt-price">${entryP}</td>
      <td class="rt-price">${exitP}</td>
      <td class="rt-pnl ${pnlCls}">${isOpen ? '<span style="color:var(--dim);font-style:italic">open</span>' : pnlStr}</td>
      <td class="rt-time">${exitTime}</td>
      <td class="rt-chevron">▶</td>
    </tr>
    <tr class="rt-expand-row hidden" id="${expandId}">
      <td colspan="9">
        <div class="rt-expand-grid">
          <div class="rt-expand-item"><span class="rt-el">SL</span><span class="rt-ev neg">${slStr}</span></div>
          <div class="rt-expand-item"><span class="rt-el">TP</span><span class="rt-ev pos">${tpStr}</span></div>
          ${pnlPct != null ? `<div class="rt-expand-item"><span class="rt-el">P&L%</span><span class="rt-ev ${pnlCls}">${pnlSign}${pnlPct}%</span></div>` : ''}
          <div class="rt-expand-item"><span class="rt-el">Entry Time</span><span class="rt-ev">${entryTime}</span></div>
          ${holdStr ? `<div class="rt-expand-item"><span class="rt-el">Held</span><span class="rt-ev">${holdStr}</span></div>` : ''}
          <div class="rt-expand-item"><span class="rt-el">Qty</span><span class="rt-ev">${t.qty ?? '—'}</span></div>
          ${t.regime ? `<div class="rt-expand-item"><span class="rt-el">Regime</span><span class="rt-ev">${t.regime}</span></div>` : ''}
          ${!isOpen ? `<div class="rt-expand-item"><span class="rt-el">Exit Reason</span><span class="rt-ev">${t.exit_reason || '—'}</span></div>` : ''}
          ${t.exit_context ? `<div class="rt-expand-item rt-expand-wide"><span class="rt-el">Detail</span><span class="rt-ev">${t.exit_context}</span></div>` : ''}
          ${t.signal_mode ? `<div class="rt-expand-item"><span class="rt-el">Strategy</span><span class="rt-ev">${t.signal_mode}</span></div>` : ''}
        </div>
      </td>
    </tr>`;
  }).join('');

  const remaining = filtered.length - visible.length;
  const showMoreBtn = filtered.length > RT_DEFAULT_LIMIT ? `
    <button class="rt-show-more-btn" onclick="toggleRtShowMore()">
      ${_rtExpanded ? '▲ Show less' : `▼ Show ${remaining} more`}
    </button>` : '';

  const subtitle = `${openCnt > 0 ? `${openCnt} open · ` : ''}${filtered.length} total`;

  panel.innerHTML = `
    <div class="rt-panel">
      <div class="rt-header-row">
        <div class="rt-title">RECENT TRADES <span class="rt-subtitle">${subtitle}</span></div>
        ${_rtFilterBar()}
      </div>
      <div class="rt-scroll">
        <table class="rt-table">
          <thead><tr>
            <th>Sim</th><th>Symbol</th><th>Dir</th><th>Contract</th>
            <th>Entry</th><th>Exit</th><th>P&L</th><th>Time</th><th></th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${showMoreBtn}
    </div>`;
}

function _rtFilterBar() {
  const curSim   = (document.getElementById('rt-f-sim')   || {}).value || '';
  const curSym   = (document.getElementById('rt-f-sym')   || {}).value || '';
  const curMinEP = (document.getElementById('rt-f-minep') || {}).value || '';
  const curMaxEP = (document.getElementById('rt-f-maxep') || {}).value || '';
  const simOpts  = _rtSims.map(s =>
    `<option value="${s}"${s === curSim ? ' selected' : ''}>${s}</option>`).join('');
  const symOpts  = _rtSymbols.map(s =>
    `<option value="${s}"${s === curSym ? ' selected' : ''}>${s}</option>`).join('');
  return `<div class="rt-filters">
    <select id="rt-f-sim" class="rt-filter-sel" onchange="applyRtFilter()">
      <option value=""${!curSim ? ' selected' : ''}>All Sims</option>${simOpts}
    </select>
    <select id="rt-f-sym" class="rt-filter-sel" onchange="applyRtFilter()">
      <option value=""${!curSym ? ' selected' : ''}>All Symbols</option>${symOpts}
    </select>
    <input id="rt-f-minep" type="text" inputmode="decimal" class="rt-filter-input" placeholder="Min entry $"
           value="${curMinEP}" onchange="applyRtFilter()" onkeydown="if(event.key==='Enter')applyRtFilter()"/>
    <input id="rt-f-maxep" type="text" inputmode="decimal" class="rt-filter-input" placeholder="Max entry $"
           value="${curMaxEP}" onchange="applyRtFilter()" onkeydown="if(event.key==='Enter')applyRtFilter()"/>
  </div>`;
}

function toggleRtExpand(expandId, rowEl) {
  const expandRow = document.getElementById(expandId);
  if (!expandRow) return;
  const isOpen = !expandRow.classList.contains('hidden');
  expandRow.classList.toggle('hidden', isOpen);
  const chevron = rowEl.querySelector('.rt-chevron');
  if (chevron) chevron.textContent = isOpen ? '▶' : '▼';
}

// ─────────────────────────────────────────────── CLASSROOM RENDERING
const COLS = 6; // desks per row (6×N layout)

function getPersonality(signalMode) {
  const m = (signalMode || '').toUpperCase();
  if (['MEAN_REVERSION','VWAP_REVERSION','ZSCORE_BOUNCE','FAILED_BREAKOUT_REVERSAL','EXTREME_EXTENSION_FADE'].includes(m)) return 'scholar';
  if (['BREAKOUT','ORB_BREAKOUT','AFTERNOON_BREAKOUT','OPENING_DRIVE'].includes(m)) return 'athlete';
  if (['TREND_PULLBACK','SWING_TREND','VWAP_CONTINUATION','TREND_RECLAIM'].includes(m)) return 'trend';
  return 'casual';
}

function buildBubbleText(sim) {
  if (sim.is_disabled) return 'z z z…';
  if (sim.open_count > 0) return '📈 In trade!';
  if (sim.pnl_dollars > 50)  return '🎉 +$' + fmt2(sim.pnl_dollars);
  if (sim.pnl_dollars < -50) return '😬 -$' + fmt2(Math.abs(sim.pnl_dollars));
  if (sim.total_trades === 0) return '⏳ Waiting…';
  return '📋 ' + shortName(sim.signal_mode || '') ;
}

// Strategy grouping order + display names
const STRATEGY_ORDER = ['scholar', 'athlete', 'trend', 'casual'];
const STRATEGY_LABEL = {
  scholar:  'MEAN REVERSION',
  athlete:  'BREAKOUT',
  trend:    'TREND',
  casual:   'OPPORTUNITY',
};

function renderDesks(sims) {
  const grid    = document.getElementById('desks-grid');
  const countEl = document.getElementById('sim-count');
  if (!sims.length) {
    grid.innerHTML = '<div class="loading-msg">No students enrolled yet.</div>';
    return;
  }

  const enabledSims = sims.filter(s => !s.is_disabled);
  const activeCount = sims.filter(s => s.open_count > 0).length;
  if (countEl) countEl.textContent = `${enabledSims.length} students · ${activeCount} active`;

  grid.innerHTML = '';

  // Separate SIM00 (live), disabled sims, and active paper sims
  const liveSim     = sims.find(s => s.sim_id === 'SIM00');
  const activeSims  = sims.filter(s => s.sim_id !== 'SIM00' && !s.is_disabled);
  const sleepSims   = sims.filter(s => s.sim_id !== 'SIM00' && s.is_disabled);

  // Group active paper sims by strategy personality
  const groups = {};
  STRATEGY_ORDER.forEach(k => { groups[k] = []; });
  activeSims.forEach(sim => {
    const p = getPersonality(sim.signal_mode);
    (groups[p] = groups[p] || []).push(sim);
  });
  STRATEGY_ORDER.forEach(k => {
    if (groups[k]) groups[k].sort((a, b) => (b.win_rate ?? -1) - (a.win_rate ?? -1));
  });

  function appendRow(label, group, extraClass) {
    const labelEl = document.createElement('div');
    labelEl.className = 'strategy-label' + (extraClass ? ' ' + extraClass : '');
    labelEl.textContent = label;
    grid.appendChild(labelEl);

    const rowEl = document.createElement('div');
    rowEl.className = 'desk-row';
    group.forEach(sim => rowEl.appendChild(buildSeat(sim)));
    grid.appendChild(rowEl);
  }

  // Live sim first — right next to teacher's desk
  if (liveSim) appendRow('LIVE', [liveSim], 'strategy-label-live');

  STRATEGY_ORDER.forEach(key => {
    const group = groups[key];
    if (!group || !group.length) return;
    appendRow(STRATEGY_LABEL[key] || key.toUpperCase(), group, '');
  });

  // Disabled sims at the bottom — on the bench / asleep
  if (sleepSims.length) {
    appendRow('💤 ON THE BENCH', sleepSims, 'strategy-label-bench');
  }
}

// ─────────────────────────────────────────────── PERSONALITY TOOLTIPS

const SIM_PERSONALITIES = {
  MEAN_REVERSION:           '"I only buy the dip."',
  VWAP_REVERSION:           '"Everything returns to VWAP. Everything."',
  ZSCORE_BOUNCE:            '"They called it extreme. I called it entry."',
  FAILED_BREAKOUT_REVERSAL: '"Fakeouts are my bread and butter."',
  EXTREME_EXTENSION_FADE:   '"The further they run, the harder they fall."',
  BREAKOUT:                 '"New highs? I\'m already in."',
  ORB_BREAKOUT:             '"First 30 minutes. That\'s all I need."',
  AFTERNOON_BREAKOUT:       '"I wait while others get chopped up."',
  OPENING_DRIVE:            '"Gap and go. No hesitation."',
  TREND_PULLBACK:           '"Buy the pullback, ride the trend."',
  SWING_TREND:              '"I don\'t day-trade. I day-wait."',
  VWAP_CONTINUATION:        '"Near VWAP and trending? Say less."',
  TREND_RECLAIM:            '"Lost levels always get reclaimed."',
  OPPORTUNITY:              '"I take whatever the market gives me."',
  FVG_4H:                   '"Gaps get filled. I have patience."',
  FVG_5M:                   '"Quick gaps, quick fills, quick profits."',
  LIQUIDITY_SWEEP:          '"They hunt stops. I hunt them back."',
  FVG_SWEEP_COMBO:          '"Sweep AND a gap? Now we\'re talking."',
  FLOW_DIVERGENCE:          '"Price lies. Flow tells the truth."',
  MULTI_TF_CONFIRM:         '"I wait for everyone to agree."',
  GAP_FADE:                 '"Overnight gaps are overreactions. Usually."',
  VPOC_REVERSION:           '"Volume doesn\'t lie. Price always returns."',
  OPENING_RANGE_RECLAIM:    '"Flush, reclaim, profit. Every time."',
  VOL_COMPRESSION_BREAKOUT: '"Quiet markets are loaded springs."',
  VOL_SPIKE_FADE:           '"Panic is my entry signal."',
};

let _simTT = null;
function _getTooltip() {
  if (!_simTT) {
    _simTT = document.createElement('div');
    _simTT.id = 'sim-tooltip';
    _simTT.innerHTML = '<div class="stt-phrase"></div><div class="stt-meta"></div>';
    document.body.appendChild(_simTT);
  }
  return _simTT;
}

function showSimTooltip(seat, sim) {
  const tt = _getTooltip();
  const phrase = SIM_PERSONALITIES[sim.signal_mode] || '"Just running my algo."';
  tt.querySelector('.stt-phrase').textContent = phrase;
  tt.querySelector('.stt-meta').textContent = `${sim.sim_id} · ${shortName(sim.signal_mode || '')}`;

  tt.classList.remove('visible', 'stt-below');

  const r   = seat.getBoundingClientRect();
  const ttH = tt.offsetHeight || 56;
  const ttW = tt.offsetWidth  || 200;

  let top, below = false;
  if (r.top - ttH - 14 < 8) { top = r.bottom + 10; below = true; }
  else                       { top = r.top - ttH - 14; }

  const centerX  = r.left + r.width / 2;
  const halfW    = ttW / 2;
  const clampedX = Math.max(halfW + 8, Math.min(window.innerWidth - halfW - 8, centerX));

  tt.style.top  = top + 'px';
  tt.style.left = clampedX + 'px';
  if (below) tt.classList.add('stt-below');
  requestAnimationFrame(() => tt.classList.add('visible'));
}

function hideSimTooltip() {
  if (_simTT) _simTT.classList.remove('visible');
}

// ─────────────────────────────────────────────── TROPHY SHELF

function renderTrophyShelf(sims) {
  const shelf = document.getElementById('trophy-shelf');
  if (!shelf) return;

  // Rank: 10+ trades, by win_rate desc then total_trades desc
  const qualified = [...sims]
    .filter(s => (s.total_trades || 0) >= 10)
    .sort((a, b) => {
      const wDiff = (b.win_rate || 0) - (a.win_rate || 0);
      return wDiff !== 0 ? wDiff : (b.total_trades || 0) - (a.total_trades || 0);
    })
    .slice(0, 3);

  const SLOTS = [
    // [cx, cup top y, cup w, cup h, color, dark, hi]
    [10,  20, 16, 14, '#FFD700', '#B8860B', '#FFEE88'],  // gold
    [32,  24, 14, 12, '#C0C0C0', '#808080', '#E8E8E8'],  // silver
    [54,  28, 12, 11, '#CD7F32', '#8B4513', '#E8A050'],  // bronze
  ];

  // Build trophy SVG paths for each slot
  function trophyRects(cx, cupTopY, cw, ch, col, dark, hi) {
    const lx = cx - Math.floor(cw / 2);
    const rx = cx + Math.ceil(cw / 2);
    const cupBotY = cupTopY + ch;
    const stemW = Math.max(3, Math.floor(cw / 4));
    const stemX = cx - Math.floor(stemW / 2);
    const stemH = 5;
    const baseW = cw - 2;
    const baseX = cx - Math.floor(baseW / 2);
    const shelfY = 46;
    const baseY = shelfY - 2;
    const stemTopY = baseY - stemH;
    // recalculate cup bottom to match stem
    const actualCupBotY = stemTopY;
    const actualCupTopY = actualCupBotY - ch;
    const rimY = actualCupTopY - 2;
    const hlW = 2;

    return `
      <rect x="${lx}"       y="${rimY}"          width="${cw+2}" height="2"     fill="${col}"/>
      <rect x="${lx}"       y="${actualCupTopY}" width="${cw}"   height="${ch}"  fill="${col}"/>
      <rect x="${lx-3}"     y="${actualCupTopY+3}" width="3"     height="${Math.floor(ch*0.4)}" fill="${dark}"/>
      <rect x="${rx}"       y="${actualCupTopY+3}" width="3"     height="${Math.floor(ch*0.4)}" fill="${dark}"/>
      <rect x="${lx+1}"     y="${actualCupTopY+1}" width="${hlW}" height="${Math.floor(ch*0.6)}" fill="${hi}" opacity="0.55"/>
      <rect x="${stemX}"    y="${stemTopY}"      width="${stemW}" height="${stemH}" fill="${dark}"/>
      <rect x="${baseX}"    y="${baseY}"         width="${baseW}" height="2"     fill="${dark}"/>
      <rect x="${baseX+1}"  y="${baseY}"         width="${baseW-2}" height="1"  fill="${col}" opacity="0.4"/>`;
  }

  const svgParts = SLOTS.map(([cx, cy, cw, ch, col, dark, hi], i) => {
    const s = qualified[i];
    return s ? trophyRects(cx, cy, cw, ch, col, dark, hi) : `
      <rect x="${cx-4}" y="36" width="8" height="10" fill="${col}" opacity="0.18"/>
      <rect x="${cx-3}" y="46" width="6" height="1"  fill="${dark}" opacity="0.15"/>`;
  }).join('');

  const labelParts = SLOTS.map(([cx], i) => {
    const s = qualified[i];
    const id = s ? s.sim_id : '—';
    const wr = s ? (s.win_rate || 0).toFixed(0) + '%' : '';
    return `<div class="trophy-label" style="flex:0 0 ${Math.round(100/3)}%;text-align:center">
      <div class="tl-id">${id}</div>
      <div class="tl-wr">${wr}</div>
    </div>`;
  }).join('');

  const emptyMsg = qualified.length === 0
    ? '<div class="trophy-empty">Term just started!</div>' : '';

  shelf.innerHTML = `
    <div class="trophy-title">TOP 3</div>
    ${emptyMsg}
    <svg viewBox="0 0 64 52" width="64" height="52"
         xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">
      ${svgParts}
      <!-- shelf surface -->
      <rect x="0" y="46" width="64" height="4" fill="#b07828"/>
      <rect x="0" y="46" width="64" height="1" fill="#c88838"/>
      <rect x="0" y="50" width="64" height="2" fill="rgba(0,0,0,0.22)"/>
      <!-- brackets -->
      <rect x="7"  y="46" width="2" height="6" fill="#7a4226"/>
      <rect x="5"  y="50" width="6" height="2" fill="#7a4226"/>
      <rect x="55" y="46" width="2" height="6" fill="#7a4226"/>
      <rect x="53" y="50" width="6" height="2" fill="#7a4226"/>
    </svg>
    <div class="trophy-labels">${labelParts}</div>`;
}

function buildSeat(sim) {
  const active      = sim.open_count > 0;
  const mood        = sim.pnl_dollars > 0 ? 'happy' : sim.pnl_dollars < 0 ? 'sad' : 'neutral';
  const colorIdx    = Math.max(0, parseInt(sim.sim_id.replace('SIM', ''), 10) - 1);
  const archetype   = SIM_STYLES[colorIdx] || 'casual';
  const personality = getPersonality(sim.signal_mode);
  const bubbleText  = buildBubbleText(sim);

  // Win rate display with colour coding
  const wr = sim.win_rate;  // already a percentage number, e.g. 62.5
  let wrText, wrClass;
  if (wr == null || sim.total_trades === 0) {
    wrText = '—'; wrClass = 'wr-none';
  } else {
    wrText = wr.toFixed(0) + '%';
    if      (wr >= 85) wrClass = 'wr-great';
    else if (wr >= 75) wrClass = 'wr-good';
    else if (wr >= 50) wrClass = 'wr-ok';
    else               wrClass = 'wr-bad';
  }

  // Sims disabled via blocked_sessions (all sessions blocked) show as sleeping/grayed.
  // New sims with 0 trades but not disabled show as normal (idle, awake).
  const sleeping = !!sim.is_disabled;
  const seat = document.createElement('div');
  const wasSelected = currentSimId === sim.sim_id;
  seat.className = 'seat' + (active ? ' active' : '') + (sleeping ? ' sleeping' : '') + (wasSelected ? ' selected' : '');
  seat.dataset.simId = sim.sim_id;
  seat.dataset.archetype = archetype;
  seat.onclick = () => openDrawer(sim.sim_id);

  // Personality tooltip (mouse-only — skipped on touch devices)
  if (!('ontouchstart' in window)) {
    seat.addEventListener('mouseenter', () => showSimTooltip(seat, sim));
    seat.addEventListener('mouseleave', hideSimTooltip);
  }

  const notebookClass = 'notebook ' + (wr == null || sim.total_trades === 0 ? '' : wr >= 50 ? 'profit' : 'loss');

  const streakBadge = (() => {
    const s = sim.streak;
    if (!s || s.count < 1) return '';
    const n = s.count;
    if (s.type === 'win') {
      const FLAME_COLORS = ['#ff2222','#ff6600','#ffdd00','#44ee44','#3399ff','#bb44ff','#111111','#888888','#ffffff'];
      const color = FLAME_COLORS[Math.min(n - 1, FLAME_COLORS.length - 1)];
      return `<div class="streak-badge" style="color:${color};text-shadow:0 0 6px ${color}88" title="${n}-win streak">🔥${n > 1 ? `<span class="streak-num">${n}</span>` : ''}</div>`;
    } else {
      const STORM_COLORS = ['#88aaff','#4477ff','#2255dd','#7733cc','#440088','#220044','#555555','#888888','#aaaaaa'];
      const color = STORM_COLORS[Math.min(n - 1, STORM_COLORS.length - 1)];
      return `<div class="streak-badge" style="color:${color};text-shadow:0 0 6px ${color}88" title="${n}-loss streak">⛈️${n > 1 ? `<span class="streak-num">${n}</span>` : ''}</div>`;
    }
  })();

  seat.innerHTML = `
    <div class="student-area">
      <div class="speech-bubble">${bubbleText}</div>
      ${sleeping ? sleepingSVG(colorIdx) : studentSVG(sim.sim_id, mood, colorIdx, personality, active)}
    </div>
    ${streakBadge}
    <div class="seat-info">
      <div class="seat-info-row">
        <span class="desk-id">${sim.sim_id}</span>
        <span class="desk-trades">${sim.total_trades || 0}t</span>
        <span class="desk-wr ${wrClass}">${wrText}</span>
      </div>
      <div class="desk-footer">${shortName(sim.signal_mode || '')}</div>
      <div class="archetype-badge archetype-badge-${archetype}">${archetype}</div>
      ${sim.symbols && sim.symbols.length ? `<div class="desk-symbols">${sim.symbols.join(' · ')}</div>` : ''}
    </div>
  `;
  return seat;
}

// ─────────────────────────────────────────────── STUDENT SVG (classroom character)
const SHIRT_COLORS = [
  '#4a6fa8','#7a4ea0','#4a8858','#b07828','#a0426a',
  '#287880','#a04e28','#4a7828','#a03030','#285880',
  '#887828','#4a3ea0','#287860','#784050','#2a4880',
  '#887048','#2a5860','#806828','#903058','#2a2880',
  '#2a5878','#6a7828','#682858',
];
const SKIN_TONES = ['#f2c08a','#e8a068','#c87840','#a86028','#7a4820'];
const HAIR_COLORS = ['#181008','#4a2810','#b08028','#a83818','#585858','#0e0e0e'];
// Desk item variation per student: determines what items appear on their desk
const DESK_ITEMS = ['papers_mug', 'books_pencil', 'papers_apple', 'notebook_mug', 'books_mug', 'papers_pencil'];

// ─────────────────────────────────────────────── STYLE ARCHETYPES (v2 — 14 styles)
const SIM_STYLES = [
  'formal','punk','nerd','pastel','grunge','street',       // SIM00–05
  'emo','cottagecore','jock','artsy','gangster','techy',   // SIM06–11
  'preppy','punk','casual','pastel','formal','emo',        // SIM12–17
  'grunge','street','artsy','nerd','cottagecore','jock',   // SIM18–23
  'gangster','techy','preppy','punk','casual','emo',       // SIM24–29
  'artsy','formal','grunge','street','nerd','cottagecore', // SIM30–35
];
const ARCHETYPE_SHIRTS = {
  punk:        ['#1a1a1a','#141414','#222222'],
  grunge:      ['#8B0000','#780000','#9e0808'],
  formal:      ['#F0F0F0','#E8E8E8','#F8F8F8'],
  street:      ['#FF8C00','#e07800','#ff9a18'],
  casual:      ['#4682B4','#3a72a0','#5090c8'],
  gangster:    ['#6A0DAD','#5a0098','#7a1ac0'],
  nerd:        ['#87CEEB','#70bcd8','#a0deff'],
  jock:        ['#E03030','#cc2828','#f04040'],
  emo:         ['#2D0040','#240034','#380050'],
  pastel:      ['#FFB6C1','#f0a0b0','#ffc8d0'],
  artsy:       ['#FFD700','#e0be00','#ffe820'],
  techy:       ['#404040','#363636','#4a4a4a'],
  cottagecore: ['#8FBC8F','#7aaa7a','#a0cca0'],
  preppy:      ['#1E3A5F','#183050','#2a4870'],
};
const ARCHETYPE_BLANKETS = {
  punk:'#1a1a1a', grunge:'#8a3018', formal:'#1e2d40', street:'#383838',
  casual:'#5878b8', gangster:'#1a2a1a', nerd:'#2a5888', jock:'#c03020',
  emo:'#180818', pastel:'#FFB6C1', artsy:'#c87828', techy:'#0a0a1a',
  cottagecore:'#c8b890', preppy:'#7ba0c8',
};

function studentSVG(simId, mood, idx, personality = 'casual', active = false) {
  const archetype = SIM_STYLES[idx] || 'casual';
  const skin      = SKIN_TONES[(idx * 3) % SKIN_TONES.length];
  const arcS      = ARCHETYPE_SHIRTS[archetype] || ARCHETYPE_SHIRTS.casual;
  const shirt     = arcS[idx % arcS.length];

  // ── Gender assignment (deterministic by idx)
  // idx = simNumber - 1 clamped to 0; SIM00+SIM01 share idx=0 → both female
  // Male at idx 1,5,9,13,18,23,28,33 = SIM02,SIM06,SIM10,SIM14,SIM19,SIM24,SIM29,SIM34 (~22%)
  const MALE_IDXS = new Set([1, 5, 9, 13, 18, 23, 28, 33]);
  const isFemale = !MALE_IDXS.has(idx);

  // Hair color — male uses original palette, female gets wider variety
  const FEMALE_HAIR_COLORS = [
    '#181008', // black
    '#3a1a08', // espresso
    '#4a2810', // dark brown
    '#7a3818', // auburn
    '#b08028', // blonde
    '#c89840', // light blonde
    '#a83818', // red
    '#585858', // dark grey
    '#7b3f8c', // purple (fun)
    '#2255aa', // blue (fun)
  ];
  let hair = isFemale
    ? FEMALE_HAIR_COLORS[(idx * 3 + 1) % FEMALE_HAIR_COLORS.length]
    : HAIR_COLORS[(idx * 2) % HAIR_COLORS.length];

  // Archetype hair overrides — strong visual signal
  if (archetype === 'punk') {
    const PUNK_HAIR = ['#39FF14','#FF00FF','#00FFFF','#FF2222'];
    hair = PUNK_HAIR[idx % PUNK_HAIR.length];
  } else if (archetype === 'emo') {
    hair = '#090909';
  } else if (archetype === 'pastel') {
    const PASTEL_HAIR = ['#C8A0D8','#F0A0C0','#E8E8FF'];
    hair = PASTEL_HAIR[idx % PASTEL_HAIR.length];
  } else if (archetype === 'artsy') {
    const ARTSY_HAIR = ['#FF4500','#FFD700','#FF1493','#00CED1'];
    hair = ARTSY_HAIR[idx % ARTSY_HAIR.length];
  }

  // Darker / lighter shades for hair detail
  const shadeHex = (hex, delta) => hex.replace(/[0-9a-f]{2}/gi, h =>
    Math.min(255, Math.max(0, parseInt(h, 16) + delta)).toString(16).padStart(2, '0'));
  const hairDark  = shadeHex(hair, -28);
  const hairLight = shadeHex(hair, +22);

  // Darker shade for shirt details (collar, shadow)
  const shirtDark = shirt.replace(/[0-9a-f]{2}/gi, (h, i) =>
    Math.max(0, parseInt(h, 16) - 30).toString(16).padStart(2, '0'));

  // Laptop screen color reflects mood
  const screenBg = mood === 'happy' ? '#0a2e14' : mood === 'sad' ? '#2e0a0a' : '#0a1a2e';
  const screenFg = mood === 'happy' ? '#44dd44' : mood === 'sad' ? '#dd4444' : '#4488dd';
  const screenGlow = active ? `<rect x="15" y="42" width="18" height="10" fill="${screenFg}" opacity="0.12"/>` : '';

  // Eyelashes for female characters (1-2 pixels above each eye)
  const lashes = isFemale ? `
      <rect x="18" y="17" width="3" height="1" fill="#2a2018" opacity="0.75"/>
      <rect x="20" y="16" width="1" height="1" fill="#2a2018" opacity="0.5"/>
      <rect x="27" y="17" width="3" height="1" fill="#2a2018" opacity="0.75"/>
      <rect x="29" y="16" width="1" height="1" fill="#2a2018" opacity="0.5"/>` : '';

  // Face expression
  let eyes = '', mouth = '';
  if (mood === 'happy') {
    eyes = `${lashes}
      <rect x="18" y="18" width="3" height="2" rx="0" fill="#2a2018"/>
      <rect x="27" y="18" width="3" height="2" rx="0" fill="#2a2018"/>
      <rect x="19" y="17" width="1" height="1" fill="#2a2018" opacity="0.4"/>
      <rect x="28" y="17" width="1" height="1" fill="#2a2018" opacity="0.4"/>`;
    mouth = `<rect x="20" y="23" width="8" height="2" rx="1" fill="#2a2018" opacity="0.6"/>
      <rect x="21" y="24" width="6" height="1" fill="#fff" opacity="0.3"/>`;
  } else if (mood === 'sad') {
    eyes = `${lashes}
      <rect x="18" y="18" width="3" height="3" rx="0" fill="#2a2018"/>
      <rect x="18" y="18" width="3" height="1" fill="${skin}" opacity="0.5"/>
      <rect x="27" y="18" width="3" height="3" rx="0" fill="#2a2018"/>
      <rect x="27" y="18" width="3" height="1" fill="${skin}" opacity="0.5"/>`;
    mouth = `<rect x="21" y="24" width="6" height="1" fill="#2a2018" opacity="0.5"/>`;
  } else {
    eyes = `${lashes}
      <rect x="18" y="18" width="3" height="3" rx="0" fill="#2a2018"/>
      <rect x="19" y="18" width="1" height="1" fill="#fff" opacity="0.6"/>
      <rect x="27" y="18" width="3" height="3" rx="0" fill="#2a2018"/>
      <rect x="28" y="18" width="1" height="1" fill="#fff" opacity="0.6"/>`;
    mouth = `<rect x="21" y="24" width="6" height="1" fill="#2a2018" opacity="0.4"/>`;
  }

  // Personality accessories — pixel art style
  let accessory = '';
  if (personality === 'scholar') {
    accessory = `
      <rect x="16" y="17" width="7" height="5" fill="none" stroke="#3a3a5a" stroke-width="1.2"/>
      <rect x="25" y="17" width="7" height="5" fill="none" stroke="#3a3a5a" stroke-width="1.2"/>
      <rect x="23" y="19" width="2" height="1" fill="#3a3a5a"/>
      <rect x="16" y="19" width="1" height="1" fill="#3a3a5a" opacity="0.6"/>
      <rect x="32" y="19" width="1" height="1" fill="#3a3a5a" opacity="0.6"/>
      <rect x="16" y="17" width="1" height="1" fill="rgba(255,255,255,0.2)"/>
      <rect x="25" y="17" width="1" height="1" fill="rgba(255,255,255,0.2)"/>`;
  } else if (personality === 'athlete') {
    accessory = `
      <rect x="11" y="11" width="26" height="4" fill="#c02828"/>
      <rect x="11" y="12" width="26" height="1" fill="#e04040" opacity="0.6"/>
      <rect x="22" y="11" width="4" height="4" fill="#e8e8e8" opacity="0.3"/>`;
  } else if (personality === 'trend') {
    accessory = `
      <rect x="22" y="32" width="4" height="2" fill="#d0a020"/>
      <rect x="22" y="34" width="4" height="2" fill="#b89018"/>
      <rect x="23" y="36" width="2" height="4" fill="#a08010"/>
      <rect x="22" y="40" width="4" height="2" fill="#b89018"/>`;
  }

  // ── HAIR ─────────────────────────────────────────────────────────────
  let hairSVG = '';
  if (!isFemale) {
    // Original male hair styles (4 variants)
    const hairStyle = idx % 4;
    if (hairStyle === 0) {
      hairSVG = `
      <rect x="12" y="4" width="24" height="10" rx="4" fill="${hair}"/>
      <rect x="14" y="3" width="20" height="4" rx="3" fill="${hair}"/>
      <rect x="13" y="10" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (hairStyle === 1) {
      hairSVG = `
      <rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/>
      <rect x="11" y="6" width="12" height="6" rx="2" fill="${hair}"/>
      <rect x="14" y="3" width="18" height="4" rx="3" fill="${hair}"/>
      <rect x="13" y="10" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (hairStyle === 2) {
      hairSVG = `
      <rect x="12" y="5" width="24" height="10" rx="4" fill="${hair}"/>
      <rect x="16" y="2" width="4" height="5" rx="1" fill="${hair}"/>
      <rect x="22" y="1" width="4" height="6" rx="1" fill="${hair}"/>
      <rect x="28" y="3" width="3" height="4" rx="1" fill="${hair}"/>
      <rect x="13" y="10" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else {
      hairSVG = `
      <rect x="11" y="4" width="26" height="14" rx="6" fill="${hair}"/>
      <ellipse cx="24" cy="5" rx="12" ry="5" fill="${hair}"/>
      <rect x="13" y="10" width="22" height="4" fill="${hair}" opacity="0.3"/>`;
    }
  } else {
    // Female hair styles (6 variants, idx % 6)
    const femStyle = idx % 6;
    if (femStyle === 0) {
      // Long straight — strands down beside shoulders
      hairSVG = `
      <rect x="11" y="4" width="26" height="12" rx="5" fill="${hair}"/>
      <rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>
      <rect x="10" y="12" width="4" height="20" fill="${hair}"/>
      <rect x="11" y="12" width="2" height="20" fill="${hairLight}" opacity="0.3"/>
      <rect x="34" y="12" width="4" height="20" fill="${hair}"/>
      <rect x="34" y="12" width="2" height="20" fill="${hairLight}" opacity="0.3"/>
      <rect x="10" y="30" width="3" height="2" fill="${hair}" opacity="0.55"/>
      <rect x="35" y="30" width="3" height="2" fill="${hair}" opacity="0.55"/>
      <rect x="13" y="11" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (femStyle === 1) {
      // Ponytail — main cap + tail sweeping right
      hairSVG = `
      <rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/>
      <rect x="14" y="3" width="20" height="5" rx="3" fill="${hair}"/>
      <rect x="34" y="7" width="4" height="4" fill="${hairDark}"/>
      <rect x="36" y="8" width="6" height="3" fill="${hair}"/>
      <rect x="38" y="10" width="5" height="15" fill="${hair}"/>
      <rect x="38" y="10" width="3" height="15" fill="${hairLight}" opacity="0.25"/>
      <rect x="39" y="24" width="4" height="2" fill="${hair}" opacity="0.55"/>
      <rect x="13" y="11" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (femStyle === 2) {
      // Bob cut — shoulder-length, straight-cut bottom
      hairSVG = `
      <rect x="11" y="4" width="26" height="12" rx="5" fill="${hair}"/>
      <rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>
      <rect x="10" y="11" width="5" height="17" fill="${hair}"/>
      <rect x="33" y="11" width="5" height="17" fill="${hair}"/>
      <rect x="10" y="27" width="5" height="2" fill="${hairDark}" opacity="0.5"/>
      <rect x="33" y="27" width="5" height="2" fill="${hairDark}" opacity="0.5"/>
      <rect x="11" y="12" width="2" height="14" fill="${hairLight}" opacity="0.25"/>
      <rect x="35" y="12" width="2" height="14" fill="${hairLight}" opacity="0.25"/>
      <rect x="13" y="11" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (femStyle === 3) {
      // Bun — gathered round bun raised above head
      hairSVG = `
      <rect x="12" y="6" width="24" height="12" rx="5" fill="${hair}"/>
      <rect x="13" y="5" width="22" height="4" rx="2" fill="${hair}"/>
      <rect x="18" y="0" width="12" height="9" rx="4" fill="${hair}"/>
      <rect x="20" y="0" width="8" height="5" rx="3" fill="${hairLight}" opacity="0.35"/>
      <rect x="20" y="1" width="4" height="2" fill="${hairLight}" opacity="0.5"/>
      <rect x="18" y="7" width="12" height="1" fill="${hairDark}" opacity="0.4"/>
      <rect x="13" y="12" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else if (femStyle === 4) {
      // Braids — two braids with alternating segments
      hairSVG = `
      <rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/>
      <rect x="14" y="3" width="20" height="5" rx="3" fill="${hair}"/>
      <rect x="10" y="14" width="4" height="4" fill="${hair}"/>
      <rect x="10" y="18" width="4" height="3" fill="${hairDark}"/>
      <rect x="10" y="21" width="4" height="4" fill="${hair}"/>
      <rect x="10" y="25" width="4" height="3" fill="${hairDark}"/>
      <rect x="10" y="28" width="4" height="3" fill="${hair}"/>
      <rect x="11" y="30" width="2" height="2" fill="${hairDark}" opacity="0.6"/>
      <rect x="34" y="14" width="4" height="4" fill="${hair}"/>
      <rect x="34" y="18" width="4" height="3" fill="${hairDark}"/>
      <rect x="34" y="21" width="4" height="4" fill="${hair}"/>
      <rect x="34" y="25" width="4" height="3" fill="${hairDark}"/>
      <rect x="34" y="28" width="4" height="3" fill="${hair}"/>
      <rect x="35" y="30" width="2" height="2" fill="${hairDark}" opacity="0.6"/>
      <rect x="13" y="11" width="22" height="3" fill="${hair}" opacity="0.3"/>`;
    } else {
      // Long with bangs — bangs across forehead, long side strands
      hairSVG = `
      <rect x="11" y="4" width="26" height="10" rx="4" fill="${hair}"/>
      <rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>
      <rect x="12" y="12" width="24" height="4" fill="${hair}"/>
      <rect x="13" y="13" width="22" height="2" fill="${hairLight}" opacity="0.2"/>
      <rect x="10" y="13" width="4" height="22" fill="${hair}"/>
      <rect x="11" y="13" width="2" height="22" fill="${hairLight}" opacity="0.25"/>
      <rect x="34" y="13" width="4" height="22" fill="${hair}"/>
      <rect x="34" y="13" width="2" height="22" fill="${hairLight}" opacity="0.25"/>
      <rect x="10" y="33" width="3" height="2" fill="${hair}" opacity="0.55"/>
      <rect x="35" y="33" width="3" height="2" fill="${hair}" opacity="0.55"/>`;
    }

    // Optional hair accessories (vary by idx, not all female characters)
    const ACC_COLORS = ['#e85080','#c030a0','#e86020','#5050d0','#30a060'];
    if (idx % 5 === 0) {
      // Small pixel bow (top-right of head)
      const bc = ACC_COLORS[idx % ACC_COLORS.length];
      hairSVG += `
      <rect x="29" y="4" width="3" height="2" fill="${bc}"/>
      <rect x="32" y="3" width="3" height="2" fill="${bc}"/>
      <rect x="29" y="6" width="3" height="2" fill="${bc}"/>
      <rect x="31" y="4" width="2" height="4" fill="${bc}" opacity="0.75"/>`;
    } else if (idx % 7 === 0) {
      // Thin headband across forehead
      const bc = ACC_COLORS[(idx + 2) % ACC_COLORS.length];
      hairSVG += `
      <rect x="12" y="12" width="24" height="2" fill="${bc}" opacity="0.85"/>
      <rect x="13" y="12" width="22" height="1" fill="rgba(255,255,255,0.2)"/>`;
    } else if (idx % 11 === 0) {
      // Hair clip on right side
      hairSVG += `
      <rect x="33" y="10" width="4" height="2" fill="#d0c020"/>
      <rect x="33" y="11" width="4" height="1" fill="#a09018"/>`;
    }
  }

  // ── Archetype clothing overlay (drawn over shirt)
  const shirtDark2 = shadeHex(shirt, -50);
  // Per-archetype accent values (deterministic)
  const neonC  = ['#00ff44','#ff00aa','#00ccff','#ff6600','#cc00ff'][(idx * 7) % 5];
  const capC   = ['#1a1a1a','#8a0000','#00008a','#2a1a2a'][(idx * 5) % 4];
  const vestC  = ['#1e3a6a','#8a1a2a','#2a5a2a','#6a2a6a'][(idx * 5) % 4];
  const bandC  = ['#ff3020','#20a030','#4040c0','#f0b020'][(idx * 7) % 4];
  const glassC = ['#2a2040','#8a2020','#3a2810'][(idx * 3) % 3];
  const catC   = ['#e89040','#1a1a1a','#888888','#e8c8a0'][(idx >> 1) % 4];
  const dogC   = ['#c87040','#d4a858','#e8e0d0'][(idx * 3) % 3];
  const birdC  = ['#f0d020','#4080d0','#40c060'][idx % 3];

  let clothingOverlay = '';
  if (archetype === 'punk') {
    clothingOverlay = `
      <rect x="7"  y="30" width="3"  height="14" fill="#1a1a1a"/>
      <rect x="38" y="30" width="3"  height="14" fill="#1a1a1a"/>
      <rect x="3"  y="31" width="4"  height="4"  fill="#252525"/>
      <rect x="41" y="31" width="4"  height="4"  fill="#252525"/>
      <rect x="22" y="31" width="4"  height="1"  fill="#d0d0d0"/>
      <rect x="21" y="32" width="2"  height="1"  fill="#d0d0d0"/>
      <rect x="23" y="33" width="3"  height="1"  fill="#d0d0d0"/>
      <rect x="21" y="34" width="3"  height="1"  fill="#d0d0d0"/>
      <rect x="22" y="35" width="4"  height="1"  fill="#d0d0d0"/>
      <rect x="14" y="34" width="4"  height="3"  fill="#ff2040" opacity="0.9"/>
      <rect x="15" y="34" width="1"  height="1"  fill="#ffffff" opacity="0.4"/>`;
  } else if (archetype === 'grunge') {
    const s1 = shadeHex(shirt, +12);
    const s2 = shadeHex(shirt, -28);
    clothingOverlay = `
      <rect x="10" y="30" width="28" height="3"  fill="${s1}"/>
      <rect x="10" y="33" width="28" height="3"  fill="${s2}"/>
      <rect x="10" y="36" width="28" height="3"  fill="${s1}"/>
      <rect x="10" y="39" width="28" height="3"  fill="${s2}"/>
      <rect x="10" y="42" width="28" height="3"  fill="${s1}"/>
      <rect x="3"  y="31" width="8"  height="3"  fill="${s1}"/>
      <rect x="3"  y="34" width="8"  height="3"  fill="${s2}"/>
      <rect x="3"  y="37" width="8"  height="3"  fill="${s1}"/>
      <rect x="3"  y="40" width="8"  height="3"  fill="${s2}"/>
      <rect x="37" y="31" width="8"  height="3"  fill="${s1}"/>
      <rect x="37" y="34" width="8"  height="3"  fill="${s2}"/>
      <rect x="37" y="37" width="8"  height="3"  fill="${s1}"/>
      <rect x="37" y="40" width="8"  height="3"  fill="${s2}"/>`;
  } else if (archetype === 'formal') {
    const lapC = shadeHex(shirt, -22);
    clothingOverlay = `
      <rect x="10" y="30" width="6"  height="15" fill="${lapC}"/>
      <rect x="32" y="30" width="6"  height="15" fill="${lapC}"/>
      <rect x="16" y="30" width="2"  height="8"  fill="${lapC}" opacity="0.5"/>
      <rect x="30" y="30" width="2"  height="8"  fill="${lapC}" opacity="0.5"/>
      <rect x="19" y="30" width="10" height="4"  fill="#e8e8e8"/>
      <rect x="20" y="30" width="8"  height="2"  fill="#f4f4f4"/>
      <rect x="22" y="36" width="4"  height="2"  fill="#FFD700"/>
      <rect x="22" y="39" width="4"  height="2"  fill="#FFD700"/>
      <rect x="22" y="42" width="4"  height="2"  fill="#FFD700"/>`;
  } else if (archetype === 'street') {
    const hoodC = shadeHex(shirt, -18);
    clothingOverlay = `
      <rect x="9"  y="22" width="30" height="10" rx="3" fill="${hoodC}"/>
      <rect x="11" y="23" width="26" height="8"  rx="2" fill="${shirt}"/>
      <rect x="9"  y="22" width="5"  height="14" fill="${hoodC}"/>
      <rect x="34" y="22" width="5"  height="14" fill="${hoodC}"/>
      <rect x="7"  y="30" width="3"  height="13" fill="${shirt}"/>
      <rect x="38" y="30" width="3"  height="13" fill="${shirt}"/>
      <rect x="22" y="30" width="1"  height="7"  fill="${hoodC}"/>
      <rect x="25" y="30" width="1"  height="7"  fill="${hoodC}"/>`;
  } else if (archetype === 'casual') {
    clothingOverlay = '';
  } else if (archetype === 'gangster') {
    clothingOverlay = `
      <rect x="8"  y="30" width="2"  height="16" fill="${shirt}"/>
      <rect x="38" y="30" width="2"  height="16" fill="${shirt}"/>
      <rect x="18" y="30" width="12" height="4"  fill="#e8e8e0"/>
      <rect x="19" y="30" width="10" height="2"  fill="#f4f4ec"/>
      <rect x="10" y="33" width="4"  height="10" fill="${shadeHex(shirt,+20)}" opacity="0.4"/>
      <rect x="34" y="33" width="4"  height="10" fill="${shadeHex(shirt,+20)}" opacity="0.4"/>
      <rect x="21" y="35" width="3"  height="5"  fill="${shirtDark2}" opacity="0.45"/>
      <rect x="24" y="35" width="3"  height="5"  fill="${shirtDark2}" opacity="0.45"/>`;
  } else if (archetype === 'nerd') {
    clothingOverlay = `
      <rect x="23" y="32" width="2"  height="2"  fill="${shirtDark2}" opacity="0.6"/>
      <rect x="23" y="35" width="2"  height="2"  fill="${shirtDark2}" opacity="0.6"/>
      <rect x="23" y="38" width="2"  height="2"  fill="${shirtDark2}" opacity="0.6"/>
      <rect x="23" y="41" width="2"  height="2"  fill="${shirtDark2}" opacity="0.6"/>
      <rect x="10" y="43" width="28" height="2"  fill="#4a3020"/>
      <rect x="22" y="43" width="4"  height="2"  fill="#8a6030"/>
      <rect x="19" y="30" width="10" height="3"  fill="${shadeHex(shirt,+30)}" opacity="0.6"/>`;
  } else if (archetype === 'jock') {
    clothingOverlay = `
      <rect x="7"  y="30" width="3"  height="15" fill="${shirt}"/>
      <rect x="38" y="30" width="3"  height="15" fill="${shirt}"/>
      <rect x="20" y="34" width="2"  height="6"  fill="rgba(255,255,255,0.7)"/>
      <rect x="22" y="34" width="2"  height="2"  fill="rgba(255,255,255,0.7)"/>
      <rect x="22" y="38" width="2"  height="2"  fill="rgba(255,255,255,0.7)"/>
      <rect x="26" y="34" width="2"  height="6"  fill="rgba(255,255,255,0.7)"/>
      <rect x="3"  y="38" width="8"  height="3"  fill="#f0f0e0"/>
      <rect x="37" y="38" width="8"  height="3"  fill="#f0f0e0"/>`;
  } else if (archetype === 'emo') {
    clothingOverlay = `
      <rect x="10" y="43" width="28" height="2"  fill="#2a2a2a"/>
      <rect x="13" y="43" width="2"  height="2"  fill="#b0b0b0"/>
      <rect x="18" y="43" width="2"  height="2"  fill="#b0b0b0"/>
      <rect x="23" y="43" width="2"  height="2"  fill="#b0b0b0"/>
      <rect x="28" y="43" width="2"  height="2"  fill="#b0b0b0"/>
      <rect x="33" y="43" width="2"  height="2"  fill="#b0b0b0"/>`;
  } else if (archetype === 'pastel') {
    clothingOverlay = `
      <rect x="8"  y="30" width="4"  height="10" rx="2" fill="${shadeHex(shirt,-12)}"/>
      <rect x="36" y="30" width="4"  height="10" rx="2" fill="${shadeHex(shirt,-12)}"/>
      <rect x="19" y="29" width="5"  height="3"  fill="#ff80a0"/>
      <rect x="24" y="29" width="5"  height="3"  fill="#ffa0b8"/>
      <rect x="22" y="30" width="4"  height="2"  fill="#ffb8c8"/>
      <rect x="20" y="31" width="2"  height="3"  fill="#ff80a0"/>
      <rect x="26" y="31" width="2"  height="3"  fill="#ff80a0"/>`;
  } else if (archetype === 'artsy') {
    clothingOverlay = `
      <rect x="14" y="32" width="3"  height="2"  fill="#ff3030"/>
      <rect x="19" y="36" width="3"  height="2"  fill="#3060ff"/>
      <rect x="22" y="33" width="2"  height="3"  fill="#f0c000"/>
      <rect x="28" y="35" width="3"  height="2"  fill="#30c030"/>
      <rect x="31" y="32" width="2"  height="2"  fill="#ff6000"/>
      <rect x="16" y="40" width="2"  height="2"  fill="#ff00c0"/>
      <rect x="26" y="41" width="3"  height="2"  fill="#00c8ff"/>
      <rect x="33" y="38" width="2"  height="3"  fill="#ff3030"/>
      <rect x="20" y="42" width="2"  height="2"  fill="#f0c000"/>
      <rect x="13" y="38" width="2"  height="2"  fill="#3060ff"/>
      <rect x="4"  y="33" width="3"  height="2"  fill="#ff6000"/>
      <rect x="5"  y="37" width="2"  height="2"  fill="#30c030"/>
      <rect x="40" y="34" width="3"  height="2"  fill="#ff3030"/>
      <rect x="39" y="39" width="2"  height="2"  fill="#f0c000"/>`;
  } else if (archetype === 'techy') {
    clothingOverlay = `
      <rect x="12" y="34" width="24" height="3"  fill="#00a0ff" opacity="0.12"/>
      <rect x="14" y="37" width="20" height="2"  fill="#00a0ff" opacity="0.07"/>
      <rect x="21" y="37" width="4"  height="2"  fill="#00d8ff"/>
      <rect x="22" y="36" width="2"  height="1"  fill="#80ffff" opacity="0.7"/>
      <rect x="3"  y="38" width="5"  height="4"  fill="#1a2a4a"/>
      <rect x="4"  y="39" width="3"  height="2"  fill="#0040a0"/>`;
  } else if (archetype === 'cottagecore') {
    const ridgeC = shadeHex(shirt, -22);
    clothingOverlay = `
      <rect x="10" y="31" width="28" height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="10" y="34" width="28" height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="10" y="37" width="28" height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="10" y="40" width="28" height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="10" y="43" width="28" height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="3"  y="32" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="3"  y="36" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="3"  y="40" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="37" y="32" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="37" y="36" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>
      <rect x="37" y="40" width="8"  height="2"  fill="${ridgeC}" opacity="0.45"/>`;
  } else if (archetype === 'preppy') {
    clothingOverlay = `
      <rect x="10" y="30" width="7"  height="15" fill="${vestC}"/>
      <rect x="31" y="30" width="7"  height="15" fill="${vestC}"/>
      <rect x="17" y="38" width="14" height="7"  fill="${vestC}"/>
      <rect x="17" y="30" width="4"  height="8"  fill="${shirt}" opacity="0.9"/>
      <rect x="27" y="30" width="4"  height="8"  fill="${shirt}" opacity="0.9"/>
      <rect x="18" y="30" width="12" height="3"  fill="#e8e8e8"/>
      <rect x="19" y="30" width="10" height="2"  fill="#f4f4f4"/>`;
  }

  // ── Archetype hat / hair accessory (rendered over hair in head section)
  let hatSVG = '';
  if (archetype === 'punk') {
    // Tall neon mohawk — dramatic silhouette from top of viewbox
    hatSVG = `
      <rect x="21" y="0"  width="6"  height="15" fill="${neonC}"/>
      <rect x="22" y="0"  width="4"  height="11" fill="rgba(255,255,255,0.22)"/>
      <rect x="20" y="7"  width="8"  height="5"  fill="${neonC}"/>
      <rect x="19" y="10" width="10" height="4"  fill="${neonC}"/>`;
  } else if (archetype === 'grunge') {
    // Messy strand across forehead
    hatSVG = `
      <rect x="12" y="12" width="8"  height="2"  fill="${hair}"/>
      <rect x="13" y="14" width="6"  height="2"  fill="${hair}"/>
      <rect x="14" y="16" width="5"  height="2"  fill="${hair}" opacity="0.85"/>`;
  } else if (archetype === 'street') {
    // Beanie with contrasting fold
    const bC  = shadeHex(shirt, +28);
    const bdC = shadeHex(shirt, -15);
    hatSVG = `
      <rect x="10" y="2"  width="28" height="11" rx="3" fill="${bC}"/>
      <rect x="10" y="10" width="28" height="3"  fill="${bdC}"/>
      <rect x="11" y="3"  width="26" height="2"  fill="rgba(255,255,255,0.12)"/>`;
  } else if (archetype === 'gangster') {
    // Flat-brim fitted cap
    hatSVG = `
      <rect x="11" y="7"  width="26" height="8"  rx="2" fill="${capC}"/>
      <rect x="12" y="5"  width="24" height="5"  rx="2" fill="${shadeHex(capC,+18)}"/>
      <rect x="9"  y="13" width="30" height="2"  fill="${capC}"/>
      <rect x="8"  y="14" width="32" height="2"  fill="${shadeHex(capC,-8)}"/>
      <rect x="13" y="7"  width="20" height="1"  fill="rgba(255,255,255,0.07)"/>`;
  } else if (archetype === 'jock') {
    // Athletic headband
    hatSVG = `
      <rect x="11" y="12" width="26" height="4"  fill="${bandC}"/>
      <rect x="12" y="12" width="24" height="2"  fill="${shadeHex(bandC,+20)}" opacity="0.5"/>
      <rect x="12" y="12" width="24" height="1"  fill="rgba(255,255,255,0.2)"/>`;
  } else if (archetype === 'nerd') {
    // Big thick-framed glasses — the defining feature
    hatSVG = `
      <rect x="14" y="17" width="8"  height="5"  fill="none" stroke="${glassC}" stroke-width="1.5"/>
      <rect x="26" y="17" width="8"  height="5"  fill="none" stroke="${glassC}" stroke-width="1.5"/>
      <rect x="22" y="19" width="4"  height="1"  fill="${glassC}"/>
      <rect x="13" y="19" width="2"  height="1"  fill="${glassC}" opacity="0.6"/>
      <rect x="34" y="19" width="2"  height="1"  fill="${glassC}" opacity="0.6"/>
      <rect x="14" y="17" width="1"  height="1"  fill="rgba(255,255,255,0.18)"/>
      <rect x="26" y="17" width="1"  height="1"  fill="rgba(255,255,255,0.18)"/>`;
  } else if (archetype === 'emo') {
    // Side-swept bang covering one eye — the signature
    hatSVG = `
      <rect x="11" y="11" width="11" height="3"  fill="${hair}"/>
      <rect x="12" y="13" width="9"  height="3"  fill="${hair}"/>
      <rect x="13" y="15" width="8"  height="2"  fill="${hair}"/>
      <rect x="14" y="17" width="7"  height="2"  fill="${hair}"/>
      <rect x="15" y="19" width="5"  height="2"  fill="${hair}" opacity="0.75"/>`;
  } else if (archetype === 'pastel') {
    // Flower crown — colorful band across head
    hatSVG = `
      <rect x="11" y="10" width="2"  height="2"  fill="#ff80a0"/>
      <rect x="12" y="9"  width="2"  height="2"  fill="#ffb0c8"/>
      <rect x="16" y="8"  width="2"  height="2"  fill="#ffe060"/>
      <rect x="15" y="9"  width="2"  height="2"  fill="#ffd040"/>
      <rect x="21" y="7"  width="2"  height="2"  fill="#c080ff"/>
      <rect x="20" y="8"  width="2"  height="2"  fill="#d0a0ff"/>
      <rect x="26" y="7"  width="2"  height="2"  fill="#80e0ff"/>
      <rect x="25" y="8"  width="2"  height="2"  fill="#a0d8ff"/>
      <rect x="31" y="8"  width="2"  height="2"  fill="#ff80a0"/>
      <rect x="30" y="9"  width="2"  height="2"  fill="#ffb0c8"/>
      <rect x="35" y="10" width="2"  height="2"  fill="#90ff90"/>
      <rect x="34" y="11" width="2"  height="2"  fill="#70e070"/>`;
  } else if (archetype === 'artsy') {
    // Tilted beret (center at x=30, clearly off-center)
    const beretC = ['#7a3a10','#2a3a8a','#1a6a2a','#6a1a4a'][(idx * 3) % 4];
    const beretL = shadeHex(beretC, +20);
    hatSVG = `
      <ellipse cx="30" cy="5" rx="14" ry="6" fill="${beretC}"/>
      <ellipse cx="30" cy="5" rx="12" ry="4" fill="${beretL}"/>
      <rect x="17" y="7"  width="20" height="3"  fill="${shadeHex(beretC,-10)}"/>
      <rect x="19" y="6"  width="8"  height="2"  fill="rgba(255,255,255,0.1)"/>
      <rect x="28" y="1"  width="3"  height="3"  fill="${beretC}" rx="1"/>`;
  } else if (archetype === 'techy') {
    // Headset — arc over head + earpieces
    hatSVG = `
      <rect x="13" y="7"  width="22" height="2"  fill="#c0c8d0"/>
      <rect x="9"  y="9"  width="5"  height="7"  fill="#c0c8d0"/>
      <rect x="34" y="9"  width="5"  height="7"  fill="#c0c8d0"/>
      <rect x="10" y="11" width="4"  height="4"  fill="#383848"/>
      <rect x="34" y="11" width="4"  height="4"  fill="#383848"/>
      <rect x="10" y="11" width="1"  height="3"  fill="rgba(255,255,255,0.1)"/>
      <rect x="34" y="11" width="1"  height="3"  fill="rgba(255,255,255,0.1)"/>`;
  } else if (archetype === 'cottagecore') {
    // Flower behind ear
    hatSVG = `
      <rect x="33" y="13" width="5"  height="5"  fill="#ff9040"/>
      <rect x="35" y="11" width="3"  height="4"  fill="#ffb050"/>
      <rect x="33" y="16" width="3"  height="2"  fill="#ff7030"/>
      <rect x="36" y="10" width="2"  height="5"  fill="#50a030"/>
      <rect x="35" y="10" width="1"  height="1"  fill="#70c040"/>`;
  } else if (archetype === 'preppy' && isFemale) {
    // Headband
    const hbC = ['#c03060','#3060c0','#30a860','#c09020'][(idx * 3) % 4];
    hatSVG = `
      <rect x="11" y="12" width="26" height="3"  fill="${hbC}"/>
      <rect x="12" y="12" width="24" height="1"  fill="rgba(255,255,255,0.2)"/>`;
  }

  // ── Archetype jewelry (earrings + necklace)
  let arcJewelry = '';
  if (archetype === 'gangster') {
    // HEAVIEST gold — huge chain arc + hoop earrings + ring
    arcJewelry = `
      <rect x="16" y="32" width="3"  height="2"  fill="#FFD700"/>
      <rect x="14" y="33" width="3"  height="2"  fill="#FFD700"/>
      <rect x="16" y="35" width="4"  height="1"  fill="#FFD700"/>
      <rect x="20" y="36" width="4"  height="2"  fill="#FFD700"/>
      <rect x="24" y="37" width="2"  height="2"  fill="#FFD700"/>
      <rect x="26" y="36" width="4"  height="2"  fill="#FFD700"/>
      <rect x="30" y="35" width="4"  height="1"  fill="#FFD700"/>
      <rect x="31" y="33" width="3"  height="2"  fill="#FFD700"/>
      <rect x="29" y="32" width="3"  height="2"  fill="#FFD700"/>
      <rect x="10" y="20" width="2"  height="1"  fill="#FFD700"/>
      <rect x="10" y="21" width="1"  height="4"  fill="#FFD700"/>
      <rect x="11" y="25" width="2"  height="1"  fill="#FFD700"/>
      <rect x="12" y="21" width="1"  height="4"  fill="#FFD700"/>
      <rect x="35" y="20" width="2"  height="1"  fill="#FFD700"/>
      <rect x="35" y="21" width="1"  height="4"  fill="#FFD700"/>
      <rect x="36" y="25" width="2"  height="1"  fill="#FFD700"/>
      <rect x="37" y="21" width="1"  height="4"  fill="#FFD700"/>
      <rect x="3"  y="43" width="4"  height="2"  fill="#FFD700"/>`;
  } else if (archetype === 'preppy') {
    // Pearls — white dots clearly visible on skin
    arcJewelry = isFemale ? `
      <rect x="10" y="20" width="3"  height="3"  fill="#FFFFFF"/>
      <rect x="10" y="22" width="3"  height="2"  fill="#F0F0F0"/>
      <rect x="35" y="20" width="3"  height="3"  fill="#FFFFFF"/>
      <rect x="35" y="22" width="3"  height="2"  fill="#F0F0F0"/>
      <rect x="16" y="31" width="2"  height="2"  fill="#FFFFFF"/>
      <rect x="19" y="30" width="2"  height="2"  fill="#FFFFFF"/>
      <rect x="22" y="30" width="2"  height="2"  fill="#FFFFFF"/>
      <rect x="25" y="30" width="2"  height="2"  fill="#FFFFFF"/>
      <rect x="28" y="30" width="2"  height="2"  fill="#FFFFFF"/>
      <rect x="31" y="31" width="2"  height="2"  fill="#FFFFFF"/>` : `
      <rect x="22" y="30" width="4"  height="2"  fill="#c03060"/>
      <rect x="23" y="29" width="2"  height="4"  fill="#a02050"/>`;
  } else if (archetype === 'pastel') {
    // Dangle earrings + heart pendant
    arcJewelry = isFemale ? `
      <rect x="10" y="21" width="2"  height="2"  fill="#ff80c0"/>
      <rect x="11" y="23" width="1"  height="3"  fill="#e060b0"/>
      <rect x="10" y="26" width="3"  height="2"  fill="#ff90d0"/>
      <rect x="35" y="21" width="2"  height="2"  fill="#c080ff"/>
      <rect x="36" y="23" width="1"  height="3"  fill="#b060e0"/>
      <rect x="35" y="26" width="3"  height="2"  fill="#d090ff"/>
      <rect x="21" y="32" width="2"  height="2"  fill="#ff6090"/>
      <rect x="24" y="32" width="2"  height="2"  fill="#ff6090"/>
      <rect x="20" y="33" width="7"  height="2"  fill="#ff6090"/>
      <rect x="21" y="35" width="5"  height="1"  fill="#ff6090"/>
      <rect x="22" y="36" width="3"  height="1"  fill="#ff6090"/>
      <rect x="22" y="30" width="1"  height="3"  fill="#e080b0"/>
      <rect x="25" y="30" width="1"  height="3"  fill="#e080b0"/>` : `
      <rect x="22" y="31" width="4"  height="1"  fill="#d080c0"/>
      <rect x="21" y="32" width="6"  height="1"  fill="#c070b0"/>`;
  } else if (archetype === 'formal') {
    // Gold studs + chain (female) or tie + watch (male)
    arcJewelry = isFemale ? `
      <rect x="10" y="20" width="3"  height="3"  fill="#FFD700"/>
      <rect x="35" y="20" width="3"  height="3"  fill="#FFD700"/>
      <rect x="19" y="31" width="2"  height="1"  fill="#FFD700"/>
      <rect x="22" y="30" width="4"  height="1"  fill="#FFD700"/>
      <rect x="27" y="31" width="2"  height="1"  fill="#FFD700"/>
      <rect x="3"  y="38" width="5"  height="3"  fill="#FFD700"/>
      <rect x="4"  y="39" width="3"  height="1"  fill="#c8a000"/>` : `
      <rect x="22" y="31" width="4"  height="7"  fill="#8a1a2a"/>
      <rect x="23" y="31" width="2"  height="5"  fill="#a02040"/>
      <rect x="21" y="37" width="6"  height="3"  fill="#8a1a2a"/>
      <rect x="3"  y="38" width="5"  height="3"  fill="#FFD700"/>
      <rect x="4"  y="39" width="3"  height="1"  fill="#c8a000"/>`;
  } else if (archetype === 'street') {
    // Gold chain + hoop earring
    arcJewelry = `
      <rect x="18" y="32" width="3"  height="2"  fill="#FFD700"/>
      <rect x="21" y="33" width="3"  height="2"  fill="#FFD700"/>
      <rect x="24" y="33" width="3"  height="2"  fill="#FFD700"/>
      <rect x="27" y="32" width="3"  height="2"  fill="#FFD700"/>
      <rect x="10" y="21" width="2"  height="2"  fill="#FFD700"/>
      <rect x="10" y="23" width="3"  height="1"  fill="#FFD700"/>`;
  } else if (archetype === 'punk') {
    // Silver safety pin + studded choker
    arcJewelry = `
      <rect x="10" y="21" width="2"  height="1"  fill="#c0c0c0"/>
      <rect x="11" y="22" width="1"  height="4"  fill="#c0c0c0"/>
      <rect x="10" y="26" width="2"  height="1"  fill="#c0c0c0"/>
      <rect x="15" y="29" width="18" height="2"  fill="#2a2a2a"/>
      <rect x="16" y="29" width="2"  height="2"  fill="#c0c0c0"/>
      <rect x="20" y="29" width="2"  height="2"  fill="#c0c0c0"/>
      <rect x="24" y="29" width="2"  height="2"  fill="#c0c0c0"/>
      <rect x="28" y="29" width="2"  height="2"  fill="#c0c0c0"/>
      <rect x="32" y="29" width="2"  height="2"  fill="#c0c0c0"/>`;
  } else if (archetype === 'emo') {
    // Multiple ear piercings stacked + black choker
    arcJewelry = `
      <rect x="10" y="17" width="2"  height="2"  fill="#9040ff"/>
      <rect x="10" y="20" width="2"  height="2"  fill="#ff0040"/>
      <rect x="10" y="23" width="2"  height="2"  fill="#40c0ff"/>
      <rect x="15" y="29" width="18" height="2"  fill="#1a1a1a"/>
      <rect x="21" y="29" width="2"  height="2"  fill="#b0b0b0"/>`;
  } else if (archetype === 'artsy') {
    // Mismatched earrings — different color each side
    arcJewelry = isFemale ? `
      <rect x="10" y="20" width="2"  height="2"  fill="#ff8040"/>
      <rect x="11" y="22" width="1"  height="3"  fill="#ff6020"/>
      <rect x="35" y="20" width="2"  height="2"  fill="#4080ff"/>
      <rect x="36" y="22" width="1"  height="3"  fill="#2060e0"/>
      <rect x="4"  y="43" width="2"  height="1"  fill="#ff3030"/>
      <rect x="42" y="43" width="2"  height="1"  fill="#30c030"/>` : `
      <rect x="10" y="20" width="2"  height="2"  fill="#ff8040"/>
      <rect x="4"  y="43" width="2"  height="1"  fill="#ff3030"/>`;
  } else if (archetype === 'cottagecore') {
    arcJewelry = isFemale ? `
      <rect x="10" y="20" width="3"  height="3"  fill="#ffb890"/>
      <rect x="35" y="20" width="3"  height="3"  fill="#ffb890"/>
      <rect x="21" y="31" width="2"  height="1"  fill="#c09880"/>
      <rect x="24" y="31" width="2"  height="1"  fill="#c09880"/>` : '';
  }

  // ── Archetype desk items
  let leftItems = '', rightItems = '';
  if (archetype === 'punk') {
    leftItems = `
      <rect x="15" y="43" width="3"  height="2"  fill="#ff2040"/>
      <rect x="19" y="44" width="2"  height="1"  fill="#00ff44"/>
      <rect x="22" y="43" width="2"  height="2"  fill="#0080ff"/>`;
    rightItems = `
      <rect x="39" y="41" width="5"  height="9"  fill="#00e040"/>
      <rect x="39" y="41" width="5"  height="2"  fill="#00ff50"/>
      <rect x="40" y="43" width="3"  height="1"  fill="rgba(255,255,255,0.2)"/>
      <rect x="40" y="42" width="1"  height="1"  fill="rgba(255,255,255,0.3)"/>`;
  } else if (archetype === 'grunge') {
    leftItems = `
      <rect x="1"  y="43" width="7"  height="7"  fill="#2a1a0a"/>
      <rect x="1"  y="43" width="7"  height="2"  fill="#3a2818"/>
      <rect x="8"  y="45" width="2"  height="3"  fill="#2a1a0a"/>
      <rect x="2"  y="44" width="2"  height="1"  fill="rgba(0,0,0,0.4)"/>`;
    rightItems = `
      <rect x="37" y="42" width="9"  height="9"  rx="4" fill="#1a1a1a"/>
      <rect x="40" y="45" width="3"  height="3"  rx="1" fill="#3a3a3a"/>
      <rect x="41" y="46" width="1"  height="1"  fill="#555"/>`;
  } else if (archetype === 'formal') {
    leftItems = `
      <rect x="1"  y="42" width="10" height="7"  fill="#6a3a18"/>
      <rect x="2"  y="43" width="8"  height="5"  fill="#7a4a20"/>
      <rect x="2"  y="43" width="8"  height="1"  fill="#9a6030"/>
      <rect x="3"  y="45" width="6"  height="1"  fill="rgba(200,160,60,0.2)"/>`;
    rightItems = `
      <rect x="39" y="42" width="2"  height="8"  fill="#FFD700" transform="rotate(-12 40 46)"/>
      <rect x="38" y="42" width="2"  height="2"  fill="#e8c000" transform="rotate(-12 39 43)"/>`;
  } else if (archetype === 'street') {
    leftItems = `
      <rect x="1"  y="43" width="10" height="4"  rx="2" fill="#2a2a2a"/>
      <rect x="3"  y="42" width="5"  height="3"  fill="#1a1a1a"/>
      <rect x="1"  y="44" width="4"  height="2"  fill="#383838"/>
      <rect x="7"  y="44" width="4"  height="2"  fill="#383838"/>`;
    rightItems = `
      <rect x="39" y="42" width="6"  height="8"  rx="1" fill="#1a1a1a"/>
      <rect x="40" y="43" width="4"  height="5"  fill="#80d0ff"/>
      <rect x="40" y="43" width="4"  height="1"  fill="#c0e8ff"/>`;
  } else if (archetype === 'casual') {
    leftItems = `
      <rect x="1"  y="43" width="7"  height="7"  fill="#a85028"/>
      <rect x="1"  y="43" width="7"  height="2"  fill="#c06030"/>
      <rect x="8"  y="45" width="2"  height="3"  fill="#a85028"/>
      <rect x="2"  y="44" width="2"  height="1"  fill="rgba(255,255,255,0.15)"/>`;
    rightItems = `
      <rect x="38" y="44" width="8"  height="4"  fill="#c09040"/>
      <rect x="38" y="44" width="8"  height="1"  fill="#e0b050"/>`;
  } else if (archetype === 'gangster') {
    leftItems = `
      <rect x="1"  y="44" width="9"  height="5"  fill="#208030"/>
      <rect x="1"  y="44" width="9"  height="2"  fill="#30a040"/>
      <rect x="2"  y="45" width="7"  height="1"  fill="rgba(255,255,255,0.1)"/>`;
    rightItems = `
      <rect x="38" y="44" width="4"  height="3"  fill="#1a1a1a"/>
      <rect x="44" y="44" width="4"  height="3"  fill="#1a1a1a"/>
      <rect x="42" y="45" width="2"  height="1"  fill="#3a3a3a"/>
      <rect x="38" y="47" width="9"  height="3"  fill="#2a2a2a"/>`;
  } else if (archetype === 'nerd') {
    leftItems = `
      <rect x="1"  y="42" width="10" height="8"  fill="#2848a8"/>
      <rect x="2"  y="43" width="8"  height="6"  fill="#3858c0"/>
      <rect x="1"  y="43" width="1"  height="5"  fill="#1a3088"/>
      <rect x="3"  y="44" width="6"  height="1"  fill="rgba(255,255,255,0.15)"/>`;
    rightItems = `
      <rect x="38" y="42" width="9"  height="7"  fill="#c8d0c0"/>
      <rect x="39" y="43" width="7"  height="2"  fill="#a0a898"/>
      <rect x="39" y="46" width="2"  height="1"  fill="#888880"/>
      <rect x="42" y="46" width="2"  height="1"  fill="#888880"/>
      <rect x="45" y="46" width="1"  height="1"  fill="#888880"/>`;
  } else if (archetype === 'jock') {
    leftItems = `
      <rect x="2"  y="41" width="5"  height="9"  rx="1" fill="#2080e0"/>
      <rect x="2"  y="41" width="5"  height="2"  fill="#40a0ff"/>
      <rect x="2"  y="42" width="2"  height="6"  fill="rgba(255,255,255,0.12)"/>`;
    rightItems = `
      <rect x="38" y="44" width="9"  height="5"  fill="#e8e0d0"/>
      <rect x="38" y="44" width="9"  height="2"  fill="#f0e8d8"/>`;
  } else if (archetype === 'emo') {
    leftItems = `
      <rect x="1"  y="42" width="9"  height="7"  fill="#1a1a1a"/>
      <rect x="2"  y="43" width="7"  height="5"  fill="#2a2a2a"/>
      <rect x="4"  y="44" width="2"  height="2"  fill="#c02040"/>
      <rect x="6"  y="44" width="2"  height="2"  fill="#c02040"/>
      <rect x="3"  y="45" width="6"  height="1"  fill="#c02040"/>`;
    rightItems = `
      <rect x="39" y="43" width="7"  height="1"  fill="#2a2a2a"/>
      <rect x="39" y="44" width="3"  height="3"  rx="1" fill="#1a1a1a"/>
      <rect x="44" y="44" width="3"  height="3"  rx="1" fill="#1a1a1a"/>`;
  } else if (archetype === 'pastel') {
    leftItems = `
      <rect x="1"  y="43" width="8"  height="6"  fill="#f8c8d8"/>
      <rect x="2"  y="44" width="3"  height="4"  fill="#e8b0c8"/>
      <rect x="6"  y="43" width="3"  height="5"  fill="#d8a8c0"/>`;
    rightItems = `
      <rect x="37" y="43" width="8"  height="7"  rx="3" fill="#ffb0d0"/>
      <rect x="39" y="44" width="2"  height="2"  fill="#ff80a8"/>
      <rect x="43" y="44" width="2"  height="2"  fill="#ff80a8"/>
      <rect x="39" y="44" width="1"  height="1"  fill="#2a1a1a"/>
      <rect x="43" y="44" width="1"  height="1"  fill="#2a1a1a"/>
      <rect x="40" y="47" width="3"  height="1"  fill="#2a1a1a" opacity="0.4"/>`;
  } else if (archetype === 'artsy') {
    leftItems = `
      <rect x="2"  y="45" width="1"  height="5"  fill="#ff4040" transform="rotate(-20 2 50)"/>
      <rect x="4"  y="44" width="1"  height="5"  fill="#4080ff" transform="rotate(-10 4 49)"/>
      <rect x="6"  y="43" width="1"  height="5"  fill="#40c040" transform="rotate(0  6 48)"/>
      <rect x="8"  y="44" width="1"  height="5"  fill="#f0c000" transform="rotate(10  8 49)"/>
      <rect x="10" y="45" width="1"  height="5"  fill="#ff6000" transform="rotate(20 10 50)"/>`;
    rightItems = `
      <rect x="38" y="42" width="9"  height="7"  fill="#fefefe"/>
      <rect x="38" y="42" width="1"  height="7"  fill="#e0d8c0"/>
      <rect x="39" y="43" width="7"  height="1"  fill="#2060c0" opacity="0.2"/>
      <rect x="39" y="45" width="5"  height="1"  fill="#c02020" opacity="0.15"/>
      <rect x="39" y="47" width="8"  height="3"  rx="1" fill="#d0c0a0"/>
      <rect x="40" y="47" width="1"  height="2"  fill="#ff3030"/>
      <rect x="42" y="47" width="1"  height="2"  fill="#4080ff"/>
      <rect x="44" y="47" width="1"  height="2"  fill="#f0c000"/>`;
  } else if (archetype === 'techy') {
    leftItems = `
      <rect x="1"  y="41" width="12" height="9"  fill="#1a1a2e"/>
      <rect x="2"  y="42" width="10" height="7"  fill="#0a0a18"/>
      <rect x="3"  y="43" width="8"  height="1"  fill="#0060ff" opacity="0.6"/>
      <rect x="3"  y="45" width="5"  height="1"  fill="#0060ff" opacity="0.3"/>`;
    rightItems = `
      <rect x="39" y="41" width="5"  height="9"  fill="#00d0ff"/>
      <rect x="39" y="41" width="5"  height="2"  fill="#80f0ff"/>
      <rect x="40" y="43" width="3"  height="1"  fill="rgba(255,255,255,0.2)"/>
      <rect x="40" y="47" width="7"  height="3"  rx="2" fill="#2a2a3a"/>
      <rect x="42" y="47" width="1"  height="2"  fill="#3a3a4a"/>`;
  } else if (archetype === 'cottagecore') {
    leftItems = `
      <rect x="2"  y="44" width="5"  height="6"  fill="#a87050"/>
      <rect x="1"  y="44" width="7"  height="1"  fill="#c09068"/>
      <rect x="3"  y="42" width="1"  height="3"  fill="#508840"/>
      <rect x="4"  y="41" width="2"  height="4"  fill="#608848"/>
      <rect x="6"  y="42" width="1"  height="3"  fill="#508840"/>
      <rect x="3"  y="43" width="1"  height="1"  fill="#ff9090"/>
      <rect x="6"  y="41" width="1"  height="1"  fill="#ffcc80"/>`;
    rightItems = `
      <rect x="38" y="44" width="8"  height="5"  fill="#e8d0b8"/>
      <rect x="37" y="49" width="10" height="1"  fill="#d0b8a0"/>
      <rect x="38" y="44" width="8"  height="2"  fill="#f0e0c8"/>
      <rect x="46" y="46" width="2"  height="2"  fill="#e8d0b8"/>
      <rect x="43" y="42" width="1"  height="3"  fill="#808060"/>
      <rect x="43" y="41" width="2"  height="2"  fill="#c0b080"/>`;
  } else if (archetype === 'preppy') {
    leftItems = `
      <rect x="1"  y="42" width="10" height="7"  fill="#f4f4f4"/>
      <rect x="2"  y="44" width="8"  height="1"  fill="rgba(100,80,40,0.2)"/>
      <rect x="2"  y="46" width="5"  height="1"  fill="rgba(100,80,40,0.15)"/>
      <rect x="10" y="42" width="1"  height="7"  fill="#c0b8a8"/>`;
    rightItems = `
      <rect x="38" y="46" width="7"  height="5"  fill="#4a7840"/>
      <rect x="38" y="46" width="9"  height="1"  fill="#5a9050"/>
      <rect x="40" y="43" width="4"  height="4"  fill="#508848"/>
      <rect x="41" y="42" width="2"  height="5"  fill="#609050"/>
      <rect x="40" y="42" width="1"  height="7"  fill="#c8a830" transform="rotate(-5 40 46)"/>`;
  }

  // ── Pets (~35% of sims, archetype-appropriate)
  const HAS_PET = new Set([0, 3, 6, 9, 11, 14, 17, 20, 24, 27, 30, 33]);
  if (HAS_PET.has(idx)) {
    if (['pastel','emo','grunge','cottagecore','casual','formal'].includes(archetype)) {
      // Cat
      rightItems = `
        <rect x="38" y="43" width="2"  height="2"  fill="${catC}"/>
        <rect x="43" y="43" width="2"  height="2"  fill="${catC}"/>
        <rect x="37" y="44" width="7"  height="5"  rx="2" fill="${catC}"/>
        <rect x="38" y="46" width="2"  height="1"  fill="#1a1a1a"/>
        <rect x="43" y="46" width="2"  height="1"  fill="#1a1a1a"/>
        <rect x="40" y="47" width="2"  height="1"  fill="#c06080"/>
        <rect x="36" y="48" width="9"  height="5"  rx="1" fill="${catC}"/>
        <rect x="44" y="50" width="4"  height="1"  fill="${catC}"/>
        <rect x="47" y="49" width="1"  height="2"  fill="${catC}"/>`;
    } else if (['street','preppy','gangster','jock'].includes(archetype)) {
      // Dog
      const dd = shadeHex(dogC, -22);
      rightItems = `
        <rect x="36" y="44" width="3"  height="5"  rx="1" fill="${dd}"/>
        <rect x="44" y="44" width="3"  height="5"  rx="1" fill="${dd}"/>
        <rect x="38" y="43" width="7"  height="5"  rx="2" fill="${dogC}"/>
        <rect x="39" y="45" width="2"  height="2"  fill="#1a1a1a"/>
        <rect x="43" y="45" width="2"  height="2"  fill="#1a1a1a"/>
        <rect x="39" y="47" width="5"  height="2"  fill="${shadeHex(dogC,+18)}"/>
        <rect x="41" y="47" width="2"  height="1"  fill="#2a1a1a"/>
        <rect x="37" y="48" width="9"  height="4"  rx="1" fill="${dogC}"/>
        <rect x="45" y="46" width="1"  height="4"  fill="${dogC}"/>`;
    } else if (['artsy','techy'].includes(archetype)) {
      // Bird perched on monitor
      rightItems = `
        <rect x="31" y="37" width="5"  height="4"  rx="2" fill="${birdC}"/>
        <rect x="32" y="35" width="4"  height="3"  rx="1" fill="${birdC}"/>
        <rect x="35" y="36" width="2"  height="1"  fill="#e8a020"/>
        <rect x="33" y="36" width="1"  height="1"  fill="#1a1a1a"/>
        <rect x="30" y="41" width="3"  height="1"  fill="${shadeHex(birdC,-20)}"/>
        <rect x="32" y="41" width="1"  height="2"  fill="#c09020"/>
        <rect x="34" y="41" width="1"  height="2"  fill="#c09020"/>`;
    } else {
      // Plant
      rightItems = `
        <rect x="39" y="47" width="6"  height="4"  fill="#a87050"/>
        <rect x="38" y="47" width="8"  height="1"  fill="#c09068"/>
        <rect x="39" y="44" width="2"  height="3"  fill="#508840"/>
        <rect x="41" y="43" width="3"  height="4"  fill="#608848"/>
        <rect x="43" y="44" width="2"  height="3"  fill="#508840"/>`;
    }
  }

  // ── Desk surface color per archetype
  const DESK_COLORS = {
    punk:        ['#222222','#2e2e2e'],
    grunge:      ['#4A4A2A','#5a5a38'],
    formal:      ['#3A1008','#4a2010'],
    street:      ['#707070','#828282'],
    casual:      ['#B07828','#c88838'],
    gangster:    ['#2A2A2A','#3a3a3a'],
    nerd:        ['#B8B8C8','#c8c8d8'],
    jock:        ['#C09040','#d0a050'],
    emo:         ['#111111','#1e1e1e'],
    pastel:      ['#E8A0B8','#f0b0c8'],
    artsy:       ['#C87040','#d88050'],
    techy:       ['#161630','#202040'],
    cottagecore: ['#90A878','#a0b888'],
    preppy:      ['#D0B888','#e0c898'],
  };
  const dPair       = DESK_COLORS[archetype] || DESK_COLORS.casual;
  const deskSurface  = dPair[0];
  const deskSurface2 = dPair[1];
  const deskGlow     = archetype === 'techy'
    ? `<rect x="0" y="50" width="48" height="2" fill="#00E5FF" opacity="0.25"/>`
    : archetype === 'gangster'
    ? `<rect x="0" y="41" width="48" height="1" fill="#FFD700" opacity="0.35"/>`
    : archetype === 'pastel'
    ? `<rect x="0" y="50" width="48" height="1" fill="#ff80a8" opacity="0.12"/>`
    : '';

  // Build the full character SVG
  return `<svg viewBox="0 0 48 64" width="72" height="96" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">

  <!-- ── CHAIR ── -->
  <rect x="8" y="28" width="32" height="5" fill="#5a2e0c"/>
  <rect x="9" y="29" width="30" height="3" fill="#7a3e18"/>
  <rect x="10" y="30" width="28" height="1" fill="rgba(255,255,255,0.08)"/>

  <!-- ── BODY ── -->
  <rect x="20" y="26" width="8" height="5" fill="${skin}"/>
  <!-- Shirt -->
  <rect x="10" y="30" width="28" height="16" fill="${shirt}"/>
  <rect x="10" y="30" width="28" height="2" fill="${shirtDark}"/>
  <!-- Collar -->
  <rect x="20" y="30" width="8" height="3" fill="${skin}" opacity="0.6"/>
  <!-- Shirt seam -->
  <rect x="23" y="33" width="2" height="10" fill="${shirtDark}" opacity="0.15"/>
  <!-- Left arm -->
  <rect x="3" y="31" width="8" height="12" fill="${shirt}"/>
  <rect x="3" y="31" width="8" height="1" fill="${shirtDark}" opacity="0.4"/>
  <rect x="2" y="42" width="8" height="4" fill="${skin}"/>
  <!-- Right arm -->
  <rect x="37" y="31" width="8" height="12" fill="${shirt}"/>
  <rect x="37" y="31" width="8" height="1" fill="${shirtDark}" opacity="0.4"/>
  <rect x="38" y="42" width="8" height="4" fill="${skin}"/>

  <!-- Body accessory (tie for trend) -->
  ${personality === 'trend' ? accessory : ''}

  <!-- Archetype clothing overlay -->
  ${clothingOverlay}

  <!-- ── DESK ── -->
  <!-- Desk surface with wood grain -->
  <rect x="0" y="41" width="48" height="10" fill="${deskSurface}"/>
  <rect x="0" y="41" width="48" height="1" fill="${deskSurface2}"/>
  <rect x="0" y="44" width="48" height="1" fill="rgba(0,0,0,0.06)"/>
  <rect x="0" y="47" width="48" height="1" fill="rgba(0,0,0,0.04)"/>
  <!-- Grain lines -->
  <rect x="5" y="42" width="38" height="1" fill="rgba(160,100,40,0.12)"/>
  <rect x="8" y="45" width="32" height="1" fill="rgba(160,100,40,0.08)"/>
  <rect x="3" y="48" width="42" height="1" fill="rgba(160,100,40,0.06)"/>
  <!-- Desk front face -->
  <rect x="0" y="51" width="48" height="5" fill="#6a3810"/>
  <rect x="0" y="51" width="48" height="1" fill="#7a4818"/>
  <rect x="0" y="55" width="48" height="1" fill="rgba(0,0,0,0.25)"/>
  ${deskGlow}
  <!-- Desk legs (pixel style) -->
  <rect x="2" y="56" width="4" height="6" fill="#5a3010"/>
  <rect x="42" y="56" width="4" height="6" fill="#5a3010"/>
  <rect x="2" y="56" width="4" height="1" fill="#6a4018"/>

  <!-- ── LAPTOP ── -->
  <rect x="14" y="44" width="20" height="6" fill="#2a2838"/>
  <rect x="15" y="45" width="18" height="4" fill="#363448"/>
  <!-- Keys -->
  <rect x="16" y="46" width="4" height="1" fill="#4a4860"/>
  <rect x="22" y="46" width="5" height="1" fill="#4a4860"/>
  <rect x="29" y="46" width="3" height="1" fill="#4a4860"/>
  <!-- Screen lid -->
  <rect x="14" y="36" width="20" height="9" fill="#1e1c2c"/>
  <rect x="15" y="37" width="18" height="7" fill="${screenBg}"/>
  ${screenGlow}
  <!-- Screen content (chart-like lines) -->
  <rect x="17" y="38" width="4" height="1" fill="${screenFg}" opacity="0.8"/>
  <rect x="22" y="38" width="3" height="1" fill="${screenFg}" opacity="0.5"/>
  <rect x="17" y="40" width="8" height="1" fill="${screenFg}" opacity="0.6"/>
  <rect x="27" y="40" width="4" height="1" fill="${screenFg}" opacity="0.4"/>
  <rect x="17" y="42" width="5" height="1" fill="${screenFg}" opacity="0.35"/>
  <!-- Hinge -->
  <rect x="14" y="44" width="20" height="1" fill="#141228" opacity="0.6"/>

  <!-- Desk items -->
  ${leftItems}
  ${rightItems}

  <!-- ── HEAD ── -->
  ${hairSVG}
  <!-- Face -->
  <rect x="13" y="10" width="22" height="18" rx="4" fill="${skin}"/>
  <!-- Ears -->
  <rect x="11" y="15" width="3" height="5" fill="${skin}"/>
  <rect x="34" y="15" width="3" height="5" fill="${skin}"/>
  <rect x="11" y="16" width="1" height="3" fill="rgba(0,0,0,0.06)"/>
  <rect x="36" y="16" width="1" height="3" fill="rgba(0,0,0,0.06)"/>

  <!-- Eyes & mouth -->
  ${eyes}
  ${mouth}

  <!-- Nose (pixel dot) -->
  <rect x="23" y="21" width="2" height="2" fill="rgba(0,0,0,0.08)"/>

  <!-- Cheek blush -->
  <rect x="15" y="22" width="3" height="2" fill="#e8a088" opacity="${mood === 'happy' ? '0.4' : '0.15'}"/>
  <rect x="30" y="22" width="3" height="2" fill="#e8a088" opacity="${mood === 'happy' ? '0.4' : '0.15'}"/>

  <!-- Head accessory (glasses, headband) -->
  ${personality !== 'trend' ? accessory : ''}

  <!-- Archetype hat / hair piece -->
  ${hatSVG}

  <!-- Archetype jewelry (earrings, necklace) -->
  ${arcJewelry}

  <!-- ── SHADOW ── -->
  <ellipse cx="24" cy="62" rx="18" ry="2" fill="rgba(0,0,0,0.12)"/>
</svg>`;
}

// ─────────────────────────────────────────────── SLEEPING (bed scene) SVG
function sleepingSVG(idx) {
  const shirt = SHIRT_COLORS[idx % SHIRT_COLORS.length];
  const skin  = SKIN_TONES[(idx * 3) % SKIN_TONES.length];

  const MALE_IDXS   = new Set([1, 5, 9, 13, 18, 23, 28, 33]);
  const isFemale    = !MALE_IDXS.has(idx);
  const FEMALE_HAIR = ['#181008','#3a1a08','#4a2810','#7a3818','#b08028',
                       '#c89840','#a83818','#585858','#7b3f8c','#2255aa'];
  const MALE_HAIR   = ['#181008','#4a2810','#b08028','#a83818','#585858','#0e0e0e'];
  const hair = isFemale ? FEMALE_HAIR[(idx * 3 + 1) % FEMALE_HAIR.length]
                        : MALE_HAIR[(idx * 2) % MALE_HAIR.length];

  const shadeHex = (hex, d) => hex.replace(/[0-9a-f]{2}/gi, h =>
    Math.min(255, Math.max(0, parseInt(h, 16) + d)).toString(16).padStart(2, '0'));
  const hairDark    = shadeHex(hair, -28);
  const hairLight   = shadeHex(hair, +22);
  const sleepArch   = SIM_STYLES[idx] || 'casual';
  const blanketBase = ARCHETYPE_BLANKETS[sleepArch] || shirt;
  const blanketFold = shadeHex(blanketBase, +28);  // lighter fold at top
  const blanketDark = shadeHex(blanketBase, -40);  // stripe lines

  // Hair cap only — no long strands (head resting on pillow, strands tucked)
  let hairCap = '';
  if (!isFemale) {
    const s = idx % 4;
    if      (s === 0) hairCap = `<rect x="12" y="4" width="24" height="10" rx="4" fill="${hair}"/><rect x="14" y="3" width="20" height="4" rx="3" fill="${hair}"/>`;
    else if (s === 1) hairCap = `<rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/><rect x="11" y="6" width="12" height="6" rx="2" fill="${hair}"/><rect x="14" y="3" width="18" height="4" rx="3" fill="${hair}"/>`;
    else if (s === 2) hairCap = `<rect x="12" y="5" width="24" height="10" rx="4" fill="${hair}"/><rect x="16" y="2" width="4" height="5" rx="1" fill="${hair}"/><rect x="22" y="1" width="4" height="6" rx="1" fill="${hair}"/><rect x="28" y="3" width="3" height="4" rx="1" fill="${hair}"/>`;
    else              hairCap = `<rect x="11" y="4" width="26" height="14" rx="6" fill="${hair}"/><ellipse cx="24" cy="5" rx="12" ry="5" fill="${hair}"/>`;
  } else {
    const s = idx % 6;
    if      (s === 0) hairCap = `<rect x="11" y="4" width="26" height="12" rx="5" fill="${hair}"/><rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>`;
    else if (s === 1) hairCap = `<rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/><rect x="34" y="7" width="4" height="4" fill="${hairDark}"/><rect x="36" y="8" width="8" height="3" fill="${hair}"/>`;
    else if (s === 2) hairCap = `<rect x="11" y="4" width="26" height="12" rx="5" fill="${hair}"/><rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>`;
    else if (s === 3) hairCap = `<rect x="12" y="6" width="24" height="12" rx="5" fill="${hair}"/><rect x="18" y="0" width="12" height="9" rx="4" fill="${hair}"/><rect x="20" y="0" width="8" height="5" rx="3" fill="${hairLight}" opacity="0.35"/>`;
    else if (s === 4) hairCap = `<rect x="12" y="4" width="24" height="12" rx="5" fill="${hair}"/><rect x="14" y="3" width="20" height="5" rx="3" fill="${hair}"/>`;
    else              hairCap = `<rect x="11" y="4" width="26" height="10" rx="4" fill="${hair}"/><rect x="13" y="3" width="22" height="5" rx="3" fill="${hair}"/>`;
  }

  return `<svg viewBox="0 0 48 64" width="72" height="96" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">

  <!-- ── HEADBOARD (left, tall wood plank) ── -->
  <rect x="0" y="22" width="8" height="34" fill="#5a2e0c"/>
  <rect x="1" y="23" width="5" height="32" fill="#7a3e18"/>
  <rect x="1" y="23" width="2" height="32" fill="rgba(255,255,255,0.07)"/>
  <rect x="0" y="20" width="8" height="3" fill="#8a4820"/>
  <rect x="1" y="20" width="6" height="1" fill="#9a5828"/>

  <!-- ── FOOTBOARD (right) ── -->
  <rect x="40" y="28" width="8" height="28" fill="#5a2e0c"/>
  <rect x="41" y="29" width="5" height="26" fill="#7a3e18"/>
  <rect x="41" y="29" width="2" height="26" fill="rgba(255,255,255,0.05)"/>

  <!-- ── MATTRESS ── -->
  <rect x="8" y="28" width="32" height="28" fill="#ede8d8"/>
  <rect x="8" y="28" width="32" height="1" fill="#f8f4e8"/>
  <rect x="8" y="28" width="1" height="28" fill="rgba(255,255,255,0.18)"/>

  <!-- ── PILLOW (under head) ── -->
  <rect x="8" y="26" width="20" height="12" rx="1" fill="#f4f0e6"/>
  <rect x="9" y="27" width="18" height="10" fill="#fefefc"/>
  <rect x="9" y="27" width="16" height="1" fill="rgba(255,255,255,0.7)"/>
  <rect x="9" y="36" width="18" height="1" fill="rgba(0,0,0,0.06)"/>

  <!-- ── BLANKET ── -->
  <!-- Top fold (lighter, visible above blanket body) -->
  <rect x="8" y="36" width="32" height="5" fill="${blanketFold}"/>
  <rect x="8" y="36" width="32" height="1" fill="rgba(255,255,255,0.22)"/>
  <!-- Main blanket body -->
  <rect x="8" y="40" width="32" height="16" fill="${blanketBase}"/>
  <!-- Horizontal stripe texture -->
  <rect x="8" y="42" width="32" height="1" fill="${blanketDark}" opacity="0.2"/>
  <rect x="8" y="45" width="32" height="1" fill="${blanketDark}" opacity="0.2"/>
  <rect x="8" y="48" width="32" height="1" fill="${blanketDark}" opacity="0.2"/>
  <rect x="8" y="51" width="32" height="1" fill="${blanketDark}" opacity="0.2"/>
  <rect x="8" y="54" width="32" height="1" fill="${blanketDark}" opacity="0.2"/>
  ${sleepArch === 'grunge' ? `
  <rect x="8" y="40" width="32" height="3" fill="${shadeHex(blanketBase,+18)}"/>
  <rect x="8" y="43" width="32" height="3" fill="${shadeHex(blanketBase,-22)}"/>
  <rect x="8" y="46" width="32" height="3" fill="${shadeHex(blanketBase,+18)}"/>
  <rect x="8" y="49" width="32" height="3" fill="${shadeHex(blanketBase,-22)}"/>
  <rect x="8" y="52" width="32" height="3" fill="${shadeHex(blanketBase,+18)}"/>` : ''}
  ${sleepArch === 'pastel' ? `
  <rect x="9"  y="28" width="7" height="7" rx="3" fill="#ffb0d0"/>
  <rect x="11" y="29" width="2" height="2" fill="#ff80a8"/>
  <rect x="14" y="29" width="2" height="2" fill="#ff80a8"/>
  <rect x="11" y="29" width="1" height="1" fill="#2a1a1a"/>
  <rect x="14" y="29" width="1" height="1" fill="#2a1a1a"/>
  <rect x="11" y="31" width="4" height="1" fill="#2a1a1a" opacity="0.3"/>` : ''}
  <!-- Blanket right-edge shadow -->
  <rect x="39" y="36" width="1" height="20" fill="rgba(0,0,0,0.09)"/>

  <!-- ── BED FRONT FACE ── -->
  <rect x="0" y="56" width="48" height="4" fill="#6a3810"/>
  <rect x="0" y="56" width="48" height="1" fill="#7a4818"/>
  <rect x="0" y="59" width="48" height="1" fill="rgba(0,0,0,0.22)"/>

  <!-- ── BED LEGS ── -->
  <rect x="2" y="60" width="4" height="4" fill="#5a3010"/>
  <rect x="42" y="60" width="4" height="4" fill="#5a3010"/>
  <rect x="2" y="60" width="4" height="1" fill="#6a4018"/>
  <rect x="42" y="60" width="4" height="1" fill="#6a4018"/>

  <!-- ── CHARACTER HEAD (resting on pillow, facing forward) ── -->
  ${hairCap}
  <!-- Face -->
  <rect x="13" y="10" width="22" height="18" rx="4" fill="${skin}"/>
  <!-- Ears -->
  <rect x="11" y="15" width="3" height="5" fill="${skin}"/>
  <rect x="34" y="15" width="3" height="5" fill="${skin}"/>
  <!-- CLOSED EYES — horizontal bars replacing open squares -->
  <rect x="17" y="19" width="5" height="1" rx="0" fill="#2a2018" opacity="0.82"/>
  <rect x="26" y="19" width="5" height="1" rx="0" fill="#2a2018" opacity="0.82"/>
  <!-- Relaxed sleeping mouth -->
  <rect x="21" y="24" width="6" height="1" fill="#2a2018" opacity="0.22"/>
  <!-- Nose -->
  <rect x="23" y="21" width="2" height="2" fill="rgba(0,0,0,0.07)"/>
  <!-- Soft cheek blush -->
  <rect x="15" y="22" width="3" height="2" fill="#e8a088" opacity="0.28"/>
  <rect x="30" y="22" width="3" height="2" fill="#e8a088" opacity="0.28"/>

  <!-- ── ZZZ (pixel art, ascending size upper-right, animated) ── -->
  <g class="zzz-float">
    <!-- small z at x=28,y=10 (3px wide) -->
    <rect x="28" y="8"  width="3" height="1" fill="rgba(210,235,210,0.7)"/>
    <rect x="29" y="9"  width="2" height="1" fill="rgba(210,235,210,0.7)"/>
    <rect x="28" y="10" width="3" height="1" fill="rgba(210,235,210,0.7)"/>
    <!-- medium z at x=33,y=5 (4px wide) -->
    <rect x="33" y="4"  width="4" height="1" fill="rgba(210,235,210,0.78)"/>
    <rect x="35" y="5"  width="2" height="1" fill="rgba(210,235,210,0.78)"/>
    <rect x="33" y="6"  width="4" height="1" fill="rgba(210,235,210,0.78)"/>
    <!-- large Z at x=38,y=1 (5px wide) -->
    <rect x="38" y="1"  width="5" height="1" fill="rgba(210,235,210,0.86)"/>
    <rect x="40" y="2"  width="3" height="1" fill="rgba(210,235,210,0.86)"/>
    <rect x="38" y="3"  width="5" height="1" fill="rgba(210,235,210,0.86)"/>
  </g>

  <!-- ── SHADOW ── -->
  <ellipse cx="24" cy="63" rx="18" ry="1.5" fill="rgba(0,0,0,0.09)"/>
</svg>`;
}

function shortName(mode) {
  const map = {
    MEAN_REVERSION: 'Mean Rev',
    BREAKOUT: 'Breakout',
    TREND_PULLBACK: 'Trend PB',
    SWING_TREND: 'Swing',
    OPPORTUNITY: 'Oppty',
    ORB_BREAKOUT: 'ORB',
    VWAP_REVERSION: 'VWAP Rev',
    ZSCORE_BOUNCE: 'ZScore',
    FAILED_BREAKOUT_REVERSAL: 'Fail-Rev',
    VWAP_CONTINUATION: 'VWAP Cont',
    OPENING_DRIVE: 'Open Dr',
    AFTERNOON_BREAKOUT: 'PM Break',
    TREND_RECLAIM: 'Reclaim',
    EXTREME_EXTENSION_FADE: 'Ext Fade',
  };
  return map[mode] || mode.replace(/_/g, ' ').toLowerCase();
}

// ─────────────────────────────────────────────── MULTI-SYMBOL CHARTS

// Convert any timestamp to a "fake UTC" epoch whose UTC wall-clock values equal ET (America/New_York).
// ApexCharts renders x-axis in UTC mode, so labels show the correct ET hour:minute.
function toETMs(t) {
  const d = new Date(typeof t === 'number' ? t : t);
  const fmt = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
  const p = {};
  fmt.formatToParts(d).forEach(({ type, value }) => { p[type] = value; });
  const h = p.hour === '24' ? 0 : +p.hour;
  return Date.UTC(+p.year, +p.month - 1, +p.day, h, +p.minute, +p.second);
}

function _makeApexOptions(sym, height) {
  return {
    chart: {
      type: 'line',
      height,
      background: 'transparent',
      toolbar: { show: false },
      animations: { enabled: false },
      foreColor: '#99bb99',
      sparkline: { enabled: false },
    },
    series: [
      { name: 'price',      type: 'candlestick', data: [] },
      { name: 'pred_cur',   type: 'line',        data: [] },
      { name: 'pred_prev',  type: 'line',        data: [] },
    ],
    stroke: { width: [1, 1.5, 1.5], dashArray: [0, 4, 4], curve: 'straight' },
    colors: ['transparent', '#4499ff', '#aa66ff'],
    xaxis: {
      type: 'datetime',
      labels: {
        datetimeUTC: true,
        style: { colors: '#88aa88', fontSize: '9px' },
        datetimeFormatter: { hour: 'HH:mm', minute: 'HH:mm' },
      },
      axisBorder: { color: 'rgba(160,210,140,0.15)' },
      axisTicks:  { color: 'rgba(160,210,140,0.15)' },
    },
    yaxis: {
      tooltip: { enabled: false },
      labels: {
        style: { colors: '#88aa88', fontSize: '9px' },
        formatter: v => v ? v.toFixed(2) : '',
      },
    },
    grid: {
      borderColor: 'rgba(140,200,120,0.1)',
      strokeDashArray: 4,
      padding: { left: 4, right: 4, top: 0, bottom: 0 },
    },
    plotOptions: {
      candlestick: {
        colors: { upward: '#88ee88', downward: '#ee8888' },
        wick: { useFillColor: true },
      },
    },
    markers: {
      size: [0, 5, 5],
      strokeWidth: 0,
      hover: { size: 6 },
    },
    legend: { show: false },
    tooltip: {
      theme: 'dark',
      x: { format: 'HH:mm' },
      shared: true,
      intersect: false,
      custom: ({ dataPointIndex, w }) => {
        // OHLCV from candle series (series 0)
        const cpt = w.config.series[0]?.data?.[dataPointIndex];
        let html = '';
        if (cpt && Array.isArray(cpt.y) && cpt.y.length >= 4) {
          const [o, h, l, c] = cpt.y;
          const up = c >= o;
          const cc = up ? '#88ee88' : '#ee8888';
          const _td = new Date(cpt.x);
          const time = String(_td.getUTCHours()).padStart(2,'0') + ':' + String(_td.getUTCMinutes()).padStart(2,'0');
          html += `<div style="padding:6px 10px 4px;font-size:10px;font-family:monospace;line-height:1.7">
            <div style="color:#88aa88;font-size:9px;margin-bottom:3px">${time}</div>
            <div><span style="color:#667766;display:inline-block;width:10px">O</span> <b style="color:${cc}">${o.toFixed(2)}</b></div>
            <div><span style="color:#667766;display:inline-block;width:10px">H</span> <b style="color:#88ee88">${h.toFixed(2)}</b></div>
            <div><span style="color:#667766;display:inline-block;width:10px">L</span> <b style="color:#ee8888">${l.toFixed(2)}</b></div>
            <div><span style="color:#667766;display:inline-block;width:10px">C</span> <b style="color:${cc}">${c.toFixed(2)}</b></div>
          </div>`;
        }
        // Check if hovered x falls within a prediction segment
        if (cpt) {
          const cx = cpt.x;
          [[1, 'Pred (cur)', '#4499ff'], [2, 'Pred (prev)', '#aa66ff']].forEach(([si, label, color]) => {
            const pd = w.config.series[si]?.data || [];
            if (pd.length === 2 && cx >= pd[0].x && cx <= pd[1].x) {
              html += `<div style="padding:3px 10px 5px;font-size:10px;border-top:1px solid rgba(255,255,255,0.08)">${label}: <b style="color:${color}">$${parseFloat(pd[0].y).toFixed(2)}</b></div>`;
            }
          });
        }
        return html || '<div style="padding:4px 8px;font-size:10px;color:#88aa88">—</div>';
      },
    },
  };
}

function focusSymbol(sym) {
  if (_focusedSym === sym) { unfocusSymbol(); return; }
  _focusedSym = sym;
  const grid = document.getElementById('symbol-charts-grid');
  if (grid) grid.classList.add('sym-focused-mode');
  document.querySelectorAll('.sym-chart-card').forEach(card => {
    card.classList.toggle('sym-focused', card.id === `sym-card-${sym}`);
  });
  const allBtn = document.getElementById('sym-all-btn');
  if (allBtn) allBtn.classList.remove('hidden');
  const titleEl = document.getElementById('chalk-overview-title');
  if (titleEl) titleEl.textContent = `${sym} · 1-MIN · LAST 60 BARS`;
  if (symbolCharts[sym]) symbolCharts[sym].updateOptions({ chart: { height: FOCUSED_CHART_H } }, false, false);
}

function unfocusSymbol() {
  const prev = _focusedSym;
  _focusedSym = null;
  const grid = document.getElementById('symbol-charts-grid');
  if (grid) grid.classList.remove('sym-focused-mode');
  document.querySelectorAll('.sym-chart-card').forEach(card => card.classList.remove('sym-focused'));
  const allBtn = document.getElementById('sym-all-btn');
  if (allBtn) allBtn.classList.add('hidden');
  const titleEl = document.getElementById('chalk-overview-title');
  if (titleEl) titleEl.textContent = 'MARKET OVERVIEW · 1-MIN · LAST 60 BARS';
  const chartH = window.innerWidth <= 480 ? 90 : window.innerWidth <= 900 ? 110 : 130;
  if (prev && symbolCharts[prev]) symbolCharts[prev].updateOptions({ chart: { height: chartH } }, false, false);
}

async function initSymbolCharts() {
  // Load symbol registry
  let registry = {};
  try {
    const r = await fetch('/api/symbols');
    if (r.ok) registry = await r.json();
  } catch {}
  _symbolRegistryCache = registry;

  const symbols = Object.keys(registry);
  if (!symbols.length) {
    // Fallback: just SPY
    symbols.push('SPY');
    registry['SPY'] = {};
  }

  const grid = document.getElementById('symbol-charts-grid');
  if (!grid) return;

  // Show shimmer skeletons while loading
  grid.innerHTML = '';
  const skeletonCount = symbols.length || 4;
  for (let i = 0; i < skeletonCount; i++) {
    const sk = document.createElement('div');
    sk.className = 'sym-chart-card';
    sk.innerHTML = `<div class="sym-card-header"><div class="skeleton" style="width:44px;height:14px;border-radius:3px"></div></div>
                    <div class="skeleton sym-chart-skeleton"></div>`;
    grid.appendChild(sk);
  }
  // Small delay so shimmer is visible before ApexCharts renders
  await new Promise(r => setTimeout(r, 60));
  grid.innerHTML = '';

  const chartH = window.innerWidth <= 480 ? 90 : window.innerWidth <= 900 ? 110 : 130;

  symbols.forEach(sym => {
    const card = document.createElement('div');
    card.className = 'sym-chart-card';
    card.id = `sym-card-${sym}`;
    card.innerHTML = `
      <div class="sym-card-header" onclick="focusSymbol('${sym}')">
        <span class="sym-badge sym-${sym} sym-card-name">${sym}</span>
        <span class="sym-card-price" id="sym-price-${sym}">—</span>
        <span class="sym-card-change" id="sym-change-${sym}"></span>
        <div class="sym-card-pred" id="sym-pred-${sym}"></div>
      </div>
      <div id="sym-apex-${sym}"></div>
    `;
    grid.appendChild(card);

    const chart = new ApexCharts(
      document.getElementById(`sym-apex-${sym}`),
      _makeApexOptions(sym, chartH)
    );
    chart.render();
    symbolCharts[sym] = chart;
  });
}

async function fetchChartAndPredictions() {
  const symbols = Object.keys(_symbolRegistryCache || {});
  if (!symbols.length) return;

  try {
    const fetches = symbols.flatMap(sym => [
      fetch(`/api/chart?symbol=${sym}&bars=60`).then(r => r.json()).catch(() => ({ candles: [], symbol: sym })),
      fetch(`/api/predictions?symbol=${sym}`).then(r => r.json()).catch(() => ({ predictions: [], latest: null })),
    ]);
    const results = await Promise.all(fetches);

    for (let i = 0; i < symbols.length; i++) {
      const sym      = symbols[i];
      const chartData = results[i * 2];
      const predData  = results[i * 2 + 1];
      _updateSymbolCard(sym, chartData.candles || [], predData.predictions || []);
    }
  } catch (e) {
    console.warn('symbol chart fetch error', e);
  }
}

function _updateSymbolCard(sym, candles, preds) {
  const chart = symbolCharts[sym];
  if (!chart) return;

  const PRED_DUR = 30 * 60 * 1000; // 30 min in ms

  // Only show a prediction after its 30-min window has elapsed
  const THIRTY_MIN_MS = 30 * 60 * 1000;
  const now = Date.now();
  const sorted = [...(preds || [])].sort((a, b) => new Date(b.time) - new Date(a.time));
  // latestPred = most recent prediction that is ≥30 min old (completed window)
  const latestPred = sorted.find(p => (now - new Date(p.time).getTime()) >= THIRTY_MIN_MS) || null;
  // prevPred = most recent completed prediction that is ≥30 min older than latestPred
  const prevPred = latestPred
    ? (sorted.find(p => new Date(latestPred.time) - new Date(p.time) >= THIRTY_MIN_MS) || null)
    : null;

  function predSegment(p) {
    if (!p) return [];
    const dir = (p.direction || '').toUpperCase();
    const priceVal = dir === 'BULLISH' ? p.high : dir === 'BEARISH' ? p.low : null;
    if (priceVal == null) return [];
    const t0 = toETMs(p.time);
    const t1 = t0 + PRED_DUR;
    const y = parseFloat(priceVal);
    return [{ x: t0, y }, { x: t1, y }];
  }

  const candleSeries = candles.map(c => ({ x: toETMs(c.t), y: [c.o, c.h, c.l, c.c] }));
  const curSeg  = predSegment(latestPred);
  const prevSeg = predSegment(prevPred);

  chart.updateSeries([
    { name: 'price',     type: 'candlestick', data: candleSeries },
    { name: 'pred_cur',  type: 'line',        data: curSeg },
    { name: 'pred_prev', type: 'line',        data: prevSeg },
  ], true);

  if (candles.length) {
    const last  = candles[candles.length - 1];
    const first = candles[0];
    const change    = last.c - first.o;
    const changePct = (change / first.o * 100).toFixed(2);
    const sign      = change >= 0 ? '+' : '';
    const priceEl  = document.getElementById(`sym-price-${sym}`);
    const changeEl = document.getElementById(`sym-change-${sym}`);
    if (priceEl)  { priceEl.textContent = `$${last.c.toFixed(2)}`; priceEl.style.color = change >= 0 ? 'var(--green)' : 'var(--red)'; }
    if (changeEl) { changeEl.textContent = `${sign}${changePct}%`; changeEl.style.color = change >= 0 ? 'var(--green)' : 'var(--red)'; }
  }

  // Update prediction badge (uses latest prediction)
  const predEl = document.getElementById(`sym-pred-${sym}`);
  if (latestPred) {
    const dir    = (latestPred.direction || '').toUpperCase();
    const conf   = latestPred.confidence != null ? `${(latestPred.confidence * 100).toFixed(0)}%` : '';
    const isBull = dir === 'BULLISH';
    const isBear = dir === 'BEARISH';
    const dirClass = isBull ? 'up' : isBear ? 'down' : 'flat';
    const arrow    = isBull ? '▲' : isBear ? '▼' : '—';
    const priceVal = isBull ? latestPred.high : isBear ? latestPred.low : null;
    const priceStr = priceVal != null ? ` $${parseFloat(priceVal).toFixed(2)}` : '';
    if (predEl) predEl.innerHTML = `<span class="sym-pred-dir ${dirClass}">${arrow}${priceStr} <span class="pred-conf">${conf}</span></span>`;
  } else {
    if (predEl) predEl.innerHTML = '';
  }
}

// ─────────────────────────────────────────────── DRAWER
async function openDrawer(simId) {
  // Mark previously selected, deselect old
  document.querySelectorAll('.seat.selected').forEach(el => el.classList.remove('selected'));
  const newSeat = document.querySelector(`.seat[data-sim-id="${simId}"]`);
  if (newSeat) newSeat.classList.add('selected');

  currentSimId = simId;

  const drawer   = document.getElementById('sim-drawer');
  const backdrop = document.getElementById('drawer-backdrop');
  drawer.classList.add('open');
  backdrop.classList.add('visible');

  // Reset to overview tab
  switchDrawerTab('overview', drawer.querySelector('.dtab'));

  // Loading state
  document.getElementById('drawer-sim-id').textContent = simId;
  document.getElementById('drawer-sim-name').textContent = '…';
  document.getElementById('drawer-signal-badge').innerHTML = '';
  document.getElementById('dpanel-overview').innerHTML =
    '<div style="padding:20px;text-align:center;color:var(--text-muted)">Loading…</div>';

  try {
    const r = await fetch(`/api/sim/${simId}`);
    const d = await r.json();
    populateDrawer(d);
  } catch (e) {
    document.getElementById('dpanel-overview').innerHTML =
      '<div style="padding:16px;color:var(--loss-text)">Failed to load sim data</div>';
  }
}

function closeDrawer() {
  document.getElementById('sim-drawer').classList.remove('open');
  document.getElementById('drawer-backdrop').classList.remove('visible');
  document.querySelectorAll('.seat.selected').forEach(el => el.classList.remove('selected'));
  if (perfChart) { perfChart.destroy(); perfChart = null; }
  currentSimId = null;
}

function switchDrawerTab(name, btn) {
  document.querySelectorAll('.drawer-panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.dtab').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('dpanel-' + name);
  if (panel) panel.classList.remove('hidden');
  if (btn) btn.classList.add('active');
  // Render perf chart lazily when that tab is opened
  if (name === 'overview' && perfChart === null && currentSimId) {
    setTimeout(() => renderWinRateChart(document._drawerWinRateChart || { runs: [] }), 50);
  }
}

function populateDrawer(d) {
  const stats   = d.stats || {};
  const profile = d.profile || {};

  const pnlCls  = stats.pnl_dollars > 0 ? 'pos' : stats.pnl_dollars < 0 ? 'neg' : '';
  const pnlSign = stats.pnl_dollars > 0 ? '+' : '';

  // Header
  document.getElementById('drawer-sim-id').textContent   = d.sim_id;
  document.getElementById('drawer-sim-name').textContent = d.name || '';
  document.getElementById('drawer-signal-badge').innerHTML =
    `<span class="signal-badge">${stats.signal_mode || '—'}</span>`;

  // Store for lazy chart render
  document._drawerBalanceHistory = d.balance_history || [];
  document._drawerBalanceStart   = stats.balance_start;
  document._drawerWinRateChart   = d.win_rate_chart || { runs: [] };

  // ── OVERVIEW TAB ──
  const ov = document.getElementById('dpanel-overview');

  const sess = stats.session || {};
  const sessPnlCls = sess.pnl > 0 ? 'pos' : sess.pnl < 0 ? 'neg' : '';
  const sessPnlSign = sess.pnl > 0 ? '+' : '';
  const sessSection = `
    <div class="session-section">
      <div class="session-label">TODAY'S SESSION</div>
      <div class="drawer-stats-grid session-grid">
        ${dStat('Trades',   (sess.trades ?? 0) + (sess.open ? ` <span class="sess-open">(${sess.open} open)</span>` : ''), '', '')}
        ${dStat('P&L',      sess.trades > 0 || sess.open > 0 ? sessPnlSign + '$' + fmt2(sess.pnl) : '—', '', sessPnlCls)}
        ${dStat('Win Rate', sess.win_rate != null ? sess.win_rate + '%' : '—')}
        ${sess.best  != null ? dStat('Best',  '+$' + fmt2(sess.best),  '', 'pos') : ''}
        ${sess.worst != null ? dStat('Worst', '$'  + fmt2(sess.worst), '', sess.worst < 0 ? 'neg' : '') : ''}
      </div>
    </div>`;

  ov.innerHTML = `
    ${sessSection}
    <div class="session-label" style="padding:0 0 6px 2px;margin-top:12px">ALL TIME</div>
    <div class="drawer-stats-grid">
      ${dStat('Balance',   '$' + fmt2(stats.balance),          `Start: $${fmt2(stats.balance_start)}`)}
      ${dStat('P&L',       pnlSign + '$' + fmt2(stats.pnl_dollars), pnlSign + fmt2(stats.pnl_pct) + '%', pnlCls)}
      ${dStat('Trades',    stats.total_trades ?? '—')}
      ${dStat('Win Rate',  stats.win_rate != null ? stats.win_rate + '%' : '—')}
      ${dStat('Avg P&L',   stats.avg_pnl != null ? '$' + fmt2(stats.avg_pnl) : '—', '',
              stats.avg_pnl > 0 ? 'pos' : stats.avg_pnl < 0 ? 'neg' : '')}
      ${dStat('Max DD',    stats.max_drawdown_pct != null ? stats.max_drawdown_pct + '%' : '—', '', 'neg')}
      ${dStat('Best',      stats.best_trade != null ? '+$' + fmt2(stats.best_trade) : '—', '', 'pos')}
      ${dStat('Worst',     stats.worst_trade != null ? '$' + fmt2(stats.worst_trade) : '—', '',
              stats.worst_trade < 0 ? 'neg' : '')}
      ${dStat('Daily P&L', '$' + fmt2(stats.daily_loss), '', stats.daily_loss < 0 ? 'neg' : '')}
    </div>
    <div class="session-label" style="padding:0 0 6px 2px;margin-top:12px">WIN RATE PROGRESSION</div>
    <div class="drawer-chart-wrap"><div id="perf-chart"></div></div>
  `;
  setTimeout(() => renderWinRateChart(d.win_rate_chart || { runs: [] }), 60);

  // ── ACTIVE TRADE TAB ──
  const openTrades = d.open_trades && d.open_trades.length ? d.open_trades
    : stats.open_trades && stats.open_trades.length ? stats.open_trades
    : stats.open_trade ? [stats.open_trade] : [];
  const tradePanel = document.getElementById('dpanel-trade');
  const _slPct = parseFloat(profile.stop_loss_pct) || 0;
  const _tpPct = parseFloat(profile.profit_target_pct) || 0;
  if (openTrades.length > 0) {
    tradePanel.innerHTML = openTrades.map(ot => {
      const op = parseOptionSymbol(ot.option_symbol);
      const symLabel = ot.symbol || (op ? op.ticker : null) || '—';
      const ep = ot.entry_price;
      const slPrice = (ep && _slPct) ? '$' + fmt4(ep * (1 - _slPct)) : '—';
      const tpPrice = (ep && _tpPct) ? '$' + fmt4(ep * (1 + _tpPct)) : '—';
      return `
      <div class="d-open-trade">
        <div class="d-open-trade-title">🟢 Open Position · ${symLabel}</div>
        <div class="d-open-trade-grid">
          ${dOtItem('Direction', `<span class="dir-badge ${ot.direction}">${ot.direction || '—'}</span>`)}
          ${dOtItem('Entered',  fmtDateTime(ot.entry_time))}
          ${dOtItem('Strike',  op ? '$' + op.strike : (ot.strike || '—'))}
          ${dOtItem('Type',    op ? op.type : '—')}
          ${dOtItem('Expiry',  op ? op.expiry : (ot.expiry || '—'))}
          ${dOtItem('Entry',   ep != null ? '$' + fmt4(ep) : '—')}
          ${dOtItem('SL',      slPrice, 'neg')}
          ${dOtItem('TP',      tpPrice, 'pos')}
          ${dOtItem('Qty',     ot.qty ?? '—')}
          ${dOtItem('Regime',  ot.regime || ot.regime_at_entry || '—')}
          ${dOtItem('Bucket',  ot.time_bucket || ot.time_of_day_bucket || '—')}
        </div>
      </div>`;
    }).join('');
  } else {
    tradePanel.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:13px">No open position</div>';
  }

  // ── HISTORY TAB ──
  renderHistoryTab(d.sim_id, d.recent_trades || []);

  // ── PROFILE TAB ──
  const profPanel = document.getElementById('dpanel-profile');
  const signalMode = (stats.signal_mode || profile.signal_mode || '').toUpperCase();
  const descText = STRATEGY_DESCRIPTIONS[signalMode] || null;
  const descHtml = descText ? `
    <div class="d-strategy-desc">
      <div class="d-strategy-desc-title">Strategy Overview</div>
      <p class="d-strategy-desc-body">${descText}</p>
    </div>` : '';
  profPanel.innerHTML = descHtml + `<div class="d-profile-grid">${
    Object.entries(profile)
      .filter(([k]) => !k.startsWith('_'))
      .map(([k, v]) => `
        <div class="d-profile-item">
          <span class="d-profile-key">${k}</span>
          <span class="d-profile-val">${Array.isArray(v) ? v.join(', ') : v ?? '—'}</span>
        </div>`)
      .join('')
  }</div>`;
}

// ─────────────────────────────────────────────── OPTION SYMBOL PARSER
// OCC format: TICKER + YYMMDD + C/P + 8-digit strike×1000
// e.g. SPY260317C00674000 → { ticker:'SPY', expiry:'Mar 17 \'26', type:'Call', strike:'674.00' }
function parseOptionSymbol(sym) {
  if (!sym) return null;
  const m = sym.match(/^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (!m) return null;
  const [, ticker, yy, mm, dd, typeChar, strikePad] = m;
  const strikeRaw = parseInt(strikePad, 10) / 1000;
  const strikeStr = strikeRaw % 1 === 0 ? String(strikeRaw) : strikeRaw.toFixed(2);
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const monthStr = months[parseInt(mm, 10) - 1] || mm;
  return {
    ticker,
    type:   typeChar === 'C' ? 'Call' : 'Put',
    strike: strikeStr,
    expiry: `${monthStr} ${parseInt(dd)} '${yy}`,
    raw:    sym,
  };
}

function fmtOptionSymbol(sym) {
  const p = parseOptionSymbol(sym);
  if (!p) return sym || '—';
  return `${p.ticker} $${p.strike} ${p.type} · ${p.expiry}`;
}

// ─────────────────────────────────────────────── TRADE ACCORDION
function renderTradeAccordion(trades, container) {
  if (!trades.length) {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:12px">No trades yet</div>';
    return;
  }
  container.innerHTML = trades.map((t, idx) => {
    const pnl     = t.pnl != null ? parseFloat(t.pnl) : null;
    const pnlPct  = t.pnl_pct != null ? (parseFloat(t.pnl_pct) * 100).toFixed(1) : null;
    const pnlCls  = pnl > 0 ? 'win' : pnl < 0 ? 'loss' : '';
    const pnlSign = pnl > 0 ? '+' : '';
    const pnlStr  = pnl != null ? pnlSign + '$' + fmt2(pnl) : '—';
    const pnlPctStr = pnlPct != null ? pnlSign + pnlPct + '%' : '';
    const dir     = (t.direction || '').toUpperCase();

    // Parse option symbol for human-readable display
    const opt = parseOptionSymbol(t.option_symbol);
    const symDisplay = opt ? `${opt.ticker} $${opt.strike} ${opt.type}` : (t.option_symbol || '—');
    const expiryDisplay = opt ? opt.expiry : (t.expiry || '—');

    return `
      <div class="trade-row-item ${pnlCls}" id="tr-${idx}">
        <div class="trade-row-summary" onclick="toggleTradeRow(${idx})">
          <span class="tr-time">${fmtTime(t.exit_time)}</span>
          <span class="tr-dir"><span class="dir-badge ${dir}">${dir || '—'}</span></span>
          <span class="tr-sym">${symDisplay}</span>
          <span class="tr-pnl ${pnlCls}">${pnlStr}${pnlPctStr ? ' <small style="opacity:.7">'+pnlPctStr+'</small>' : ''}</span>
          <span class="tr-toggle">▼</span>
        </div>
        <div class="trade-row-detail">
          <div class="trade-detail-grid">
            ${tdDetail('Exit Time',   fmtTime(t.exit_time))}
            ${tdDetail('Expiry',      expiryDisplay)}
            ${tdDetail('Strike',      opt ? '$' + opt.strike : (t.strike || '—'))}
            ${tdDetail('Type',        opt ? opt.type : dir)}
            ${tdDetail('Entry',       t.entry_price != null ? '$' + fmt4(t.entry_price) : '—')}
            ${tdDetail('Exit',        t.exit_price  != null ? '$' + fmt4(t.exit_price)  : '—')}
            ${tdDetail('P&L $',       pnlStr,  pnlCls)}
            ${tdDetail('P&L %',       pnlPctStr || '—', pnlCls)}
            ${tdDetail('Bucket',      t.time_bucket || '—')}
            ${tdDetail('Exit Reason', t.exit_reason || '—')}
            ${t.regime ? tdDetail('Regime', t.regime) : ''}
            ${tdDetail('Raw Symbol',  t.option_symbol || '—')}
          </div>
        </div>
      </div>`;
  }).join('');
}

function toggleTradeRow(idx) {
  const el = document.getElementById('tr-' + idx);
  if (!el) return;
  el.classList.toggle('expanded');
}

function toggleReplayChart(simId, tradeId, idx, e) {
  if (e) e.stopPropagation();
  const wrap = document.getElementById('tr-replay-wrap-' + idx);
  const img  = document.getElementById('tr-replay-chart-' + idx);
  const btn  = document.getElementById('tr-replay-btn-' + idx);
  if (!wrap) return;
  const isOpen = !wrap.classList.contains('hidden');
  if (isOpen) {
    wrap.classList.add('hidden');
    if (btn) btn.textContent = '▶ Replay';
  } else {
    wrap.classList.remove('hidden');
    if (btn) btn.textContent = '▼ Hide';
    if (img && !img.src.includes('/api/trades/')) {
      img.src = `/api/trades/${simId}/${encodeURIComponent(tradeId)}/chart?t=${Date.now()}`;
    }
  }
}

function tdDetail(label, val, cls = '') {
  return `<div class="td-detail-item">
    <span class="td-detail-label">${label}</span>
    <span class="td-detail-val ${cls}">${val}</span>
  </div>`;
}

function renderWinRateChart(winRateChart) {
  if (perfChart) { perfChart.destroy(); perfChart = null; }
  const el = document.getElementById('perf-chart');
  if (!el) return;

  const runs = (winRateChart || {}).runs || [];
  const validRuns = runs.filter(r => r.points && r.points.length > 0);

  if (!validRuns.length) {
    el.innerHTML = '<div style="padding:20px;text-align:center;color:#888;font-size:12px">No trades yet</div>';
    return;
  }

  const hasMultipleRuns = validRuns.length > 1;
  const maxTrades = Math.max(...validRuns.map(r => r.points.length));

  const series = validRuns.map(run => ({
    name: run.is_current ? 'Current Run' : `Run #${run.run_number} (ended)`,
    data: run.points.map(p => ({ x: p.trade_num, y: p.win_rate })),
  }));

  const colors = validRuns.map(run => run.is_current ? '#4f8ef7' : 'rgba(79,142,247,0.35)');
  const strokeWidths = validRuns.map(run => run.is_current ? 2.5 : 1.5);

  perfChart = new ApexCharts(el, {
    chart: {
      type: 'line',
      height: 160,
      background: 'transparent',
      toolbar: { show: false },
      animations: { enabled: false },
      foreColor: '#888',
    },
    series,
    stroke: { curve: 'smooth', width: strokeWidths },
    colors,
    xaxis: {
      type: 'numeric',
      tickAmount: Math.min(10, maxTrades - 1),
      labels: {
        style: { colors: '#64748b', fontSize: '9px' },
        formatter: v => Number.isInteger(+v) ? String(Math.round(v)) : '',
      },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      min: 0,
      max: 100,
      tickAmount: 5,
      labels: {
        style: { colors: '#64748b', fontSize: '9px' },
        formatter: v => v + '%',
      },
    },
    grid: { borderColor: 'rgba(0,0,0,0.08)', strokeDashArray: 4 },
    tooltip: {
      theme: 'light',
      x: { formatter: v => 'Trade #' + Math.round(v) },
      y: {
        formatter: (v, { seriesIndex }) => {
          const run = validRuns[seriesIndex];
          const label = run.is_current ? 'Win Rate' : `Run #${run.run_number} WR`;
          return label + ': ' + (v || 0).toFixed(1) + '%';
        },
      },
    },
    legend: {
      show: hasMultipleRuns,
      position: 'top',
      horizontalAlign: 'right',
      fontSize: '9px',
      labels: { colors: '#888' },
    },
    dataLabels: { enabled: false },
    markers: { size: maxTrades <= 30 ? 3 : 0, hover: { size: 5 } },
  });
  perfChart.render();
}

// ─────────────────────────────────────────────── HELPERS
function dStat(label, value, sub = '', cls = '') {
  return `<div class="dstat">
    <div class="dstat-label">${label}</div>
    <div class="dstat-value ${cls}">${value}</div>
    ${sub ? `<div class="dstat-sub">${sub}</div>` : ''}
  </div>`;
}

function dOtItem(label, val, cls = '') {
  return `<div class="d-ot-item">
    <div class="d-ot-label">${label}</div>
    <div class="d-ot-val${cls ? ' ' + cls : ''}">${val}</div>
  </div>`;
}

// Legacy helpers (kept for any remaining references)
function statCard(label, value, sub, cls = '') {
  return dStat(label, value, sub, cls);
}
function perfStat(label, value, cls = '') {
  return `<div class="dperf-stat"><div class="dperf-stat-label">${label}</div><div class="dperf-stat-value ${cls}">${value}</div></div>`;
}
function otItem(label, val) { return dOtItem(label, val); }

function fmt2(n)  { return n != null ? parseFloat(n).toFixed(2) : '—'; }
function fmt4(n)  { return n != null ? parseFloat(n).toFixed(4) : '—'; }

function fmtTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  } catch { return iso.slice(11, 16) || iso; }
}

function fmtDateTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    return `${date} ${time}`;
  } catch { return iso.slice(0, 16) || iso; }
}

// Close drawer on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDrawer();
});

// ─────────────────────────────────────────────── HISTORY TAB (enhanced)
let _histSimId = null;
let _histPage  = 1;
let _histTotal = 0;
let _histFilters = { direction: 'ALL', result: 'all', date: '' };
let _selectedTradeId = null;
let _symbolRegistry = {};

async function _loadSymbolRegistry() {
  if (Object.keys(_symbolRegistry).length > 0) return _symbolRegistry;
  try {
    const r = await fetch('/api/symbols');
    if (r.ok) _symbolRegistry = await r.json();
  } catch {}
  return _symbolRegistry;
}

async function renderHistoryTab(simId, seedTrades) {
  _histSimId = simId;
  _histPage  = 1;
  const histPanel = document.getElementById('dpanel-history');

  // Load symbol registry to populate dropdown
  const syms = await _loadSymbolRegistry();
  const symOptions = Object.keys(syms).map(s =>
    `<option value="${s}">${s}</option>`).join('');

  histPanel.innerHTML = `
    <div class="history-header">
      <span class="history-title">TRADE LOG</span>
      <span class="history-count" id="hist-count">loading…</span>
    </div>
    <div class="hist-filters">
      <input  type="date" id="hist-date" class="hist-filter-input"
              onchange="applyHistFilter()" title="Filter by entry date"/>
      <select id="hist-dir" class="hist-filter-select" onchange="applyHistFilter()">
        <option value="ALL">All Directions</option>
        <option value="BULLISH">Calls</option>
        <option value="BEARISH">Puts</option>
      </select>
      <select id="hist-result" class="hist-filter-select" onchange="applyHistFilter()">
        <option value="all">All Results</option>
        <option value="win">Wins</option>
        <option value="loss">Losses</option>
      </select>
      ${symOptions ? `<select id="hist-symbol" class="hist-filter-select" onchange="applyHistFilter()">
        <option value="">All Symbols</option>
        ${symOptions}
      </select>` : ''}
    </div>
    <div class="trade-accordion" id="trade-accordion"></div>
    <div class="hist-pagination" id="hist-pagination"></div>
  `;
  // Try to load from dedicated endpoint; fall back to seedTrades
  loadHistoryPage(1, seedTrades);
}

async function loadHistoryPage(page, seedTrades) {
  _histPage = page;
  if (!_histSimId) return;

  const dateVal = (document.getElementById('hist-date')   || {}).value || '';
  const dirVal  = (document.getElementById('hist-dir')    || {}).value || 'ALL';
  const resVal  = (document.getElementById('hist-result') || {}).value || 'all';
  const symVal  = (document.getElementById('hist-symbol') || {}).value || '';

  let trades = [], total = 0;
  try {
    let url = `/api/trades/${_histSimId}/history?page=${page}&per_page=50`;
    if (dateVal) url += `&date=${dateVal}`;
    if (dirVal  !== 'ALL')  url += `&direction=${dirVal}`;
    if (resVal  !== 'all')  url += `&result=${resVal}`;
    if (symVal)             url += `&symbol=${symVal}`;
    const r = await fetch(url);
    if (r.ok) {
      const data = await r.json();
      trades = data.trades || [];
      total  = data.total  || 0;
    } else if (seedTrades && seedTrades.length) {
      trades = seedTrades; total = seedTrades.length;
    }
  } catch {
    trades = seedTrades || []; total = trades.length;
  }

  _histTotal = total;
  const countEl = document.getElementById('hist-count');
  if (countEl) countEl.textContent = `${total} trade${total !== 1 ? 's' : ''}`;

  const container = document.getElementById('trade-accordion');
  if (container) renderTradeAccordionWithDetails(trades, container, _histSimId);

  // Pagination
  const pgEl = document.getElementById('hist-pagination');
  if (pgEl) {
    const totalPages = Math.ceil(total / 50);
    pgEl.innerHTML = totalPages > 1 ? `
      <button class="hist-pg-btn" onclick="loadHistoryPage(${page-1})"
              ${page <= 1 ? 'disabled' : ''}>← Prev</button>
      <span class="hist-pg-label">Page ${page} / ${totalPages}</span>
      <button class="hist-pg-btn" onclick="loadHistoryPage(${page+1})"
              ${page >= totalPages ? 'disabled' : ''}>Next →</button>
    ` : '';
  }
}

function applyHistFilter() { loadHistoryPage(1); }

// ─────────────────────────────────────────────── TRADE CARDS WITH DETAILS BUTTON
function renderTradeAccordionWithDetails(trades, container, simId) {
  if (!trades.length) {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:12px">No trades</div>';
    return;
  }
  let _lastDate = null;
  container.innerHTML = trades.map((t, idx) => {
    const pnl     = t.pnl != null ? parseFloat(t.pnl) : null;
    const pnlPct  = t.pnl_pct != null ? (parseFloat(t.pnl_pct) * 100).toFixed(1) : null;
    const pnlCls  = pnl > 0 ? 'win' : pnl < 0 ? 'loss' : '';
    const pnlSign = pnl > 0 ? '+' : '';
    const pnlStr  = pnl != null ? pnlSign + '$' + fmt2(Math.abs(pnl)) : '—';
    const dir     = (t.direction || '').toUpperCase();
    const opt     = parseOptionSymbol(t.option_symbol);
    const symDisp = opt ? `${opt.ticker} $${opt.strike} ${opt.type}` : (t.option_symbol || '—');
    const isSelected = t.trade_id && t.trade_id === _selectedTradeId;
    const undSym  = t.symbol || (opt ? opt.ticker : '');
    const symBadge = undSym ? `<span class="sym-badge sym-${undSym}">${undSym}</span>` : '';

    // Date separator
    const tsStr = t.exit_time || t.entry_time || '';
    const dateStr = tsStr.slice(0, 10);
    let dateSep = '';
    if (dateStr && dateStr !== _lastDate) {
      _lastDate = dateStr;
      const d = new Date(dateStr + 'T12:00:00');
      const label = isNaN(d) ? dateStr : d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
      dateSep = `<div class="tr-date-sep">${label}</div>`;
    }

    return dateSep + `
      <div class="trade-row-item ${pnlCls}${isSelected ? ' tr-selected' : ''}" id="tr-${idx}">
        <div class="trade-row-summary" onclick="toggleTradeRow(${idx})">
          <span class="tr-time">${fmtTime(t.exit_time || t.entry_time)}</span>
          <span class="tr-dir"><span class="dir-badge ${dir}">${dir || '—'}</span></span>
          ${symBadge}
          <span class="tr-sym">${symDisp}</span>
          <span class="tr-pnl ${pnlCls}">${pnl != null ? (pnlSign + '$' + fmt2(Math.abs(pnl))) : '—'}</span>
          ${t.grade ? `<span class="tr-grade grade-badge grade-${t.grade.toLowerCase()}">${t.grade}</span>` : '<span class="tr-grade"></span>'}
          <span class="tr-toggle">▼</span>
        </div>
        <div class="trade-row-detail">
          <div class="trade-detail-grid">
            ${tdDetail('Expiry',      opt ? opt.expiry : (t.expiry || '—'))}
            ${tdDetail('Strike',      opt ? '$' + opt.strike : (t.strike ? '$' + t.strike : '—'))}
            ${tdDetail('Type',        opt ? opt.type : (t.contract_type === 'C' ? 'Call' : t.contract_type === 'P' ? 'Put' : dir))}
            ${tdDetail('Entry',       t.entry_price != null ? '$' + fmt4(t.entry_price) : '—')}
            ${tdDetail('Exit',        t.exit_price  != null ? '$' + fmt4(t.exit_price)  : '—')}
            ${t.sl_price != null ? tdDetail('SL', '$' + fmt4(t.sl_price), 'neg') : ''}
            ${t.tp_price != null ? tdDetail('TP', '$' + fmt4(t.tp_price), 'pos') : ''}
            ${tdDetail('P&L $',       pnl != null ? (pnlSign + '$' + fmt2(Math.abs(pnl))) : '—',  pnlCls)}
            ${tdDetail('P&L %',       pnlPct != null ? (pnlSign + pnlPct + '%') : '—',            pnlCls)}
            ${tdDetail('Exit Reason', t.exit_reason || '—')}
            ${t.exit_context ? tdDetail('Detail', t.exit_context) : ''}
            ${t.regime ? tdDetail('Regime', t.regime) : ''}
          </div>
          ${t.trade_id ? `
          <div class="tr-action-row">
            <button class="tr-details-btn${isSelected ? ' tr-details-btn-active' : ''}"
                    onclick="openTradeDetails('${simId}','${t.trade_id}',${idx},event)">
              ${isSelected ? '✓ Viewing' : '✨ Analyze'}
            </button>
            <button class="tr-replay-btn" id="tr-replay-btn-${idx}"
                    onclick="toggleReplayChart('${simId}','${t.trade_id}',${idx},event)">
              ▶ Replay
            </button>
          </div>
          <div class="tr-replay-wrap hidden" id="tr-replay-wrap-${idx}">
            <div class="tr-replay-loading" id="tr-replay-load-${idx}">Loading chart…</div>
            <img class="tr-replay-img hidden" id="tr-replay-chart-${idx}" src="" alt="Trade replay"
                 onload="this.classList.remove('hidden');document.getElementById('tr-replay-load-${idx}').classList.add('hidden')"
                 onerror="this.classList.add('hidden');document.getElementById('tr-replay-load-${idx}').textContent='Replay unavailable'"/>
          </div>` : ''}
        </div>
      </div>`;
  }).join('');
}

// ─────────────────────────────────────────────── CHALKBOARD TRADE ANALYSIS
async function openTradeDetails(simId, tradeId, rowIdx, e) {
  if (e) e.stopPropagation();
  _selectedTradeId = tradeId;

  // Re-render accordion to update selected highlight
  const container = document.getElementById('trade-accordion');
  if (container) {
    container.querySelectorAll('.trade-row-item').forEach((el, i) => {
      el.classList.toggle('tr-selected', i === rowIdx);
    });
    const btn = container.querySelector(`#tr-${rowIdx} .tr-details-btn`);
    if (btn) { btn.textContent = '✓ Viewing'; btn.classList.add('tr-details-btn-active'); }
  }

  // Switch chalkboard to trade analysis mode
  showTradeChart(simId, tradeId);

  // If on mobile, close drawer so chalkboard is visible
  if (window.innerWidth < 768) closeDrawer();
}

const _isDesktop = () => window.innerWidth >= 769;

function showTradeChart(simId, tradeId) {
  // On desktop: both panels always visible (split mode) — don't hide live view
  if (!_isDesktop()) {
    document.getElementById('chalk-live-view').classList.add('hidden');
  }
  document.getElementById('chalk-trade-view').classList.remove('hidden');
  // Show content, hide placeholder
  const placeholder = document.getElementById('chalk-trade-placeholder');
  const content     = document.getElementById('chalk-trade-content');
  if (placeholder) placeholder.classList.add('hidden');
  if (content)     content.classList.remove('hidden');

  const img     = document.getElementById('trade-chart-img');
  const loading = document.getElementById('trade-chart-loading');
  const errEl   = document.getElementById('trade-chart-error');
  const narBtn  = document.getElementById('narrate-btn');

  img.classList.add('hidden');
  errEl.classList.add('hidden');
  loading.classList.remove('hidden');

  // Store for narrate button
  document._currentTradeSimId = simId;
  document._currentTradeId    = tradeId;
  if (narBtn) narBtn.textContent = '✨ Analyze';

  // Update title
  const titleEl = document.getElementById('trade-chart-title');
  if (titleEl) titleEl.textContent = `${simId} · TRADE ANALYSIS`;

  const src = `/api/trades/${simId}/${encodeURIComponent(tradeId)}/chart?t=${Date.now()}`;
  img.onload  = () => { loading.classList.add('hidden'); img.classList.remove('hidden'); };
  img.onerror = () => { loading.classList.add('hidden'); errEl.classList.remove('hidden'); };
  img.src = src;

  // Check for cached narrative
  loadCachedNarrative(simId, tradeId);
}

function showLiveChart() {
  document.getElementById('chalk-live-view').classList.remove('hidden');
  // On desktop: keep trade view visible but show placeholder
  if (_isDesktop()) {
    const placeholder = document.getElementById('chalk-trade-placeholder');
    const content     = document.getElementById('chalk-trade-content');
    if (placeholder) placeholder.classList.remove('hidden');
    if (content)     content.classList.add('hidden');
  } else {
    document.getElementById('chalk-trade-view').classList.add('hidden');
  }
  const narPanel = document.getElementById('narrative-panel');
  if (narPanel) narPanel.classList.add('hidden');
  _selectedTradeId = null;
}

async function loadCachedNarrative(simId, tradeId) {
  const panel = document.getElementById('narrative-panel');
  try {
    const r = await fetch(`/api/trades/${simId}/${encodeURIComponent(tradeId)}/narrative`);
    if (r.ok) {
      const narr = await r.json();
      renderNarrative(narr);
      const btn = document.getElementById('narrate-btn');
      if (btn) btn.textContent = '↻ Re-analyze';
    } else {
      panel.classList.remove('hidden');
      document.getElementById('narrative-loading').classList.add('hidden');
      document.getElementById('narrative-content').classList.add('hidden');
    }
  } catch { /* no narrative yet */ }
}

async function requestNarrative(force = false) {
  const simId   = document._currentTradeSimId;
  const tradeId = document._currentTradeId;
  if (!simId || !tradeId) return;

  const panel   = document.getElementById('narrative-panel');
  const loading = document.getElementById('narrative-loading');
  const content = document.getElementById('narrative-content');
  const btn     = document.getElementById('narrate-btn');

  panel.classList.remove('hidden');
  loading.classList.remove('hidden');
  content.classList.add('hidden');
  if (btn) { btn.disabled = true; btn.textContent = '…'; }

  try {
    const url = `/api/trades/${simId}/${encodeURIComponent(tradeId)}/narrate${force ? '?force=true' : ''}`;
    const r   = await fetch(url, { method: 'POST' });
    if (r.ok) {
      const narr = await r.json();
      renderNarrative(narr);
      // Re-fetch chart with new S/R lines
      if (force) showTradeChart(simId, tradeId);
      if (btn) { btn.disabled = false; btn.textContent = '↻ Re-analyze'; }
    } else {
      loading.classList.add('hidden');
      content.innerHTML = '<div style="padding:12px;color:var(--loss-text)">Analysis failed — check OPENAI_API_KEY.</div>';
      content.classList.remove('hidden');
      if (btn) { btn.disabled = false; btn.textContent = '✨ Analyze'; }
    }
  } catch(e) {
    loading.classList.add('hidden');
    if (btn) { btn.disabled = false; btn.textContent = '✨ Analyze'; }
  }
}

function renderNarrative(narr) {
  const loading = document.getElementById('narrative-loading');
  const content = document.getElementById('narrative-content');
  const panel   = document.getElementById('narrative-panel');

  const _unavailMsgs = [
    'Analysis unavailable — GPT service unreachable or API key not set.',
    'Analysis unavailable — OpenAI API key invalid or expired. Update OPENAI_API_KEY in .env',
  ];
  if (!narr || _unavailMsgs.includes(narr.entry_reasoning)) {
    loading.classList.add('hidden');
    const msg = (narr && narr.entry_reasoning && narr.entry_reasoning.includes('invalid or expired'))
      ? 'AI analysis unavailable — OpenAI API key invalid or expired.'
      : 'AI analysis unavailable — OPENAI_API_KEY not set in .env';
    content.innerHTML = `<div style="padding:12px;color:var(--loss-text);font-size:12px">${msg}</div>`;
    content.classList.remove('hidden');
    panel.classList.remove('hidden');
    return;
  }

  document.getElementById('narr-summary').textContent  = narr.strategy_summary || '';
  document.getElementById('narr-entry-text').textContent  = narr.entry_reasoning  || '—';
  document.getElementById('narr-exit-text').textContent   = narr.exit_reasoning   || '—';
  document.getElementById('narr-outcome-text').textContent = narr.outcome_analysis || '—';

  // Grade
  const gradeEl = document.getElementById('narr-grade');
  const grade   = narr.grade || '';
  const gradeCls = {A:'grade-a',B:'grade-b',C:'grade-c',D:'grade-d',F:'grade-f'}[grade] || '';
  gradeEl.innerHTML = grade ? `<span class="grade-badge ${gradeCls}">${grade}</span>` : '';

  // Tags
  const tagsEl = document.getElementById('narr-tags');
  const tags   = narr.tags || [];
  tagsEl.innerHTML = tags.map(t => `<span class="narr-tag">${t}</span>`).join('');

  loading.classList.add('hidden');
  content.classList.remove('hidden');
  panel.classList.remove('hidden');
}
