"""
generate_sample_data.py
------------------------
IMPORTANT / READ FIRST
-----------------------
This environment has no network access to NSE, brokers, or market-data
vendors (its outbound access is limited to package registries such as
PyPI/GitHub), so it cannot download a real, live Nifty50 option chain.

To keep the rest of the pipeline runnable end-to-end, this script builds
a SYNTHETIC-BUT-REALISTIC dataset:
  - Spot path is bootstrapped to look like a short Nifty50 trading window
    anchored near its actual late-July-2026 level (~24,000).
  - "Market" option prices are generated from Black-Scholes with a
    volatility SMILE/SKEW and a small amount of random microstructure
    noise added on top -- i.e. the "real" prices are deliberately NOT
    pure Black-Scholes prices, so the backtest has genuine (but bounded)
    pricing error to measure, similar to what you would see against
    real NSE quotes.

TO USE REAL DATA INSTEAD:
  Replace data/nifty50_options_sample.csv with actual NSE option-chain /
  historical bhavcopy data with the same column names. Good free/paid
  sources: NSE India bhavcopy archives (nseindia.com), the `nsepython`
  or `jugaad-data` PyPI packages, or a broker API (Kite Connect, Dhan,
  Angel One, etc.) -- run this on a machine with outbound internet
  access to those domains and export to the same schema below.

Output columns
--------------
trade_date, expiry_date, spot, strike, option_type, days_to_expiry,
risk_free_rate, hist_vol_20d, market_price
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "nifty50_options_sample.csv"

rng = np.random.default_rng(42)

# ---- BS pricer (local import so this script is standalone-runnable) ----
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from black_scholes import bs_price


def simulate_spot_path(s0=23996.0, n_days=40, mu=0.06, sigma=0.13, seed=1):
    """Simulate a plausible ~2-month daily Nifty50 closing path (GBM)."""
    g = np.random.default_rng(seed)
    dt = 1 / 252
    z = g.standard_normal(n_days)
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    path = s0 * np.exp(np.cumsum(log_ret))
    return np.round(path, 2)


def realized_vol(path, window=20):
    """Rolling annualised realised volatility from a price path."""
    log_ret = np.diff(np.log(path))
    rv = pd.Series(log_ret).rolling(window, min_periods=5).std() * np.sqrt(252)
    rv = rv.bfill().to_numpy()
    return np.concatenate([[rv[0]], rv])  # align length to path


def smile_vol(base_vol, moneyness, T):
    """
    Simple skew: NSE index options typically show higher IV for OTM puts
    (crash protection demand) and a milder smile on the call wing.
    moneyness = K / S  (1.0 = ATM)
    """
    skew = -0.15 * (moneyness - 1.0)                 # downside skew
    smile = 0.35 * (moneyness - 1.0) ** 2             # convexity far ITM/OTM
    term_decay = 0.02 / np.sqrt(np.maximum(T, 1 / 365))  # short-dated IV bump
    return np.clip(base_vol + skew + smile + 0.15 * term_decay, 0.05, 1.5)


def generate_dataset():
    spot_path = simulate_spot_path()
    rv20 = realized_vol(spot_path)
    trade_dates = pd.bdate_range("2026-06-01", periods=len(spot_path))

    expiries = pd.to_datetime(["2026-07-31", "2026-08-28", "2026-09-25"])  # monthly NSE expiries
    strike_step = 50  # Nifty50 weekly/monthly strikes are in steps of 50
    moneyness_grid = np.arange(0.90, 1.101, 0.01)  # +/-10% around spot, 1% steps

    r = 0.065  # approx short-term Indian G-sec / T-bill yield used as risk-free rate

    rows = []
    for i, tdate in enumerate(trade_dates):
        S = spot_path[i]
        base_vol = max(rv20[i], 0.08)
        for expiry in expiries:
            T = (expiry - tdate).days / 365.0
            if T <= 0:
                continue
            for m in moneyness_grid:
                K = round(S * m / strike_step) * strike_step
                for opt_type in ("call", "put"):
                    sigma_true = smile_vol(base_vol, K / S, T)
                    theo = bs_price(S, K, T, r, sigma_true, opt_type)
                    if theo < 0.5:  # skip worthless/illiquid far strikes
                        continue
                    # microstructure noise: bid-ask + rounding, scaled to price level
                    noise = rng.normal(0, 0.015 * max(theo, 5)) + rng.choice([-0.05, 0, 0.05])
                    market_price = max(round(theo + noise, 2), 0.05)

                    rows.append({
                        "trade_date": tdate.date(),
                        "expiry_date": expiry.date(),
                        "spot": round(S, 2),
                        "strike": int(K),
                        "option_type": opt_type,
                        "days_to_expiry": (expiry - tdate).days,
                        "risk_free_rate": r,
                        "hist_vol_20d": round(base_vol, 4),
                        "market_price": market_price,
                    })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic option quotes -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    generate_dataset()
