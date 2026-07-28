"""
black_scholes.py
-----------------
Core option-pricing utilities used across the project:

  1. Analytic Black-Scholes (1973) pricer for European calls/puts
  2. Greeks (delta, gamma, vega, theta, rho)
  3. Newton-Raphson implied-volatility solver
  4. A Monte-Carlo (GBM) European option pricer, used purely as a
     cross-check / illustration alongside the closed-form model.

All functions are vectorised with NumPy so they can be applied
directly to a pandas DataFrame of option quotes.
"""

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------
# 1. Analytic Black-Scholes price
# ----------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S : float or array  - spot price of the underlying (Nifty50 level)
    K : float or array  - strike price
    T : float or array  - time to expiry, in YEARS
    r : float or array  - continuously-compounded risk-free rate (decimal)
    sigma : float/array - annualised volatility (decimal)
    option_type : "call" or "put"
    q : float            - continuous dividend / index yield (decimal)

    Returns
    -------
    price : float or array
    """
    S, K, T, r, sigma = map(np.asarray, (S, K, T, r, sigma))
    # Guard against non-positive time/vol (avoids div-by-zero warnings)
    T = np.where(T <= 0, 1e-8, T)
    sigma = np.where(sigma <= 0, 1e-8, sigma)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


# ----------------------------------------------------------------------
# 2. Greeks (used in the report / diagnostics, not required for pricing)
# ----------------------------------------------------------------------
def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    S, K, T, r, sigma = map(float, (S, K, T, r, sigma))
    T = max(T, 1e-8)
    sigma = max(sigma, 1e-8)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta / 365.0, "rho": rho / 100.0}


# ----------------------------------------------------------------------
# 3. Implied volatility (Newton-Raphson with bisection fallback)
# ----------------------------------------------------------------------
def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                 tol=1e-6, max_iter=100):
    """
    Solve sigma such that bs_price(S,K,T,r,sigma) == price.
    Falls back to bisection if Newton-Raphson fails to converge
    (vega can be near-zero for deep ITM/OTM or very short-dated options).
    """
    if price <= 0 or T <= 0:
        return np.nan

    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        price_est = bs_price(S, K, T, r, sigma, option_type, q)
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"]
        if vega < 1e-8:
            break
        diff = price_est - price
        if abs(diff) < tol:
            return float(sigma)
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.01

    # Bisection fallback
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        est = bs_price(S, K, T, r, mid, option_type, q)
        if est > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float((lo + hi) / 2)


# ----------------------------------------------------------------------
# 4. Monte-Carlo European option pricer (GBM), used for cross-validation
# ----------------------------------------------------------------------
def mc_price(S, K, T, r, sigma, option_type="call", q=0.0,
             n_paths=100_000, n_steps=1, antithetic=True, seed=None):
    """
    Prices a European option by simulating terminal Nifty50 values under
    geometric Brownian motion (the same lognormal assumption Black-Scholes
    makes) and discounting the average payoff back to today.

    With n_steps=1 this collapses to sampling S_T directly, which is the
    standard, fastest way to Monte-Carlo a European (non-path-dependent)
    payoff. n_steps > 1 is provided for illustration / extension to
    path-dependent payoffs.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    n = n_paths // 2 if antithetic else n_paths

    Z = rng.standard_normal((n, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(drift + diffusion, axis=1)
    S_T = S * np.exp(log_paths[:, -1])

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, stderr
