# Black-Scholes Accuracy Backtest — Nifty50 Index Options

This repo prices Nifty50 index options with the **Black-Scholes (1973)**
model and measures how closely those theoretical prices track observed
market prices, bucketed by moneyness, maturity, and option type. It also
includes a **Monte-Carlo pricer** used to cross-validate the closed-form
formula, and a short primer on Monte-Carlo methods (below).

```
nifty-bs-accuracy/
├── data/
│   └── nifty50_options_sample.csv   # backtest dataset (see "About the data")
├── src/
│   ├── black_scholes.py             # BS price, Greeks, implied vol, Monte-Carlo pricer
│   ├── generate_sample_data.py      # builds the sample dataset
│   └── backtest_accuracy.py         # main entry point — runs the backtest
├── outputs/                         # metrics CSVs + PNG plots (generated)
├── requirements.txt
└── README.md
```

## Quick start

```bash
git clone <your-repo-url>.git
cd nifty-bs-accuracy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/backtest_accuracy.py
```

This will (re)generate `data/nifty50_options_sample.csv` if it's missing,
price every quote with Black-Scholes, print summary tables to the
console, and write everything to `outputs/`:

| File | Contents |
|---|---|
| `metrics_overall.csv` | MAE, RMSE, MAPE, bias, correlation across all quotes |
| `metrics_by_option_type.csv` | Same metrics split by call/put |
| `metrics_by_moneyness.csv` | Same metrics split into 5 moneyness buckets |
| `metrics_by_maturity.csv` | Same metrics split by days-to-expiry buckets |
| `priced_quotes_full.csv` | Every quote with its BS price, error, % error |
| `bs_vs_market_scatter.png` | BS price vs market price, 45° = perfect fit |
| `error_distribution.png` | Histogram of % pricing error |
| `error_by_moneyness.png` | Mean % bias per moneyness bucket |
| `implied_vol_smile.png` | The implied-vol smile/skew the market prices embed |
| `monte_carlo_crosscheck.csv` | BS closed-form vs Monte-Carlo simulated price, same inputs |

## Methodology

For every option quote `(spot, strike, days_to_expiry, option_type)` the
script computes:

```
BS price = BlackScholes(S, K, T, r, σ)
```

where **σ is the trailing 20-day realised (historical) volatility** of
the underlying, *not* volatility implied from that same option's price.
Using historical vol as the input and comparing the output to the market
price is what actually tests the model — feeding in implied volatility
and "predicting" the market price back would be circular, since implied
vol is defined as *the volatility that makes BS match the market price*.

Error metrics reported:

- **MAE** — mean absolute error (₹)
- **RMSE** — root-mean-squared error (₹, penalises large misses more)
- **MAPE** — mean absolute % error (comparable across strikes/maturities)
- **Bias** — signed mean error (negative = BS underprices vs. market on average)
- **Corr** — correlation between BS and market prices (fit quality on rank/scale, not calibration)

The **implied-volatility smile plot** is itself a diagnostic of
Black-Scholes' main weakness: BS assumes one constant σ for every strike
and maturity, but the model only reprices the market well at the single
strike from which σ was extracted. If the smile were flat, BS would be
(nearly) exact everywhere; the curvature and skew you see is a direct
visual measure of the model's mis-specification versus how the real
Nifty50 options market actually prices risk (fat tails, especially on
the downside, i.e. put skew from crash-hedging demand).

### Typical findings (see `outputs/` after running)

- Black-Scholes tracks market prices closely in level (`Corr` ≈ 0.999+
  — spot and moneyness dominate price, and BS gets that right), but the
  **residual mis-pricing concentrates near-the-money and on far
  OTM/ITM puts**, exactly where the volatility skew is steepest.
- **MAPE is typically much higher on puts than calls**, reflecting the
  well-documented Indian-index put skew (elevated implied vol on
  downside strikes that a flat-vol model cannot capture).
- Deep ITM options show the smallest % errors (their price is dominated
  by intrinsic value, which every model, including BS, gets exactly right).
- Errors generally grow with days-to-expiry, since realised vol drifting
  from what will actually realize compounds pricing error over a longer horizon.

Exact numbers will differ every time you regenerate the sample dataset
(it uses a fixed random seed by default, so re-running `backtest_accuracy.py`
without touching `generate_sample_data.py` reproduces the same numbers).

## About the data

**This sandboxed environment has no outbound network access to NSE,
brokers, or market-data vendors** (only software package registries are
reachable), so a live/historical Nifty50 option chain could not be
downloaded here. `src/generate_sample_data.py` instead builds a
**synthetic-but-realistic** dataset:

- A ~40-trading-day Nifty50 spot path is simulated via GBM, anchored
  near its actual late-July-2026 level (~₹24,000).
- "Market" prices are generated from Black-Scholes using a volatility
  **smile/skew** (not a single flat σ) plus small random microstructure
  noise — so the "true" data-generating process deliberately deviates
  from flat-vol Black-Scholes, giving the backtest genuine, bounded
  pricing error to detect (similar in character to what you'd see
  against real NSE quotes).
- Risk-free rate is fixed at 6.5%, roughly in line with short-term
  Indian G-sec/T-bill yields.

**To back-test against real data:** replace
`data/nifty50_options_sample.csv` with an actual NSE option-chain /
bhavcopy export using the same columns
(`trade_date, expiry_date, spot, strike, option_type, days_to_expiry,
risk_free_rate, hist_vol_20d, market_price`). Good sources, to be run
from a machine with normal internet access:
- NSE India historical bhavcopy / option-chain archives (nseindia.com)
- `nsepython` or `jugaad-data` (PyPI packages that wrap NSE endpoints)
- A broker API such as Zerodha Kite Connect, Dhan, or Angel One

Everything downstream (`backtest_accuracy.py`) works unchanged once the
CSV schema matches.

## Monte-Carlo simulation — brief overview

Black-Scholes gives a **closed-form** price by solving the option-pricing
PDE analytically under specific assumptions (lognormal prices, constant
volatility and rates, continuous frictionless trading, European
exercise). **Monte-Carlo simulation** is a general **numerical**
alternative that works even when no closed form exists:

1. **Model the underlying's evolution.** For a European option under
   the same Black-Scholes assumptions, the Nifty50 level at expiry
   follows geometric Brownian motion:
   `S_T = S_0 · exp[(r − q − ½σ²)T + σ√T · Z]`, where `Z ~ N(0,1)`.
2. **Simulate many random price paths** (or, for a European payoff, many
   random draws of just the terminal value `S_T`) — tens of thousands to
   millions of scenarios, each representing one possible future.
3. **Compute the option's payoff in each scenario** — e.g.
   `max(S_T − K, 0)` for a call, `max(K − S_T, 0)` for a put.
4. **Average the payoffs and discount back to today** at the risk-free
   rate: `Price ≈ e^(−rT) · mean(payoffs)`. By the law of large numbers
   this average converges to the true expected discounted payoff as the
   number of simulated paths grows; the **standard error shrinks with
   `1/√N`**, which is why `mc_price()` in this repo also reports a
   standard error alongside the price.

**Why it matters here:** for a plain European option under GBM,
Monte-Carlo and Black-Scholes are pricing *the exact same model*, just
one analytically and one by simulation — so `outputs/monte_carlo_crosscheck.csv`
shows them agreeing to within a few paise/rupees (well within Monte-Carlo
standard error). That's a correctness check on the code, not a market
accuracy test.

Monte-Carlo earns its keep on richer problems Black-Scholes can't handle
in closed form:
- **Path-dependent payoffs** (Asian, barrier, lookback options)
- **American-style early exercise** (via Longstaff-Schwartz least-squares MC)
- **Multiple/correlated underlyings** (basket options, index-vs-stock spreads)
- **Stochastic or jump-diffusion dynamics** (Heston, SABR, Merton jump
  models) that better match the volatility smile this backtest exposes,
  at the cost of no closed-form solution
- **Stochastic interest rates or exotic path/barrier features**

The trade-off is computational cost and simulation (sampling) error vs.
the instant, exact answer a closed form gives when one exists — which is
exactly why Black-Scholes, despite its flaws, remains the default
quoting/benchmark convention (via implied vol) even though the market
it's used on visibly violates its own assumptions.

## Publishing this to GitHub

This repo has already been initialised locally (`git init`) with the
files above committed. Anthropic's sandbox that produced these files has
no GitHub credentials and cannot push on your behalf — push it from your
own machine:

```bash
# from inside the nifty-bs-accuracy/ folder
git add -A
git commit -m "Black-Scholes accuracy backtest on Nifty50 index options"

# create an empty repo on GitHub first (via github.com or `gh repo create`), then:
git remote add origin https://github.com/<your-username>/nifty-bs-accuracy.git
git branch -M main
git push -u origin main
```

If you have the GitHub CLI installed, `gh repo create nifty-bs-accuracy
--public --source=. --push` does all of the above in one step.

## Limitations & honest caveats

- The bundled dataset is **synthetic**, calibrated to look like a real
  Nifty50 option chain but not sourced from live market data (see
  "About the data"). Treat the specific error numbers as illustrative
  of the *pipeline*, not as a claim about real-world Black-Scholes
  accuracy — re-run against real NSE data for a genuine result.
  Black-Scholes typically fares reasonably well near-the-money over
  short horizons and worse on skewed OTM strikes and longer maturities.
- Historical (realised) volatility is a crude, backward-looking proxy
  for the volatility that will actually realise over the option's life;
  most of Black-Scholes' real-world "error" is really volatility
  forecast error, not a flaw in the pricing formula itself.
- The risk-free rate is held constant; real term structures vary by
  maturity.
- No transaction costs, bid-ask spread, or liquidity effects are modelled.
