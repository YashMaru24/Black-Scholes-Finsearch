"""
backtest_accuracy.py
---------------------
Assesses how accurately the Black-Scholes model reproduces observed
("market") Nifty50 index-option prices.

Workflow
--------
1. Load data/nifty50_options_sample.csv (trade_date, expiry_date, spot,
   strike, option_type, days_to_expiry, risk_free_rate, hist_vol_20d,
   market_price).
2. For every quote, price the option with Black-Scholes using the trailing
   20-day realised volatility (hist_vol_20d) as the volatility input --
   this is the classic, honest way to test BS out-of-sample: you are NOT
   allowed to back out sigma from the same price you're trying to predict
   (that would be implied vol, which is circular and always "fits").
3. Compute error metrics overall and by moneyness / maturity bucket.
4. Also compute each quote's Black-Scholes IMPLIED volatility (for
   diagnostic purposes, e.g. to plot the volatility smile the market
   embeds -- BS assumes a flat surface, so its shape is itself a
   measure of the model's mis-specification).
5. Save a metrics summary (CSV) and a few diagnostic plots to outputs/.

Run:
    python src/backtest_accuracy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from black_scholes import bs_price, implied_vol, mc_price

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nifty50_options_sample.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["trade_date", "expiry_date"])
    df["T"] = df["days_to_expiry"] / 365.0
    df["moneyness"] = df["strike"] / df["spot"]
    return df


def price_with_black_scholes(df):
    # bs_price is vectorised over S/K/T/r/sigma but option_type is a single
    # string flag, so calls and puts are priced in two vectorised passes.
    df["bs_price"] = np.nan
    is_call = df["option_type"] == "call"
    df.loc[is_call, "bs_price"] = bs_price(
        df.loc[is_call, "spot"], df.loc[is_call, "strike"], df.loc[is_call, "T"],
        df.loc[is_call, "risk_free_rate"], df.loc[is_call, "hist_vol_20d"], "call")
    df.loc[~is_call, "bs_price"] = bs_price(
        df.loc[~is_call, "spot"], df.loc[~is_call, "strike"], df.loc[~is_call, "T"],
        df.loc[~is_call, "risk_free_rate"], df.loc[~is_call, "hist_vol_20d"], "put")

    df["error"] = df["bs_price"] - df["market_price"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df["market_price"] * 100
    return df


def add_implied_vol(df):
    ivs = []
    for row in df.itertuples(index=False):
        iv = implied_vol(row.market_price, row.spot, row.strike, row.T,
                          row.risk_free_rate, row.option_type)
        ivs.append(iv)
    df["implied_vol"] = ivs
    return df


def bucket_labels(df):
    bins = [0, 0.95, 0.99, 1.01, 1.05, np.inf]
    labels = ["Deep OTM/ITM (<0.95)", "OTM/ITM (0.95-0.99)",
              "ATM (0.99-1.01)", "ITM/OTM (1.01-1.05)", "Deep ITM/OTM (>1.05)"]
    df["moneyness_bucket"] = pd.cut(df["moneyness"], bins=bins, labels=labels)

    tbins = [0, 7, 30, 60, np.inf]
    tlabels = ["<=1wk", "1wk-1m", "1-2m", ">2m"]
    df["maturity_bucket"] = pd.cut(df["days_to_expiry"], bins=tbins, labels=tlabels)
    return df


def compute_metrics(group):
    mae = group["abs_error"].mean()
    rmse = np.sqrt((group["error"] ** 2).mean())
    mape = group["pct_error"].abs().mean()
    bias = group["error"].mean()  # signed -> systematic over/under pricing
    corr = group["bs_price"].corr(group["market_price"])
    return pd.Series({"n": len(group), "MAE": mae, "RMSE": rmse,
                       "MAPE_%": mape, "Bias": bias, "Corr": corr})


def run():
    if not DATA_PATH.exists():
        print("No dataset found -- generating synthetic sample dataset first...")
        from generate_sample_data import generate_dataset
        generate_dataset()

    df = load_data()
    df = price_with_black_scholes(df)
    df = bucket_labels(df)

    print(f"Loaded {len(df):,} option quotes")

    overall = compute_metrics(df)
    print("\n=== Overall accuracy ===")
    print(overall.to_string())

    by_type = df.groupby("option_type").apply(compute_metrics)
    by_moneyness = df.groupby("moneyness_bucket", observed=True).apply(compute_metrics)
    by_maturity = df.groupby("maturity_bucket", observed=True).apply(compute_metrics)

    print("\n=== By option type ===")
    print(by_type)
    print("\n=== By moneyness ===")
    print(by_moneyness)
    print("\n=== By maturity ===")
    print(by_maturity)

    # Save metrics
    overall.to_frame("value").to_csv(OUT_DIR / "metrics_overall.csv")
    by_type.to_csv(OUT_DIR / "metrics_by_option_type.csv")
    by_moneyness.to_csv(OUT_DIR / "metrics_by_moneyness.csv")
    by_maturity.to_csv(OUT_DIR / "metrics_by_maturity.csv")
    df.to_csv(OUT_DIR / "priced_quotes_full.csv", index=False)

    # ---- Diagnostic plots ----
    plot_scatter(df)
    plot_error_distribution(df)
    plot_error_by_moneyness(df)

    # ---- Implied vol smile + Monte-Carlo cross-check on a sample ----
    sample = df.sample(min(300, len(df)), random_state=0).copy()
    sample = add_implied_vol(sample)
    plot_smile(sample)
    monte_carlo_crosscheck(df)

    print(f"\nAll metrics and plots saved to: {OUT_DIR}")


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["market_price"], df["bs_price"], s=4, alpha=0.3)
    lim = [0, df[["market_price", "bs_price"]].max().max() * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="Perfect agreement")
    ax.set_xlabel("Market price (Rs)")
    ax.set_ylabel("Black-Scholes price (Rs)")
    ax.set_title("Black-Scholes vs Market Price - Nifty50 Options")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bs_vs_market_scatter.png", dpi=150)
    plt.close(fig)


def plot_error_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["pct_error"].clip(-100, 100), bins=60)
    ax.axvline(0, color="r", lw=1)
    ax.set_xlabel("Pricing error (% of market price)")
    ax.set_ylabel("Number of quotes")
    ax.set_title("Distribution of Black-Scholes Pricing Error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_distribution.png", dpi=150)
    plt.close(fig)


def plot_error_by_moneyness(df):
    grp = df.groupby("moneyness_bucket", observed=True)["pct_error"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    grp.plot(kind="bar", ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean % pricing error")
    ax.set_title("Black-Scholes Bias by Moneyness")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_by_moneyness.png", dpi=150)
    plt.close(fig)


def plot_smile(sample):
    fig, ax = plt.subplots(figsize=(6, 4))
    for opt_type, marker in [("call", "o"), ("put", "x")]:
        sub = sample[sample["option_type"] == opt_type]
        ax.scatter(sub["moneyness"], sub["implied_vol"], s=10, marker=marker, label=opt_type)
    ax.set_xlabel("Moneyness (Strike / Spot)")
    ax.set_ylabel("Black-Scholes implied volatility")
    ax.set_title("Implied Volatility Smile/Skew (sample of 300 quotes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "implied_vol_smile.png", dpi=150)
    plt.close(fig)


def monte_carlo_crosscheck(df, n=15):
    """
    Sanity-check the closed-form Black-Scholes formula against a Monte-Carlo
    simulation of the same lognormal (GBM) model on a handful of contracts.
    They should agree to within Monte-Carlo standard error -- this validates
    the analytic implementation rather than measuring market accuracy.
    """
    sample = df.sample(n, random_state=1)
    records = []
    for row in sample.itertuples(index=False):
        mc_p, se = mc_price(row.spot, row.strike, row.T, row.risk_free_rate,
                             row.hist_vol_20d, row.option_type, n_paths=200_000, seed=0)
        records.append({
            "option_type": row.option_type, "strike": row.strike,
            "days_to_expiry": row.days_to_expiry,
            "bs_price": round(row.bs_price, 2),
            "mc_price": round(mc_p, 2), "mc_stderr": round(se, 3),
        })
    mc_df = pd.DataFrame(records)
    mc_df.to_csv(OUT_DIR / "monte_carlo_crosscheck.csv", index=False)
    print("\n=== Monte-Carlo vs closed-form Black-Scholes (model validation) ===")
    print(mc_df.to_string(index=False))


if __name__ == "__main__":
    run()
