"""
data_processing.py
==================
Synthetic dataset generators for the FARO platform.

Generates realistic financial, loan, market, and ESG datasets used across
all four FARO engines.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Financial data — monthly P&L / cash flow
# ---------------------------------------------------------------------------

def generate_financial_data(periods: int = 36, seed: int = 42) -> pd.DataFrame:
    """
    Generate monthly financial statements showing realistic growth + seasonality.

    Revenue model:
        base = $5 000 000 / month
        annual growth = 8 %
        Q4 seasonal spike = +15 %
        white-noise sigma = 3 % of trended revenue

    Expenses: 62-68 % of revenue with mild mean-reversion noise.
    Net income: revenue – expenses.
    Operating cash flow: net_income ± working-capital swings.

    Returns
    -------
    pd.DataFrame  with columns: date, revenue, expenses, net_income, cash_flow
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=periods, freq="MS")

    monthly_growth = (1 + 0.08) ** (1 / 12)
    trend = np.array([5_000_000 * (monthly_growth ** i) for i in range(periods)])

    # Seasonality: Q4 months (Oct, Nov, Dec) get +15 %, Q1 slight dip -5 %
    month_nums = dates.month
    seasonality = np.where(month_nums.isin([10, 11, 12]), 1.15,
                  np.where(month_nums.isin([1, 2]), 0.95, 1.00))

    noise = rng.normal(1.0, 0.03, periods)
    revenue = trend * seasonality * noise

    # Expenses: base ratio 65 % with some cost rigidity (autoregressive)
    base_expense_ratio = 0.65
    expense_ratio = np.zeros(periods)
    expense_ratio[0] = base_expense_ratio + rng.normal(0, 0.02)
    for t in range(1, periods):
        expense_ratio[t] = (
            0.7 * expense_ratio[t - 1]
            + 0.3 * base_expense_ratio
            + rng.normal(0, 0.015)
        )
    expense_ratio = np.clip(expense_ratio, 0.55, 0.80)
    expenses = revenue * expense_ratio

    net_income = revenue - expenses

    # Cash flow = net_income + D&A (8 % of expenses) - capex (random 2-5 % rev)
    # plus working-capital swing
    da = expenses * 0.08
    capex = revenue * rng.uniform(0.02, 0.05, periods)
    wc_swing = rng.normal(0, revenue * 0.02)
    cash_flow = net_income + da - capex + wc_swing

    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue.round(2),
        "expenses": expenses.round(2),
        "net_income": net_income.round(2),
        "cash_flow": cash_flow.round(2),
    })

    out_path = RAW_DIR / "financial_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[data_processing] Saved financial data → {out_path}")
    return df


# ---------------------------------------------------------------------------
# 2. Loan data — credit risk dataset
# ---------------------------------------------------------------------------

def generate_loan_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic consumer loan portfolio (~15 % default rate).

    Default probability is a logistic function of:
        - credit_score (negative effect)
        - dti_ratio    (positive effect)
        - annual_income (negative effect, log-scale)
        - interest_rate (positive effect)
        - employment_length (negative effect)
    """
    rng = np.random.default_rng(seed)

    purposes = ["debt_consolidation", "home_improvement", "business",
                "medical", "vacation", "education", "other"]
    ownerships = ["RENT", "OWN", "MORTGAGE"]

    credit_score = rng.integers(550, 850, n).astype(float)
    annual_income = np.exp(rng.normal(np.log(75_000), 0.5, n))
    annual_income = np.clip(annual_income, 25_000, 500_000)
    loan_amount = annual_income * rng.uniform(0.05, 0.60, n)
    loan_amount = np.round(loan_amount / 500) * 500
    interest_rate = (
        3.0
        + (850 - credit_score) / 100
        + rng.normal(0, 0.8, n)
    )
    interest_rate = np.clip(interest_rate, 2.5, 28.0).round(2)
    dti_ratio = rng.beta(2, 5, n) * 60        # 0–60 %
    dti_ratio = np.clip(dti_ratio, 0, 55).round(2)
    employment_length = rng.integers(0, 25, n).astype(float)
    loan_purpose = rng.choice(purposes, n)
    home_ownership = rng.choice(ownerships, n, p=[0.45, 0.15, 0.40])

    # Logistic default model (calibrated for ~15% base default rate)
    log_income = np.log(annual_income)
    z = (
        -2.50
        + (700 - credit_score) * 0.018
        + dti_ratio * 0.040
        - (log_income - 11.0) * 0.35
        + (interest_rate - 10) * 0.09
        - employment_length * 0.025
        + rng.normal(0, 0.55, n)   # idiosyncratic noise
    )
    default_prob = 1 / (1 + np.exp(-z))
    default = (rng.uniform(0, 1, n) < default_prob).astype(int)

    df = pd.DataFrame({
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest_rate,
        "credit_score": credit_score.astype(int),
        "annual_income": annual_income.round(2),
        "dti_ratio": dti_ratio,
        "employment_length": employment_length.astype(int),
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
        "default": default,
    })

    actual_rate = df["default"].mean()
    print(f"[data_processing] Loan data: n={n}, default_rate={actual_rate:.2%}")

    out_path = RAW_DIR / "loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[data_processing] Saved loan data → {out_path}")
    return df


# ---------------------------------------------------------------------------
# 3. Market data — daily equity returns
# ---------------------------------------------------------------------------

_SECTOR_MAP = {
    "AAPL":  "tech",
    "GOOGL": "tech",
    "MSFT":  "tech",
    "JPM":   "finance",
    "GS":    "finance",
    "BLK":   "finance",
    "V":     "finance",
    "JNJ":   "healthcare",
    "PG":    "consumer",
    "XOM":   "energy",
}

def generate_market_data(
    tickers: list[str] | None = None,
    periods: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate daily log-returns for a set of equities using a multi-factor model.

    Factor structure:
        r_i,t = alpha_i + beta_market * F_market,t
                         + beta_sector * F_sector,t
                         + epsilon_i,t

    Drift ≈ 0.08 % / day, daily vol ≈ 1.5 %, sector correlation ≈ 0.4-0.6.

    Returns
    -------
    pd.DataFrame  with columns: date  + one column per ticker (daily returns)
    """
    if tickers is None:
        tickers = list(_SECTOR_MAP.keys())

    rng = np.random.default_rng(seed)
    n = len(tickers)

    start = pd.Timestamp("2024-01-02")
    bdays = pd.bdate_range(start, periods=periods)

    # Market factor
    market_vol = 0.010
    market_factor = rng.normal(0.0004, market_vol, periods)

    # Sector factors
    sectors = list(set(_SECTOR_MAP.get(t, "other") for t in tickers))
    sector_factor = {s: rng.normal(0, 0.006, periods) for s in sectors}

    # Stock-specific parameters
    alpha = rng.normal(0.0002, 0.0001, n)        # daily drift 0.02 %
    beta_market = rng.uniform(0.8, 1.3, n)
    beta_sector = rng.uniform(0.3, 0.7, n)
    idio_vol = rng.uniform(0.008, 0.015, n)      # idiosyncratic 0.8-1.5 %

    returns = np.zeros((periods, n))
    for i, ticker in enumerate(tickers):
        sec = _SECTOR_MAP.get(ticker, "other")
        sf = sector_factor.get(sec, np.zeros(periods))
        idio = rng.normal(0, idio_vol[i], periods)
        returns[:, i] = (
            alpha[i]
            + beta_market[i] * market_factor
            + beta_sector[i] * sf
            + idio
        )

    df = pd.DataFrame(returns, columns=tickers, index=bdays)
    df.index.name = "date"
    df = df.round(6)
    df_out = df.reset_index()

    out_path = RAW_DIR / "market_data.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[data_processing] Saved market data → {out_path}  "
          f"({periods} days × {n} tickers)")
    return df_out


# ---------------------------------------------------------------------------
# 4. ESG scores
# ---------------------------------------------------------------------------

def generate_esg_scores(
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate plausible ESG scores (0–100) for a list of tickers.

    Sector biases mirror real-world patterns:
        - Tech: high governance, moderate social, moderate environmental
        - Finance: high governance, moderate social, lower environmental
        - Energy: low environmental, moderate social
        - Healthcare: high social, moderate governance
        - Consumer: moderate all

    Returns
    -------
    pd.DataFrame  with columns: ticker, environmental_score, social_score,
                  governance_score, overall_esg_score
    """
    if tickers is None:
        tickers = list(_SECTOR_MAP.keys())

    rng = np.random.default_rng(seed)

    sector_priors = {
        "tech":       {"E": 65, "S": 68, "G": 78},
        "finance":    {"E": 55, "S": 65, "G": 80},
        "healthcare": {"E": 60, "S": 80, "G": 72},
        "consumer":   {"E": 62, "S": 70, "G": 68},
        "energy":     {"E": 38, "S": 58, "G": 65},
        "other":      {"E": 60, "S": 65, "G": 68},
    }

    rows = []
    for ticker in tickers:
        sec = _SECTOR_MAP.get(ticker, "other")
        prior = sector_priors[sec]
        e = float(np.clip(rng.normal(prior["E"], 8), 10, 100))
        s = float(np.clip(rng.normal(prior["S"], 7), 10, 100))
        g = float(np.clip(rng.normal(prior["G"], 6), 10, 100))
        overall = round(0.30 * e + 0.30 * s + 0.40 * g, 2)
        rows.append({
            "ticker": ticker,
            "environmental_score": round(e, 2),
            "social_score": round(s, 2),
            "governance_score": round(g, 2),
            "overall_esg_score": overall,
        })

    df = pd.DataFrame(rows)
    out_path = RAW_DIR / "esg_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"[data_processing] Saved ESG scores → {out_path}")
    return df
