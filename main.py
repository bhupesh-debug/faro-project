"""
main.py
=======
FARO — Financial Analytics, Risk & Optimization
Orchestrator: runs all four engines and saves processed outputs.

Usage:
    python main.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (
    generate_esg_scores,
    generate_financial_data,
    generate_loan_data,
    generate_market_data,
)
from src.fpa_engine import FPAEngine
from src.risk_engine import RiskEngine
from src.portfolio_engine import PortfolioEngine
from src.rpa_simulation import RPASimulation
from src.ai_insights import AIInsightsEngine

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "GOOGL", "MSFT", "JPM", "GS", "BLK", "V", "JNJ", "PG", "XOM"]

DIVIDER = "=" * 70


def _banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _section(title: str) -> None:
    print(f"\n  ── {title} ──")


# ===========================================================================
# STEP 1 — Data generation
# ===========================================================================

def step_generate_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _banner("STEP 1 / 5  |  DATA GENERATION")

    t0 = time.perf_counter()
    fin_df = generate_financial_data(periods=36)
    loan_df = generate_loan_data(n=5000)
    market_df = generate_market_data(tickers=TICKERS, periods=252)
    esg_df = generate_esg_scores(tickers=TICKERS)
    elapsed = time.perf_counter() - t0

    _section("Summary")
    print(f"    Financial data   : {len(fin_df)} months")
    print(f"    Loan portfolio   : {len(loan_df):,} loans  "
          f"(default rate: {loan_df['default'].mean():.1%})")
    print(f"    Market data      : {len(market_df)} trading days × {len(TICKERS)} tickers")
    print(f"    ESG scores       : {len(esg_df)} tickers")
    print(f"    Elapsed          : {elapsed:.2f}s")
    return fin_df, loan_df, market_df, esg_df


# ===========================================================================
# STEP 2 — FP&A Engine
# ===========================================================================

def step_fpa(fin_df: pd.DataFrame) -> dict:
    _banner("STEP 2 / 5  |  FP&A ENGINE")

    t0 = time.perf_counter()
    fpa = FPAEngine(fin_df)

    _section("KPIs")
    kpis = fpa.generate_kpis()
    for k, v in kpis.items():
        print(f"    {k:<25s}: {v}")

    _section("Revenue Forecast (next 12 months)")
    rev_fc = fpa.forecast_revenue(periods=12)
    print(rev_fc.to_string(index=False))

    _section("Expense Forecast (next 12 months)")
    exp_fc = fpa.forecast_expenses(periods=12)
    print(exp_fc.to_string(index=False))

    _section("Variance Analysis (last 6 months)")
    var_df = fpa.variance_analysis()
    print(var_df.tail(6).to_string(index=False))

    print(f"\n{fpa.get_summary()}")

    # Save processed outputs
    rev_fc.to_csv(PROCESSED_DIR / "revenue_forecast.csv", index=False)
    exp_fc.to_csv(PROCESSED_DIR / "expense_forecast.csv", index=False)
    var_df.to_csv(PROCESSED_DIR / "variance_analysis.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n    [FP&A] Done — {elapsed:.2f}s")
    return kpis


# ===========================================================================
# STEP 3 — Risk Engine
# ===========================================================================

def step_risk(loan_df: pd.DataFrame, market_df: pd.DataFrame) -> dict:
    _banner("STEP 3 / 5  |  RISK ENGINE")

    t0 = time.perf_counter()
    risk = RiskEngine(loan_df, market_df)

    _section("Training Credit Risk Model")
    metrics = risk.train_credit_model()
    for k, v in metrics.items():
        print(f"    {k:<12s}: {v}")

    _section("Sample Default Prediction")
    sample_loan = {
        "loan_amount": 25_000,
        "interest_rate": 14.5,
        "credit_score": 640,
        "annual_income": 72_000,
        "dti_ratio": 28.5,
        "employment_length": 4,
        "loan_purpose": "debt_consolidation",
        "home_ownership": "RENT",
    }
    prob = risk.predict_default(sample_loan)
    print(f"    Sample loan default probability: {prob:.4f}  "
          f"({'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.3 else 'LOW'} risk)")

    _section("Risk Distribution")
    dist = risk.risk_distribution()
    print(dist.to_string(index=False))

    _section("Market Risk (equal-weight $10M portfolio)")
    var_info = risk.calculate_var(confidence=0.95, horizon=1)
    for k, v in var_info.items():
        print(f"    {k:<20s}: {v}")

    _section("Annualised Volatility")
    vol_df = risk.calculate_volatility()
    print(vol_df.to_string(index=False))

    risk_summary = risk.get_risk_summary()
    print(f"\n{risk_summary}")

    # Save
    dist.to_csv(PROCESSED_DIR / "risk_distribution.csv", index=False)
    vol_df.to_csv(PROCESSED_DIR / "volatility.csv", index=False)

    # Attach extra metrics for AI engine
    risk_metrics_ext = {**metrics, **var_info}
    high_row = dist[dist["risk_category"] == "High"]
    if not high_row.empty:
        risk_metrics_ext["high_risk_pct"] = float(high_row["pct"].values[0])

    elapsed = time.perf_counter() - t0
    print(f"\n    [Risk] Done — {elapsed:.2f}s")
    return {"summary": risk_summary, "metrics": risk_metrics_ext, "engine": risk}


# ===========================================================================
# STEP 4 — Portfolio Engine
# ===========================================================================

def step_portfolio(market_df: pd.DataFrame, esg_df: pd.DataFrame) -> dict:
    _banner("STEP 4 / 5  |  PORTFOLIO ENGINE")

    t0 = time.perf_counter()
    port = PortfolioEngine(market_df, esg_df)

    _section("Efficient Frontier (5,000 portfolios)")
    frontier = port.efficient_frontier(n_portfolios=5_000)
    max_sharpe_row = frontier.loc[frontier["sharpe_ratio"].idxmax()]
    min_vol_row = frontier.loc[frontier["volatility"].idxmin()]
    print(f"    Max Sharpe portfolio → "
          f"return={max_sharpe_row['return']*100:.2f}%  "
          f"vol={max_sharpe_row['volatility']*100:.2f}%  "
          f"sharpe={max_sharpe_row['sharpe_ratio']:.4f}")
    print(f"    Min Vol portfolio   → "
          f"return={min_vol_row['return']*100:.2f}%  "
          f"vol={min_vol_row['volatility']*100:.2f}%  "
          f"sharpe={min_vol_row['sharpe_ratio']:.4f}")

    _section("Optimal Portfolio (Max Sharpe)")
    opt = port.optimize_portfolio(target="max_sharpe")
    for t, w in opt["optimal_weights"].items():
        print(f"    {t:<6s}: {w * 100:.2f}%")
    print(f"    Expected Return : {opt['expected_return']:.2f}%")
    print(f"    Volatility      : {opt['volatility']:.2f}%")
    print(f"    Sharpe Ratio    : {opt['sharpe_ratio']:.4f}")

    _section("ESG-Constrained Portfolio (min ESG=60)")
    esg_opt = port.esg_weighted_optimization(min_esg=60)
    print(f"    Expected Return : {esg_opt['expected_return']:.2f}%")
    print(f"    Volatility      : {esg_opt['volatility']:.2f}%")
    print(f"    Sharpe Ratio    : {esg_opt['sharpe_ratio']:.4f}")
    print(f"    Portfolio ESG   : {esg_opt['esg_score']:.2f}")

    _section("Beta Analysis")
    beta_df = port.calculate_beta()
    print(beta_df.to_string(index=False))

    port_summary = port.get_portfolio_summary()
    print(f"\n{port_summary}")

    # Save
    frontier.to_csv(PROCESSED_DIR / "efficient_frontier.csv", index=False)
    beta_df.to_csv(PROCESSED_DIR / "beta_analysis.csv", index=False)
    pd.DataFrame([opt["optimal_weights"]]).to_csv(
        PROCESSED_DIR / "optimal_weights.csv", index=False
    )

    elapsed = time.perf_counter() - t0
    print(f"\n    [Portfolio] Done — {elapsed:.2f}s")
    return {"summary": port_summary, "metrics": opt}


# ===========================================================================
# STEP 5 — RPA Simulation
# ===========================================================================

def step_rpa() -> dict:
    _banner("STEP 5 / 5  |  RPA SIMULATION")

    t0 = time.perf_counter()
    rpa = RPASimulation(seed=42)

    _section("Loan Processing (100 loans)")
    loan_log = rpa.simulate_loan_processing(n_loans=100)
    status_counts = loan_log.groupby("status")["loan_id"].nunique()
    print(status_counts.to_string())
    avg_ms = loan_log.groupby("step")["processing_time_ms"].mean().round(1)
    print(f"\n    Avg processing time per step (ms):\n{avg_ms.to_string()}")

    _section("GL Reconciliation (500 transactions)")
    recon = rpa.simulate_reconciliation(n_transactions=500)
    print(recon["summary"].to_string(index=False))
    print(f"\n    Exceptions:\n{recon['exceptions'].head(10).to_string(index=False)}")

    _section("Month-End Financial Close (2026-03)")
    close_df = rpa.simulate_financial_close(period="2026-03")
    print(close_df[["task", "owner", "estimated_hours",
                     "actual_hours", "automated", "status"]].to_string(index=False))

    _section("Automation Metrics")
    auto_metrics = rpa.calculate_automation_metrics()
    for k, v in auto_metrics.items():
        print(f"    {k:<35s}: {v}")

    # Save
    loan_log.to_csv(PROCESSED_DIR / "loan_processing_log.csv", index=False)
    recon["summary"].to_csv(PROCESSED_DIR / "reconciliation_summary.csv", index=False)
    recon["exceptions"].to_csv(PROCESSED_DIR / "reconciliation_exceptions.csv", index=False)
    close_df.to_csv(PROCESSED_DIR / "financial_close_checklist.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"\n    [RPA] Done — {elapsed:.2f}s")
    return {"metrics": auto_metrics}


# ===========================================================================
# AI Insights
# ===========================================================================

def step_ai_insights(
    kpis: dict,
    risk_data: dict,
    portfolio_data: dict,
) -> None:
    _banner("AI INSIGHTS ENGINE")

    ai = AIInsightsEngine(
        fpa_kpis=kpis,
        risk_summary=risk_data["summary"],
        portfolio_summary=portfolio_data["summary"],
        risk_metrics=risk_data["metrics"],
        portfolio_metrics=portfolio_data["metrics"],
    )

    _section("CFO Insights")
    for i, insight in enumerate(ai.generate_cfo_insights(), 1):
        print(f"\n  [{i}] {insight}")

    _section("Risk Alerts")
    for i, alert in enumerate(ai.generate_risk_alerts(), 1):
        print(f"\n  [{i}] {alert}")

    _section("Strategic Recommendations")
    for i, rec in enumerate(ai.generate_recommendations(), 1):
        print(f"\n  [{i}] {rec}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    wall_start = time.perf_counter()

    print(DIVIDER)
    print("  FARO — Financial Analytics, Risk & Optimization")
    print("  Version 1.0  |  github.com/faro-finance/faro")
    print(DIVIDER)

    # 1. Generate datasets
    fin_df, loan_df, market_df, esg_df = step_generate_data()

    # 2. FP&A Engine
    kpis = step_fpa(fin_df)

    # 3. Risk Engine
    risk_data = step_risk(loan_df, market_df)

    # 4. Portfolio Engine
    portfolio_data = step_portfolio(market_df, esg_df)

    # 5. RPA Simulation
    step_rpa()

    # 6. AI Insights (no data generation needed)
    step_ai_insights(kpis, risk_data, portfolio_data)

    wall_elapsed = time.perf_counter() - wall_start
    _banner(f"FARO COMPLETE  |  Total runtime: {wall_elapsed:.1f}s")
    print(f"  Processed outputs saved to: {PROCESSED_DIR}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
