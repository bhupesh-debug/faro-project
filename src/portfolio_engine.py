"""
portfolio_engine.py
===================
Portfolio Optimization Engine for FARO.

Implements:
  - Efficient frontier via Monte Carlo simulation
  - Maximum Sharpe / Minimum Volatility optimisation (scipy)
  - ESG-constrained optimisation
  - Beta calculation
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

RISK_FREE_RATE_ANNUAL = 0.0525  # approx 1-year T-bill 2025
RISK_FREE_DAILY = RISK_FREE_RATE_ANNUAL / 252


class PortfolioEngine:
    """
    Portfolio optimisation using mean-variance theory.

    Parameters
    ----------
    market_data : pd.DataFrame  — output of generate_market_data()
    esg_data    : pd.DataFrame  — output of generate_esg_scores()
    """

    def __init__(
        self, market_data: pd.DataFrame, esg_data: pd.DataFrame
    ) -> None:
        self.market_df = market_data.copy()
        self.esg_df = esg_data.copy()

        if "date" in self.market_df.columns:
            self.returns = self.market_df.drop(columns=["date"])
        else:
            self.returns = self.market_df.copy()

        self.tickers = list(self.returns.columns)
        self.n = len(self.tickers)

        # Annualised mean returns and covariance matrix
        self.mu = self.returns.mean() * 252          # (n,)
        self.cov = self.returns.cov() * 252          # (n, n)

        # ESG scores aligned to tickers
        esg_idx = esg_data.set_index("ticker")
        self.esg_scores = np.array([
            esg_idx.loc[t, "overall_esg_score"] if t in esg_idx.index else 60.0
            for t in self.tickers
        ])

        self._optimal_weights: np.ndarray | None = None
        self._summary_text: str = ""

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(self, n_portfolios: int = 5000) -> pd.DataFrame:
        """
        Monte Carlo simulation of random portfolio allocations.

        Returns
        -------
        pd.DataFrame : weight_<ticker>..., return, volatility, sharpe_ratio
        """
        rng = np.random.default_rng(42)
        results = []

        mu = self.mu.values
        cov = self.cov.values

        for _ in range(n_portfolios):
            raw = rng.exponential(1.0, self.n)
            w = raw / raw.sum()

            port_return = float(w @ mu)
            port_var = float(w @ cov @ w)
            port_vol = np.sqrt(port_var)
            sharpe = (port_return - RISK_FREE_RATE_ANNUAL) / port_vol

            row: dict[str, Any] = {f"w_{t}": round(float(w[i]), 6)
                                   for i, t in enumerate(self.tickers)}
            row["return"] = round(port_return, 6)
            row["volatility"] = round(port_vol, 6)
            row["sharpe_ratio"] = round(float(sharpe), 6)
            results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Mean-variance optimisation
    # ------------------------------------------------------------------

    def optimize_portfolio(
        self, target: str = "max_sharpe"
    ) -> dict[str, Any]:
        """
        Scipy-based portfolio optimisation.

        Parameters
        ----------
        target : 'max_sharpe' | 'min_volatility'

        Returns
        -------
        dict : optimal_weights, expected_return, volatility, sharpe_ratio
        """
        mu = self.mu.values
        cov = self.cov.values

        w0 = np.ones(self.n) / self.n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.0, 1.0)] * self.n

        if target == "max_sharpe":
            def neg_sharpe(w: np.ndarray) -> float:
                ret = float(w @ mu)
                vol = float(np.sqrt(w @ cov @ w))
                return -(ret - RISK_FREE_RATE_ANNUAL) / max(vol, 1e-9)
            obj = neg_sharpe
        else:  # min_volatility
            def portfolio_vol(w: np.ndarray) -> float:
                return float(np.sqrt(w @ cov @ w))
            obj = portfolio_vol

        result = minimize(
            obj, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not result.success:
            # Fallback to equal weight
            opt_w = w0
        else:
            opt_w = result.x
            opt_w = np.clip(opt_w, 0, 1)
            opt_w /= opt_w.sum()

        port_return = float(opt_w @ mu)
        port_vol = float(np.sqrt(opt_w @ cov @ opt_w))
        sharpe = (port_return - RISK_FREE_RATE_ANNUAL) / max(port_vol, 1e-9)

        self._optimal_weights = opt_w

        return {
            "target": target,
            "optimal_weights": {
                t: round(float(opt_w[i]), 4) for i, t in enumerate(self.tickers)
            },
            "expected_return": round(port_return * 100, 2),   # %
            "volatility": round(port_vol * 100, 2),            # %
            "sharpe_ratio": round(sharpe, 4),
        }

    # ------------------------------------------------------------------
    # ESG-constrained optimisation
    # ------------------------------------------------------------------

    def esg_weighted_optimization(self, min_esg: float = 60) -> dict[str, Any]:
        """
        Mean-variance optimisation with a minimum weighted-average ESG score.

        Parameters
        ----------
        min_esg : minimum acceptable portfolio ESG score (default 60)

        Returns
        -------
        dict : same structure as optimize_portfolio() + esg_score
        """
        mu = self.mu.values
        cov = self.cov.values
        esg = self.esg_scores

        w0 = np.ones(self.n) / self.n
        constraints = [
            {"type": "eq",  "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: float(w @ esg) - min_esg},
        ]
        bounds = [(0.0, 1.0)] * self.n

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov @ w))
            return -(ret - RISK_FREE_RATE_ANNUAL) / max(vol, 1e-9)

        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )

        if not result.success:
            opt_w = w0
        else:
            opt_w = result.x
            opt_w = np.clip(opt_w, 0, 1)
            opt_w /= opt_w.sum()

        port_return = float(opt_w @ mu)
        port_vol = float(np.sqrt(opt_w @ cov @ opt_w))
        sharpe = (port_return - RISK_FREE_RATE_ANNUAL) / max(port_vol, 1e-9)
        achieved_esg = float(opt_w @ esg)

        return {
            "target": "max_sharpe_esg_constrained",
            "min_esg_constraint": min_esg,
            "optimal_weights": {
                t: round(float(opt_w[i]), 4) for i, t in enumerate(self.tickers)
            },
            "expected_return": round(port_return * 100, 2),
            "volatility": round(port_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "esg_score": round(achieved_esg, 2),
        }

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    def calculate_beta(self, benchmark_ticker: str = "SPY") -> pd.DataFrame:
        """
        Calculate CAPM beta for each stock relative to a benchmark.

        If benchmark_ticker is not in the data, synthetic SPY returns are
        generated using the average market return.

        Returns
        -------
        pd.DataFrame : ticker, beta, correlation
        """
        ret = self.returns

        if benchmark_ticker in ret.columns:
            benchmark = ret[benchmark_ticker].values
        else:
            # Synthetic benchmark: equal-weight of all stocks ± noise
            rng = np.random.default_rng(99)
            benchmark = ret.mean(axis=1).values + rng.normal(0, 0.002, len(ret))

        bench_var = float(np.var(benchmark, ddof=1))
        rows = []
        for ticker in self.tickers:
            stock = ret[ticker].values
            cov_sb = float(np.cov(stock, benchmark, ddof=1)[0, 1])
            beta = cov_sb / bench_var if bench_var > 0 else 1.0
            corr = float(np.corrcoef(stock, benchmark)[0, 1])
            rows.append({
                "ticker": ticker,
                "beta": round(beta, 4),
                "correlation_to_benchmark": round(corr, 4),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> str:
        """Return a formatted text summary of portfolio optimisation results."""
        opt = self.optimize_portfolio(target="max_sharpe")
        esg_opt = self.esg_weighted_optimization(min_esg=60)
        beta_df = self.calculate_beta()

        lines = [
            "=" * 60,
            "  PORTFOLIO ENGINE SUMMARY",
            "=" * 60,
            "  [Max-Sharpe Optimal Portfolio]",
            f"    Expected Return : {opt['expected_return']:>6.2f}%",
            f"    Volatility      : {opt['volatility']:>6.2f}%",
            f"    Sharpe Ratio    : {opt['sharpe_ratio']:>6.4f}",
            "",
            "  Top 5 weights:",
        ]
        sorted_w = sorted(opt["optimal_weights"].items(),
                          key=lambda kv: kv[1], reverse=True)
        for ticker, w in sorted_w[:5]:
            lines.append(f"    {ticker:<6s} : {w * 100:.2f}%")

        lines += [
            "",
            "  [ESG-Constrained Portfolio (min ESG=60)]",
            f"    Expected Return : {esg_opt['expected_return']:>6.2f}%",
            f"    Volatility      : {esg_opt['volatility']:>6.2f}%",
            f"    Sharpe Ratio    : {esg_opt['sharpe_ratio']:>6.4f}",
            f"    Achieved ESG    : {esg_opt['esg_score']:>6.2f}",
            "",
            "  [Beta vs Synthetic Market]",
        ]
        for _, row in beta_df.iterrows():
            lines.append(
                f"    {row['ticker']:<6s} β={row['beta']:.3f}"
                f"  ρ={row['correlation_to_benchmark']:.3f}"
            )
        lines.append("=" * 60)

        text = "\n".join(lines)
        self._summary_text = text
        return text
