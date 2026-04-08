"""
fpa_engine.py
=============
FP&A Intelligence Engine for FARO.

Provides time-series forecasting, variance analysis, and KPI calculation
for monthly financial statement data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class FPAEngine:
    """
    Financial Planning & Analysis engine.

    Parameters
    ----------
    financial_data : pd.DataFrame
        Monthly DataFrame with columns: date, revenue, expenses,
        net_income, cash_flow.
    """

    def __init__(self, financial_data: pd.DataFrame) -> None:
        df = financial_data.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        self.df = df
        self._kpis: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Revenue forecast
    # ------------------------------------------------------------------

    def forecast_revenue(self, periods: int = 12) -> pd.DataFrame:
        """
        ARIMA(1,1,1) revenue forecast with 95 % confidence intervals.

        Returns
        -------
        pd.DataFrame : date, forecast, lower_ci, upper_ci
        """
        return self._arima_forecast("revenue", periods)

    # ------------------------------------------------------------------
    # Expense forecast
    # ------------------------------------------------------------------

    def forecast_expenses(self, periods: int = 12) -> pd.DataFrame:
        """
        ARIMA(1,1,1) expense forecast with 95 % confidence intervals.

        Returns
        -------
        pd.DataFrame : date, forecast, lower_ci, upper_ci
        """
        return self._arima_forecast("expenses", periods)

    # ------------------------------------------------------------------
    # Variance analysis
    # ------------------------------------------------------------------

    def variance_analysis(self) -> pd.DataFrame:
        """
        Compute actual vs. rolling-3-month budget variance.

        Budget is defined as the 3-month rolling average of the previous
        periods (shifted by 1 so budget is set before actuals are known).

        Returns
        -------
        pd.DataFrame : period, actual, budget, variance, variance_pct
        """
        df = self.df.copy()
        # Budget = 3-month rolling mean shifted forward (lag 1)
        df["budget"] = (
            df["revenue"]
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
        )
        # Fill the first budget with the first revenue value
        df["budget"] = df["budget"].fillna(df["revenue"].iloc[0])

        df["variance"] = df["revenue"] - df["budget"]
        df["variance_pct"] = (df["variance"] / df["budget"] * 100).round(2)

        result = df[["date", "revenue", "budget", "variance", "variance_pct"]].copy()
        result = result.rename(columns={"date": "period", "revenue": "actual"})
        result[["actual", "budget", "variance"]] = result[
            ["actual", "budget", "variance"]
        ].round(2)
        return result

    # ------------------------------------------------------------------
    # KPI generation
    # ------------------------------------------------------------------

    def generate_kpis(self) -> dict[str, float | str]:
        """
        Compute standard FP&A KPIs from the financial dataset.

        Returns
        -------
        dict with keys:
            revenue_growth_yoy     – % YoY revenue growth (last 12 m vs prior 12 m)
            expense_ratio          – avg expenses / revenue (%)
            net_margin             – avg net income / revenue (%)
            cash_flow_trend        – linear slope of cash_flow ($/month)
            operating_leverage     – % Δ net_income / % Δ revenue (last vs prev year)
        """
        df = self.df.copy()
        n = len(df)

        # YoY growth (compare last 12 months vs previous 12 months)
        if n >= 24:
            rev_last12 = df["revenue"].iloc[-12:].sum()
            rev_prev12 = df["revenue"].iloc[-24:-12].sum()
            rev_growth_yoy = (rev_last12 / rev_prev12 - 1) * 100
        else:
            # Annualise from available data
            rev_growth_yoy = (
                (df["revenue"].iloc[-1] / df["revenue"].iloc[0])
                ** (12 / max(n - 1, 1))
                - 1
            ) * 100

        expense_ratio = (df["expenses"] / df["revenue"]).mean() * 100
        net_margin = (df["net_income"] / df["revenue"]).mean() * 100

        # Cash flow linear trend
        x = np.arange(n, dtype=float)
        cf = df["cash_flow"].values
        slope, _ = np.polyfit(x, cf, 1)

        # Operating leverage (last year vs prior)
        if n >= 24:
            ni_last = df["net_income"].iloc[-12:].sum()
            ni_prev = df["net_income"].iloc[-24:-12].sum()
            rev_last = df["revenue"].iloc[-12:].sum()
            rev_prev = df["revenue"].iloc[-24:-12].sum()
            pct_ni = (ni_last / ni_prev - 1) * 100 if ni_prev != 0 else 0
            pct_rev = (rev_last / rev_prev - 1) * 100 if rev_prev != 0 else 0
            op_lev = pct_ni / pct_rev if pct_rev != 0 else float("nan")
        else:
            op_lev = float("nan")

        kpis = {
            "revenue_growth_yoy": round(rev_growth_yoy, 2),
            "expense_ratio": round(expense_ratio, 2),
            "net_margin": round(net_margin, 2),
            "cash_flow_trend": round(float(slope), 2),
            "operating_leverage": round(op_lev, 2) if not np.isnan(op_lev) else "N/A",
        }
        self._kpis = kpis
        return kpis

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> str:
        """
        Return a human-readable text summary of key financial metrics.
        """
        if self._kpis is None:
            self.generate_kpis()
        kpis = self._kpis
        df = self.df

        latest_rev = df["revenue"].iloc[-1]
        latest_margin = df["net_income"].iloc[-1] / df["revenue"].iloc[-1] * 100
        avg_cf = df["cash_flow"].mean()
        cf_trend = "positive" if kpis["cash_flow_trend"] > 0 else "negative"

        lines = [
            "=" * 60,
            "  FP&A ENGINE SUMMARY",
            "=" * 60,
            f"  Latest Monthly Revenue  : ${latest_rev:>12,.0f}",
            f"  Revenue Growth (YoY)    : {kpis['revenue_growth_yoy']:>7.1f} %",
            f"  Average Expense Ratio   : {kpis['expense_ratio']:>7.1f} %",
            f"  Average Net Margin      : {kpis['net_margin']:>7.1f} %",
            f"  Latest Net Margin       : {latest_margin:>7.1f} %",
            f"  Avg Monthly Cash Flow   : ${avg_cf:>12,.0f}",
            f"  Cash Flow Trend         : {cf_trend} (${kpis['cash_flow_trend']:,.0f}/month)",
            f"  Operating Leverage      : {kpis['operating_leverage']}x",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _arima_forecast(self, col: str, periods: int) -> pd.DataFrame:
        """Fit ARIMA(1,1,1) on `col` and return forecast DataFrame."""
        series = self.df[col].values.astype(float)
        last_date = pd.to_datetime(self.df["date"].iloc[-1])

        try:
            model = ARIMA(series, order=(1, 1, 1))
            result = model.fit()
            fc = result.get_forecast(steps=periods)
            mean_fc = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            lower = ci[:, 0]
            upper = ci[:, 1]
        except Exception:
            # Fallback: naive drift model
            drift = np.mean(np.diff(series))
            mean_fc = series[-1] + drift * np.arange(1, periods + 1)
            se = np.std(np.diff(series)) * np.sqrt(np.arange(1, periods + 1))
            lower = mean_fc - 1.96 * se
            upper = mean_fc + 1.96 * se

        future_dates = pd.date_range(
            last_date + pd.DateOffset(months=1),
            periods=periods,
            freq="MS",
        )

        return pd.DataFrame({
            "date": future_dates,
            "forecast": np.round(mean_fc, 2),
            "lower_ci": np.round(lower, 2),
            "upper_ci": np.round(upper, 2),
        })
