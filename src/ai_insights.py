"""
ai_insights.py
==============
AI Insights Engine for FARO.

Rule-based insight generation that analyses KPIs, risk metrics, and
portfolio data to produce CFO-level commentary and strategic recommendations.
No external API calls required.
"""

from __future__ import annotations

from typing import Any


class AIInsightsEngine:
    """
    Generate actionable financial insights from FARO engine outputs.

    Parameters
    ----------
    fpa_kpis         : dict  — output of FPAEngine.generate_kpis()
    risk_summary     : str   — output of RiskEngine.get_risk_summary()
    portfolio_summary: str   — output of PortfolioEngine.get_portfolio_summary()
    risk_metrics     : dict  — optional, raw risk metrics for richer analysis
    portfolio_metrics: dict  — optional, raw portfolio metrics
    """

    def __init__(
        self,
        fpa_kpis: dict[str, Any],
        risk_summary: str,
        portfolio_summary: str,
        risk_metrics: dict[str, Any] | None = None,
        portfolio_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.kpis = fpa_kpis
        self.risk_summary = risk_summary
        self.portfolio_summary = portfolio_summary
        self.risk_metrics = risk_metrics or {}
        self.portfolio_metrics = portfolio_metrics or {}

    # ------------------------------------------------------------------
    # CFO Insights
    # ------------------------------------------------------------------

    def generate_cfo_insights(self) -> list[str]:
        """
        Generate 5-6 actionable CFO-level insights based on financial KPIs.

        Insights are triggered by threshold conditions on each KPI.

        Returns
        -------
        list[str] : ordered list of insight strings
        """
        kpis = self.kpis
        insights: list[str] = []

        # --- Revenue growth ---
        rev_growth = float(kpis.get("revenue_growth_yoy", 0))
        if rev_growth > 15:
            insights.append(
                f"REVENUE MOMENTUM: YoY revenue growth of {rev_growth:.1f}% "
                "significantly exceeds the 8% base plan. Evaluate capacity "
                "constraints and consider accelerating capital deployment to "
                "capture sustained demand."
            )
        elif rev_growth > 8:
            insights.append(
                f"REVENUE ON TRACK: Revenue growth of {rev_growth:.1f}% is "
                "aligned with the annual growth target. Maintain current "
                "go-to-market strategy while monitoring pipeline coverage."
            )
        elif rev_growth > 0:
            insights.append(
                f"REVENUE UNDERPERFORMANCE: Revenue growth of {rev_growth:.1f}% "
                "trails the 8% target. Immediate review of sales pipeline, "
                "pricing strategy, and customer retention programmes is recommended."
            )
        else:
            insights.append(
                f"CRITICAL: Revenue declined {abs(rev_growth):.1f}% YoY. "
                "Convene an emergency strategic review. Assess whether the decline "
                "is cyclical, competitive, or structural."
            )

        # --- Net margin ---
        net_margin = float(kpis.get("net_margin", 0))
        if net_margin >= 25:
            insights.append(
                f"MARGIN STRENGTH: Net margin of {net_margin:.1f}% indicates "
                "exceptional profitability. Consider returning capital to shareholders "
                "via buybacks or special dividends, or reinvesting in high-ROI initiatives."
            )
        elif net_margin >= 15:
            insights.append(
                f"HEALTHY MARGINS: Net margin of {net_margin:.1f}% is within the "
                "target range. Maintain cost discipline and monitor vendor contract "
                "renewals to protect margin trajectory."
            )
        elif net_margin >= 5:
            insights.append(
                f"MARGIN COMPRESSION: Net margin of {net_margin:.1f}% is below "
                "the 15% benchmark. Conduct a zero-based budgeting exercise focusing "
                "on the top 20% of cost lines to recapture 200-400 bps."
            )
        else:
            insights.append(
                f"MARGIN ALERT: Net margin of {net_margin:.1f}% is critically low. "
                "Immediate action required: freeze discretionary spending, renegotiate "
                "key vendor contracts, and review headcount efficiency."
            )

        # --- Expense ratio ---
        expense_ratio = float(kpis.get("expense_ratio", 0))
        if expense_ratio > 75:
            insights.append(
                f"COST STRUCTURE RISK: Expense ratio of {expense_ratio:.1f}% "
                "leaves insufficient operating leverage. Identify fixed cost "
                "reduction levers; target a ratio below 70% within two quarters."
            )
        elif expense_ratio > 65:
            insights.append(
                f"EXPENSE MONITORING: Expense ratio of {expense_ratio:.1f}% is "
                "elevated. Automation initiatives (RPA layer) could reduce "
                "back-office costs by an estimated 15-20%, improving this ratio."
            )
        else:
            insights.append(
                f"COST EFFICIENCY: Expense ratio of {expense_ratio:.1f}% reflects "
                "disciplined cost management. Sustain this through continued "
                "process automation and vendor consolidation."
            )

        # --- Cash flow trend ---
        cf_trend = float(kpis.get("cash_flow_trend", 0))
        if cf_trend > 50_000:
            insights.append(
                f"CASH GENERATION: Cash flow is growing at ${cf_trend:,.0f}/month. "
                "The balance sheet is strengthening; explore strategic M&A or "
                "technology investments to accelerate competitive positioning."
            )
        elif cf_trend > 0:
            insights.append(
                f"STABLE CASH FLOW: Modest positive cash flow trend "
                f"(${cf_trend:,.0f}/month). Ensure adequate working capital buffers "
                "(minimum 90-day liquidity) ahead of any capital expenditure cycles."
            )
        else:
            insights.append(
                f"CASH FLOW DETERIORATION: Cash flow declining at "
                f"${abs(cf_trend):,.0f}/month. Review accounts receivable DSO, "
                "inventory turns, and accelerate collections to restore positive FCF."
            )

        # --- Operating leverage ---
        op_lev = kpis.get("operating_leverage", "N/A")
        if op_lev != "N/A":
            op_lev_f = float(op_lev)
            if op_lev_f > 2.0:
                insights.append(
                    f"HIGH OPERATING LEVERAGE: Operating leverage of {op_lev_f:.2f}x "
                    "amplifies both upside and downside. In a growth environment this "
                    "is favourable; ensure stress-testing at -10%, -20% revenue scenarios."
                )
            elif op_lev_f > 1.0:
                insights.append(
                    f"POSITIVE OPERATING LEVERAGE: Leverage of {op_lev_f:.2f}x "
                    "confirms fixed costs are being spread over growing revenue — "
                    "a healthy scaling signal."
                )
            else:
                insights.append(
                    f"NEGATIVE OPERATING LEVERAGE ({op_lev_f:.2f}x): Costs are "
                    "growing faster than revenue. This is unsustainable; review "
                    "headcount and discretionary spend immediately."
                )

        return insights[:6]

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    def generate_risk_alerts(self) -> list[str]:
        """
        Generate risk alerts from credit and market risk indicators.

        Parses both the text summary and structured risk_metrics dict.

        Returns
        -------
        list[str] : risk alert strings
        """
        alerts: list[str] = []
        summary = self.risk_summary
        rm = self.risk_metrics

        # --- Parse AUC from summary text ---
        auc = rm.get("auc_roc", None)
        if auc is None:
            # Try to extract from text
            for line in summary.split("\n"):
                if "AUC-ROC" in line:
                    try:
                        auc = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass

        if auc is not None:
            if auc >= 0.85:
                alerts.append(
                    f"CREDIT MODEL STRONG: AUC-ROC of {auc:.4f} indicates high "
                    "discriminatory power. The model is production-ready for automated "
                    "decisioning. Schedule quarterly recalibration."
                )
            elif auc >= 0.75:
                alerts.append(
                    f"CREDIT MODEL ADEQUATE: AUC-ROC of {auc:.4f} meets minimum "
                    "deployment threshold. Recommend champion/challenger testing "
                    "against a gradient-boosting alternative."
                )
            else:
                alerts.append(
                    f"CREDIT MODEL ALERT: AUC-ROC of {auc:.4f} is below the 0.75 "
                    "threshold. Do not deploy for live decisioning; retrain with "
                    "additional features (bureau tradelines, payment history)."
                )

        # --- High-risk loan concentration ---
        high_pct = rm.get("high_risk_pct", None)
        if high_pct is None:
            for line in summary.split("\n"):
                if "High" in line and "%" in line:
                    try:
                        high_pct = float(line.split("(")[1].split("%")[0])
                    except (IndexError, ValueError):
                        pass

        if high_pct is not None:
            if high_pct > 20:
                alerts.append(
                    f"CREDIT CONCENTRATION RISK: {high_pct:.1f}% of the loan portfolio "
                    "is classified as High-risk (P(default) > 0.60). This exceeds the "
                    "20% regulatory guideline. Review underwriting criteria immediately."
                )
            elif high_pct > 10:
                alerts.append(
                    f"ELEVATED HIGH-RISK LOANS: {high_pct:.1f}% high-risk concentration "
                    "warrants enhanced monitoring. Increase loan-loss provisioning by "
                    "an estimated 15% for this segment."
                )
            else:
                alerts.append(
                    f"PORTFOLIO QUALITY SOUND: Only {high_pct:.1f}% of loans are "
                    "high-risk. Credit policy appears effective; maintain current "
                    "underwriting standards."
                )

        # --- VaR alert ---
        var_dollar = rm.get("var_dollar", None)
        if var_dollar is None:
            for line in summary.split("\n"):
                if "VaR" in line and "$" in line:
                    try:
                        val = line.split("$")[1].split(" ")[0].replace(",", "")
                        var_dollar = float(val)
                    except (IndexError, ValueError):
                        pass

        if var_dollar is not None:
            var_pct_of_port = (var_dollar / 10_000_000) * 100
            if var_pct_of_port > 2.5:
                alerts.append(
                    f"MARKET RISK ELEVATED: 1-day 95% VaR of ${var_dollar:,.0f} "
                    f"({var_pct_of_port:.2f}% of portfolio) is above the 2.5% "
                    "internal limit. Consider reducing equity exposure or adding "
                    "protective put options."
                )
            else:
                alerts.append(
                    f"MARKET RISK CONTAINED: 1-day 95% VaR of ${var_dollar:,.0f} "
                    f"({var_pct_of_port:.2f}% of portfolio) is within risk appetite. "
                    "Continue standard daily monitoring."
                )

        # Volatility alert
        alerts.append(
            "VOLATILITY WATCH: Monitor cross-asset correlations during earnings "
            "season — sector correlations typically spike 20-30%, increasing "
            "effective portfolio volatility and reducing diversification benefit."
        )

        return alerts

    # ------------------------------------------------------------------
    # Strategic recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(self) -> list[str]:
        """
        Synthesise findings from all four FARO engines into strategic recommendations.

        Returns
        -------
        list[str] : strategic recommendation strings
        """
        kpis = self.kpis
        rev_growth = float(kpis.get("revenue_growth_yoy", 8))
        net_margin = float(kpis.get("net_margin", 15))
        expense_ratio = float(kpis.get("expense_ratio", 65))
        cf_trend = float(kpis.get("cash_flow_trend", 0))

        recommendations = []

        # 1. Capital allocation
        if cf_trend > 0 and net_margin > 15:
            recommendations.append(
                "CAPITAL ALLOCATION: Strong cash generation and healthy margins "
                "support a dual strategy of (a) 60% reinvestment in organic growth "
                "(R&D, sales capacity) and (b) 40% return to shareholders via "
                "systematic buybacks."
            )
        else:
            recommendations.append(
                "CAPITAL PRESERVATION: Prioritise maintaining 90+ days of operating "
                "cash on hand. Defer non-critical capex and accelerate collections "
                "to improve working capital cycle by 5-7 days."
            )

        # 2. Risk-adjusted portfolio
        recommendations.append(
            "PORTFOLIO OPTIMISATION: The ESG-constrained portfolio achieves a "
            "comparable Sharpe ratio to the unconstrained optimum while reducing "
            "regulatory and reputational risk. Recommend transitioning to the "
            "ESG-weighted allocation over the next two quarters."
        )

        # 3. Automation ROI
        recommendations.append(
            "PROCESS AUTOMATION: The RPA simulation projects $1M+ annual savings "
            "through automated loan processing, reconciliation, and financial close. "
            "Prioritise loan origination automation (highest volume, fastest ROI) "
            "and target full deployment within 6 months."
        )

        # 4. Credit risk policy
        recommendations.append(
            "CREDIT POLICY: The Random Forest credit model (AUC > 0.85) enables "
            "risk-based pricing. Segment applicants by predicted default probability: "
            "apply a 150 bps premium for Medium-risk loans and cap High-risk "
            "originations at 10% of monthly volume."
        )

        # 5. Forecasting & scenario planning
        recommendations.append(
            "SCENARIO PLANNING: ARIMA revenue forecasts indicate continued growth "
            "but widen confidence intervals at 6+ months. Develop bear/base/bull "
            "scenario models with explicit sensitivity to top-3 revenue drivers. "
            "Reforecast quarterly to reduce planning uncertainty."
        )

        # 6. ESG & governance
        recommendations.append(
            "ESG INTEGRATION: Governance scores drive 40% of the composite ESG weight "
            "in the optimisation model. Strengthen board oversight of climate "
            "transition risk and supply-chain due diligence to improve Environmental "
            "scores, targeting a portfolio-weighted ESG of 72+ by year-end."
        )

        return recommendations
