"""
risk_engine.py
==============
Risk Analytics Engine for FARO.

Covers:
  - Credit risk modelling (RandomForest classifier on loan data)
  - Market risk: VaR via historical simulation, annualised volatility
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class RiskEngine:
    """
    Credit + Market Risk Engine.

    Parameters
    ----------
    loan_data   : pd.DataFrame  — output of generate_loan_data()
    market_data : pd.DataFrame  — output of generate_market_data()
    """

    def __init__(self, loan_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
        self.loan_df = loan_data.copy()
        self.market_df = market_data.copy()

        # Isolate return columns (drop date)
        if "date" in self.market_df.columns:
            self.returns = self.market_df.drop(columns=["date"]).values
            self.tickers = [c for c in self.market_df.columns if c != "date"]
        else:
            self.returns = self.market_df.values
            self.tickers = list(self.market_df.columns)

        self._model: RandomForestClassifier | None = None
        self._feature_cols: list[str] = []
        self._model_metrics: dict[str, float] = {}
        self._predicted_probs: np.ndarray | None = None
        self._risk_summary_text: str = ""

    # ------------------------------------------------------------------
    # Credit model
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering shared between training and prediction."""
        df = df.copy()

        # Credit score bins → ordinal risk_score (1=best, 5=worst)
        bins = [0, 580, 620, 680, 740, 900]
        labels = [5, 4, 3, 2, 1]
        df["risk_score"] = pd.cut(
            df["credit_score"], bins=bins, labels=labels
        ).astype(int)

        # Income-to-loan ratio (higher = safer)
        df["income_to_loan"] = df["annual_income"] / (df["loan_amount"] + 1)

        # Log-transform income to reduce skew
        df["log_income"] = np.log1p(df["annual_income"])

        # Log-transform loan amount
        df["log_loan"] = np.log1p(df["loan_amount"])

        # Encode categoricals
        le_purpose = LabelEncoder()
        le_home = LabelEncoder()
        df["purpose_enc"] = le_purpose.fit_transform(df["loan_purpose"])
        df["home_enc"] = le_home.fit_transform(df["home_ownership"])

        return df

    def train_credit_model(self) -> dict[str, float]:
        """
        Train a RandomForestClassifier on loan data.

        Feature engineering:
          - credit_score bins → risk_score
          - income_to_loan ratio
          - log-transformed income & loan amount
          - encoded categoricals

        Returns
        -------
        dict : accuracy, precision, recall, f1, auc_roc
        """
        df = self._engineer_features(self.loan_df)

        feature_cols = [
            "loan_amount", "interest_rate", "credit_score",
            "annual_income", "dti_ratio", "employment_length",
            "risk_score", "income_to_loan", "log_income", "log_loan",
            "purpose_enc", "home_enc",
        ]
        self._feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["default"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        self._model = clf

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        }
        self._model_metrics = metrics

        # Store full-dataset predicted probabilities for distribution analysis
        df_eng = self._engineer_features(self.loan_df)
        X_all = df_eng[feature_cols].values
        self._predicted_probs = clf.predict_proba(X_all)[:, 1]

        return metrics

    def predict_default(self, loan_features: dict[str, Any]) -> float:
        """
        Predict default probability for a single loan application.

        Parameters
        ----------
        loan_features : dict
            Keys: loan_amount, interest_rate, credit_score, annual_income,
                  dti_ratio, employment_length, loan_purpose, home_ownership

        Returns
        -------
        float : probability of default in [0, 1]
        """
        if self._model is None:
            raise RuntimeError("Call train_credit_model() first.")

        row = pd.DataFrame([loan_features])
        row = self._engineer_features(row)

        # Encode categoricals relative to known corpus
        df_ref = self._engineer_features(self.loan_df)
        for col_src, col_enc in [("loan_purpose", "purpose_enc"),
                                  ("home_ownership", "home_enc")]:
            mapping = (
                df_ref.groupby(col_src)[col_enc].first().to_dict()
            )
            row[col_enc] = row[col_src].map(mapping).fillna(0).astype(int)

        X = row[self._feature_cols].values
        prob = float(self._model.predict_proba(X)[0, 1])
        return round(prob, 4)

    def risk_distribution(self) -> pd.DataFrame:
        """
        Categorise portfolio loans into Low / Medium / High risk.

        Thresholds: < 0.30 → Low | 0.30–0.60 → Medium | > 0.60 → High

        Returns
        -------
        pd.DataFrame : risk_category, count, pct
        """
        if self._predicted_probs is None:
            self.train_credit_model()

        probs = self._predicted_probs
        cats = np.where(probs < 0.30, "Low",
               np.where(probs <= 0.60, "Medium", "High"))

        total = len(cats)
        rows = []
        for cat in ["Low", "Medium", "High"]:
            cnt = int((cats == cat).sum())
            rows.append({
                "risk_category": cat,
                "count": cnt,
                "pct": round(cnt / total * 100, 2),
                "avg_default_prob": round(probs[cats == cat].mean(), 4),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Market risk
    # ------------------------------------------------------------------

    def calculate_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        portfolio_value: float = 10_000_000,
    ) -> dict[str, float]:
        """
        Calculate portfolio Value at Risk using historical simulation.

        Assumes equal-weight portfolio across all tickers.

        Parameters
        ----------
        confidence      : VaR confidence level (default 0.95)
        horizon         : holding period in days (default 1)
        portfolio_value : notional portfolio size ($, default $10 M)

        Returns
        -------
        dict : var_pct, var_dollar, cvar_pct, cvar_dollar
        """
        # Equal-weight portfolio daily P&L
        weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]
        port_returns = self.returns @ weights  # shape (T,)

        # Scale to horizon
        if horizon > 1:
            port_returns = port_returns * np.sqrt(horizon)

        # Historical VaR
        var_pct = float(np.percentile(port_returns, (1 - confidence) * 100))
        var_dollar = abs(var_pct) * portfolio_value

        # CVaR (Expected Shortfall)
        tail = port_returns[port_returns <= var_pct]
        cvar_pct = float(tail.mean()) if len(tail) > 0 else var_pct
        cvar_dollar = abs(cvar_pct) * portfolio_value

        return {
            "confidence": confidence,
            "horizon_days": horizon,
            "var_pct": round(var_pct * 100, 4),
            "var_dollar": round(var_dollar, 2),
            "cvar_pct": round(cvar_pct * 100, 4),
            "cvar_dollar": round(cvar_dollar, 2),
            "portfolio_value": portfolio_value,
        }

    def calculate_volatility(self) -> pd.DataFrame:
        """
        Annualised volatility for each ticker (252 trading days / year).

        Returns
        -------
        pd.DataFrame : ticker, daily_vol_pct, annual_vol_pct
        """
        rows = []
        for i, ticker in enumerate(self.tickers):
            daily_vol = float(np.std(self.returns[:, i], ddof=1))
            annual_vol = daily_vol * np.sqrt(252)
            rows.append({
                "ticker": ticker,
                "daily_vol_pct": round(daily_vol * 100, 4),
                "annual_vol_pct": round(annual_vol * 100, 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_risk_summary(self) -> str:
        """Return a formatted text summary of risk metrics."""
        if not self._model_metrics:
            self.train_credit_model()

        var_info = self.calculate_var()
        vol_df = self.calculate_volatility()
        dist_df = self.risk_distribution()
        m = self._model_metrics

        high_pct = dist_df.loc[dist_df["risk_category"] == "High", "pct"].values[0]
        avg_annual_vol = vol_df["annual_vol_pct"].mean()

        lines = [
            "=" * 60,
            "  RISK ENGINE SUMMARY",
            "=" * 60,
            "  [Credit Risk Model]",
            f"    Accuracy   : {m['accuracy']:.2%}",
            f"    Precision  : {m['precision']:.2%}",
            f"    Recall     : {m['recall']:.2%}",
            f"    F1 Score   : {m['f1']:.2%}",
            f"    AUC-ROC    : {m['auc_roc']:.4f}",
            "",
            "  [Loan Risk Distribution]",
        ]
        for _, row in dist_df.iterrows():
            lines.append(
                f"    {row['risk_category']:6s} : {row['count']:5d} loans "
                f"({row['pct']:.1f}%)  avg_prob={row['avg_default_prob']:.3f}"
            )
        lines += [
            "",
            "  [Market Risk — $10M Equal-Weight Portfolio]",
            f"    95% 1-Day VaR  : ${var_info['var_dollar']:>12,.0f}"
            f"  ({var_info['var_pct']:.3f}%)",
            f"    95% CVaR       : ${var_info['cvar_dollar']:>12,.0f}"
            f"  ({var_info['cvar_pct']:.3f}%)",
            f"    Avg Annual Vol : {avg_annual_vol:.2f}%",
            "=" * 60,
        ]
        text = "\n".join(lines)
        self._risk_summary_text = text
        return text
