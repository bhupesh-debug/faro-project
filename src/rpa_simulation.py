"""
rpa_simulation.py
=================
Robotic Process Automation Simulation Layer for FARO.

Simulates three common back-office financial workflows:
  1. Loan processing pipeline
  2. GL vs. bank reconciliation
  3. Month-end financial close

All timestamps are deterministic (seeded) for reproducibility.
"""

from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


class RPASimulation:
    """
    Simulate automated back-office financial workflows.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self._automation_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Loan processing simulation
    # ------------------------------------------------------------------

    _LOAN_STEPS = [
        "receive_application",
        "validate_data",
        "run_credit_check",
        "calculate_risk_score",
        "approve_reject_decision",
        "generate_letter",
    ]

    _STEP_BASE_MS = {
        "receive_application":    50,
        "validate_data":         120,
        "run_credit_check":      800,
        "calculate_risk_score":  350,
        "approve_reject_decision": 200,
        "generate_letter":       180,
    }

    def simulate_loan_processing(self, n_loans: int = 100) -> pd.DataFrame:
        """
        Simulate end-to-end automated loan approval workflow.

        Each loan goes through 6 steps. Data validation failures (~5 %)
        cause early exit with FAILED status. Credit checks drive ~30 %
        rejection rate.

        Returns
        -------
        pd.DataFrame : loan_id, step, status, timestamp, processing_time_ms
        """
        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, Any]] = []
        base_ts = datetime(2026, 3, 1, 8, 0, 0)

        approval_count = 0
        rejection_count = 0
        failure_count = 0

        for i in range(n_loans):
            loan_id = f"LOAN-{i + 1:04d}"
            credit_score = int(rng.integers(550, 850))
            ts = base_ts + timedelta(minutes=i * 3)

            # Determine loan fate upfront
            validation_fail = rng.random() < 0.05
            # Approval probability rises with credit score
            approve_prob = (credit_score - 550) / 300
            approved = rng.random() < approve_prob

            for step in self._LOAN_STEPS:
                base_ms = self._STEP_BASE_MS[step]
                proc_ms = int(rng.normal(base_ms, base_ms * 0.15))
                proc_ms = max(proc_ms, 10)

                # Determine step outcome
                if validation_fail and step == "validate_data":
                    status = "FAILED"
                elif step == "approve_reject_decision":
                    status = "APPROVED" if approved else "REJECTED"
                elif step == "generate_letter" and not approved:
                    status = "SKIPPED"
                else:
                    status = "COMPLETED"

                rows.append({
                    "loan_id": loan_id,
                    "credit_score": credit_score,
                    "step": step,
                    "status": status,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time_ms": proc_ms,
                })

                ts += timedelta(milliseconds=int(proc_ms + rng.integers(5, 50)))

                # Stop pipeline on validation failure
                if status == "FAILED":
                    failure_count += 1
                    break

            if not validation_fail:
                if approved:
                    approval_count += 1
                else:
                    rejection_count += 1

        df = pd.DataFrame(rows)
        summary = {
            "approved": approval_count,
            "rejected": rejection_count,
            "failed_validation": failure_count,
            "total": n_loans,
        }
        self._automation_log.append({"task": "loan_processing", "summary": summary})
        print(
            f"[rpa] Loan processing: {approval_count} approved, "
            f"{rejection_count} rejected, {failure_count} validation failures"
        )
        return df

    # ------------------------------------------------------------------
    # 2. Reconciliation simulation
    # ------------------------------------------------------------------

    def simulate_reconciliation(
        self, n_transactions: int = 500
    ) -> dict[str, Any]:
        """
        Simulate GL vs. bank-statement reconciliation.

        Creates matching GL entries and bank statement lines, then
        introduces realistic exceptions:
          - Missing in bank (GL only): ~3 %
          - Missing in GL (bank only): ~2 %
          - Amount mismatch (data entry error): ~4 %
          - Duplicate bank entry: ~1 %

        Returns
        -------
        dict with keys: summary (DataFrame), exceptions (DataFrame),
                        reconciliation_status, match_rate
        """
        rng = np.random.default_rng(self.seed + 1)
        base_date = datetime(2026, 3, 1)

        # Generate GL entries
        gl_rows = []
        for i in range(n_transactions):
            txn_date = base_date + timedelta(days=int(rng.integers(0, 28)))
            amount = round(float(rng.lognormal(7, 1.2)), 2)   # ~$1k mean
            gl_rows.append({
                "txn_id": f"GL-{i + 1:05d}",
                "date": txn_date.strftime("%Y-%m-%d"),
                "description": rng.choice([
                    "Vendor Payment", "Payroll", "Utilities",
                    "Rent", "Software License", "Travel",
                ]),
                "amount": amount,
                "account_code": rng.choice(["5000", "5100", "5200", "5300", "5400"]),
                "source": "GL",
            })
        gl_df = pd.DataFrame(gl_rows)

        # Create bank entries with controlled exceptions
        bank_rows = []
        exceptions = []
        missing_in_bank = set()
        missing_in_gl_ids = set()

        for i, gl_row in enumerate(gl_rows):
            r = rng.random()

            if r < 0.03:
                # Missing in bank
                missing_in_bank.add(gl_row["txn_id"])
                exceptions.append({
                    "txn_id": gl_row["txn_id"],
                    "exception_type": "MISSING_IN_BANK",
                    "gl_amount": gl_row["amount"],
                    "bank_amount": None,
                    "difference": gl_row["amount"],
                    "date": gl_row["date"],
                })
            elif r < 0.07:
                # Amount mismatch (data-entry error on bank side ± 1-5 %)
                error_pct = rng.uniform(-0.05, 0.05)
                bank_amount = round(gl_row["amount"] * (1 + error_pct), 2)
                diff = round(bank_amount - gl_row["amount"], 2)
                bank_rows.append({
                    "bank_txn_id": f"BNK-{i + 1:05d}",
                    "date": gl_row["date"],
                    "description": gl_row["description"],
                    "amount": bank_amount,
                    "matched_gl_id": gl_row["txn_id"],
                })
                exceptions.append({
                    "txn_id": gl_row["txn_id"],
                    "exception_type": "AMOUNT_MISMATCH",
                    "gl_amount": gl_row["amount"],
                    "bank_amount": bank_amount,
                    "difference": diff,
                    "date": gl_row["date"],
                })
            else:
                # Clean match
                bank_rows.append({
                    "bank_txn_id": f"BNK-{i + 1:05d}",
                    "date": gl_row["date"],
                    "description": gl_row["description"],
                    "amount": gl_row["amount"],
                    "matched_gl_id": gl_row["txn_id"],
                })

        # Add extra bank-only transactions (~2 % of n)
        n_bank_only = max(int(n_transactions * 0.02), 1)
        for j in range(n_bank_only):
            eid = f"BNK-EXTRA-{j + 1:03d}"
            amount = round(float(rng.lognormal(6, 1)), 2)
            txn_date = (base_date + timedelta(days=int(rng.integers(0, 28)))).strftime(
                "%Y-%m-%d"
            )
            bank_rows.append({
                "bank_txn_id": eid,
                "date": txn_date,
                "description": "Unidentified Debit",
                "amount": amount,
                "matched_gl_id": None,
            })
            missing_in_gl_ids.add(eid)
            exceptions.append({
                "txn_id": eid,
                "exception_type": "MISSING_IN_GL",
                "gl_amount": None,
                "bank_amount": amount,
                "difference": -amount,
                "date": txn_date,
            })

        bank_df = pd.DataFrame(bank_rows)
        exc_df = pd.DataFrame(exceptions)

        total_txns = n_transactions + n_bank_only
        matched = total_txns - len(exceptions)
        match_rate = matched / total_txns

        summary_rows = [
            {"category": "Total GL Entries",        "count": n_transactions},
            {"category": "Total Bank Entries",      "count": len(bank_rows)},
            {"category": "Matched",                 "count": matched},
            {"category": "Unmatched / Exceptions",  "count": len(exceptions)},
            {"category": "Missing in Bank",         "count": len(missing_in_bank)},
            {"category": "Missing in GL",           "count": len(missing_in_gl_ids)},
            {"category": "Amount Mismatches",
             "count": len(exc_df[exc_df["exception_type"] == "AMOUNT_MISMATCH"])},
        ]
        summary_df = pd.DataFrame(summary_rows)

        self._automation_log.append({
            "task": "reconciliation",
            "summary": {"match_rate": match_rate, "exceptions": len(exceptions)},
        })
        print(
            f"[rpa] Reconciliation: {matched}/{total_txns} matched "
            f"({match_rate:.1%}), {len(exceptions)} exceptions"
        )

        return {
            "summary": summary_df,
            "exceptions": exc_df,
            "gl_data": gl_df,
            "bank_data": bank_df,
            "reconciliation_status": "COMPLETED",
            "match_rate": round(match_rate, 4),
        }

    # ------------------------------------------------------------------
    # 3. Month-end close simulation
    # ------------------------------------------------------------------

    _CLOSE_TASKS = [
        ("journal_entry_validation",  "Accounting",  4.5),
        ("subledger_reconciliation",  "Accounting",  3.0),
        ("intercompany_elimination",  "Consolidation", 2.0),
        ("fixed_asset_depreciation",  "Fixed Assets", 1.5),
        ("prepaid_amortization",      "Accounting",   1.0),
        ("accruals_posting",          "Accounting",   2.5),
        ("trial_balance_review",      "Controller",   2.0),
        ("variance_analysis",         "FP&A",         3.0),
        ("flux_commentary",           "FP&A",         2.5),
        ("management_reporting",      "FP&A",         4.0),
        ("tax_provision",             "Tax",          3.5),
        ("external_reporting",        "Reporting",    2.0),
        ("close_sign_off",            "CFO",          1.0),
    ]

    def simulate_financial_close(
        self, period: str = "2026-03"
    ) -> pd.DataFrame:
        """
        Simulate month-end close checklist with automated vs. manual tasks.

        Returns
        -------
        pd.DataFrame : task, owner, estimated_hours, automated, status,
                       start_time, end_time, notes
        """
        rng = np.random.default_rng(self.seed + 2)

        # Automation map: some tasks are fully automated
        auto_tasks = {
            "journal_entry_validation", "subledger_reconciliation",
            "fixed_asset_depreciation", "prepaid_amortization",
            "trial_balance_review",
        }

        rows = []
        ts = datetime.strptime(period + "-01", "%Y-%m-%d")
        ts += timedelta(days=30)  # Close starts after month ends

        for task_name, owner, est_hours in self._CLOSE_TASKS:
            automated = task_name in auto_tasks
            # Automated tasks complete much faster
            if automated:
                actual_hours = est_hours * rng.uniform(0.15, 0.30)
                status = "COMPLETED_AUTO"
            else:
                actual_hours = est_hours * rng.uniform(0.80, 1.20)
                status = "COMPLETED_MANUAL"

            end_ts = ts + timedelta(hours=float(actual_hours))

            notes_map = {
                "journal_entry_validation":  "1,847 entries validated; 12 exceptions auto-corrected",
                "subledger_reconciliation":  "AP, AR, payroll subs reconciled",
                "intercompany_elimination":  "3 entities; $2.1M eliminated",
                "fixed_asset_depreciation":  "Auto-calculated per depreciation schedule",
                "prepaid_amortization":      "47 prepaid schedules processed",
                "accruals_posting":          "Accruals posted per FP&A estimates",
                "trial_balance_review":      "TB balanced; no out-of-balance",
                "variance_analysis":         "Revenue +4.2% vs budget; flagged 2 cost centres",
                "flux_commentary":           "CFO deck commentary drafted",
                "management_reporting":      "Board package generated",
                "tax_provision":             "Effective rate 22.4%",
                "external_reporting":        "Press release financials verified",
                "close_sign_off":            "CFO approval obtained",
            }

            rows.append({
                "task": task_name,
                "owner": owner,
                "estimated_hours": round(est_hours, 1),
                "actual_hours": round(float(actual_hours), 2),
                "automated": automated,
                "status": status,
                "start_time": ts.strftime("%Y-%m-%d %H:%M"),
                "end_time": end_ts.strftime("%Y-%m-%d %H:%M"),
                "notes": notes_map.get(task_name, ""),
            })
            ts = end_ts + timedelta(minutes=int(rng.integers(15, 60)))

        df = pd.DataFrame(rows)
        total_auto = df.loc[df["automated"], "estimated_hours"].sum()
        total_hours = df["estimated_hours"].sum()
        print(
            f"[rpa] Financial close ({period}): "
            f"{len(df)} tasks | "
            f"{total_auto:.1f}/{total_hours:.1f} hours automated"
        )
        self._automation_log.append({
            "task": "financial_close",
            "period": period,
            "auto_hours": total_auto,
            "total_hours": total_hours,
        })
        return df

    # ------------------------------------------------------------------
    # 4. Automation metrics
    # ------------------------------------------------------------------

    def calculate_automation_metrics(self) -> dict[str, Any]:
        """
        Summarise automation ROI across all simulated workflows.

        Returns
        -------
        dict : tasks_automated, time_saved_hours, error_reduction_pct,
               cost_savings (annual $), automation_rate_pct
        """
        # Loan processing metrics
        loan_throughput = 100          # loans / run
        manual_time_per_loan = 45      # minutes
        auto_time_per_loan = 3         # minutes
        loan_time_saved = (manual_time_per_loan - auto_time_per_loan) * loan_throughput / 60
        # Monthly loan volume assumption: 2000 loans/month
        monthly_loans = 2000
        monthly_loan_savings_hrs = (manual_time_per_loan - auto_time_per_loan) * monthly_loans / 60

        # Reconciliation metrics
        recon_manual_hours = 8         # per month
        recon_auto_hours = 0.5
        recon_savings = (recon_manual_hours - recon_auto_hours) * 12  # annual

        # Close metrics
        close_manual_hours = 32        # per month
        close_auto_hours = 32 * 0.30   # 30 % reduction
        close_savings_annual = (close_manual_hours - close_auto_hours) * 12

        total_time_saved_annual = (
            monthly_loan_savings_hrs * 12
            + recon_savings
            + close_savings_annual
        )

        # Cost at $85/hour blended rate
        hourly_rate = 85
        cost_savings = total_time_saved_annual * hourly_rate

        tasks_automated = 5 + 1 + 5    # loan steps + recon + close auto tasks
        error_rate_manual = 0.08       # 8 % error rate manual
        error_rate_auto = 0.005        # 0.5 % error rate automated
        error_reduction = (error_rate_manual - error_rate_auto) / error_rate_manual * 100

        return {
            "tasks_automated": tasks_automated,
            "time_saved_hours_annual": round(total_time_saved_annual, 1),
            "error_reduction_pct": round(error_reduction, 1),
            "cost_savings_annual_usd": round(cost_savings, 2),
            "hourly_rate_assumed": hourly_rate,
            "automation_rate_pct": round(tasks_automated / (tasks_automated + 8) * 100, 1),
        }
