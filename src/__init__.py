# FARO — Financial Analytics, Risk & Optimization
# src package init

from .data_processing import (
    generate_financial_data,
    generate_loan_data,
    generate_market_data,
    generate_esg_scores,
)
from .fpa_engine import FPAEngine
from .risk_engine import RiskEngine
from .portfolio_engine import PortfolioEngine
from .rpa_simulation import RPASimulation
from .ai_insights import AIInsightsEngine

__all__ = [
    "generate_financial_data",
    "generate_loan_data",
    "generate_market_data",
    "generate_esg_scores",
    "FPAEngine",
    "RiskEngine",
    "PortfolioEngine",
    "RPASimulation",
    "AIInsightsEngine",
]
