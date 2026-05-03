"""Centralized factory wiring — single place to assemble services with their deps.

Honors Dependency Inversion: callers receive ready services, not concrete classes.
"""
from __future__ import annotations
from pathlib import Path

from config import ROOT
from portfolio.repository import JsonPortfolioRepository, PortfolioRepository
from portfolio.lifecycle import PositionFactory, ExitEvaluator
from portfolio.service import PortfolioService

from risk.position_sizer import VolatilityAdjustedSizer, SizerConfig
from risk.regime import MultiMarketRegimeDetector
from risk.sector_rotation import CompositeSectorRanker
from risk.correlation import CorrelationAnalyzer
from risk.red_flags import RedFlagScanner
from risk.tax_optimizer import IndianTaxOptimizer
from risk.gate import RiskGate
from risk.checks import (
    ConcentrationCheck, SectorCheck, MarketCheck, DrawdownCheck,
    RegimeCheck, SectorMomentumCheck, CorrelationClusterCheck,
)


PORTFOLIO_FILE = ROOT / "portfolio.json"


def build_portfolio_service(repo_path: Path | None = None) -> PortfolioService:
    repo: PortfolioRepository = JsonPortfolioRepository(repo_path or PORTFOLIO_FILE)
    return PortfolioService(repo, PositionFactory(), ExitEvaluator())


def build_sizer() -> VolatilityAdjustedSizer:
    return VolatilityAdjustedSizer(SizerConfig())


def build_regime_detector() -> MultiMarketRegimeDetector:
    return MultiMarketRegimeDetector(("IN", "US"))


def build_sector_ranker() -> CompositeSectorRanker:
    return CompositeSectorRanker()


def build_correlation_analyzer() -> CorrelationAnalyzer:
    return CorrelationAnalyzer()


def build_red_flag_scanner() -> RedFlagScanner:
    return RedFlagScanner()


def build_tax_optimizer(ltcg_used: float = 0.0) -> IndianTaxOptimizer:
    return IndianTaxOptimizer(ltcg_used_this_year=ltcg_used)


def build_default_risk_gate() -> RiskGate:
    """Phase A + B checks composed in priority order."""
    return RiskGate([
        DrawdownCheck(),               # gates everything in deep DD
        RegimeCheck(),                 # scales by market regime
        ConcentrationCheck(),          # single-position cap
        SectorCheck(),                 # sector cap
        MarketCheck(),                 # India/US split
        SectorMomentumCheck(),         # Phase B
        CorrelationClusterCheck(),     # Phase B
    ])
