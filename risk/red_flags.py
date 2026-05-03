"""Red flag scanner — composable forensic/quality rules.

Each rule is a small class implementing the RedFlagRule protocol.
Add new rules without touching existing ones (Open/Closed).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .interfaces import RedFlag


# ── Rule base + helpers ──────────────────────────────────────────────────────
def _flag(symbol: str, code: str, severity: str, message: str) -> RedFlag:
    return RedFlag(symbol=symbol, code=code, severity=severity, message=message)


# ── Concrete rules ───────────────────────────────────────────────────────────
@dataclass
class HighDebtRule:
    code: str = "HIGH_DEBT"
    threshold: float = 200.0          # D/E percent

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        de = info.get("debtToEquity")
        if de is None:
            return None
        if de >= self.threshold:
            return _flag(symbol, self.code, "WARN", f"D/E {de:.0f} ≥ {self.threshold}")
        return None


@dataclass
class NegativeFCFRule:
    code: str = "NEGATIVE_FCF"

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        fcf = info.get("freeCashflow")
        if fcf is None:
            return None
        if fcf < 0:
            return _flag(symbol, self.code, "WARN", f"Free cash flow negative: {fcf:,.0f}")
        return None


@dataclass
class LowCashConversionRule:
    """Operating CF / Net Income — proxy for earnings quality."""
    code: str = "LOW_CASH_CONVERSION"
    threshold: float = 0.6

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        ocf = info.get("operatingCashflow")
        ni = info.get("netIncomeToCommon") or info.get("netIncome")
        if not ocf or not ni or ni <= 0:
            return None
        ratio = ocf / ni
        if ratio < self.threshold:
            return _flag(symbol, self.code, "WARN",
                         f"OCF/NI {ratio:.2f} < {self.threshold} (earnings quality risk)")
        return None


@dataclass
class HighGoodwillRule:
    code: str = "HIGH_GOODWILL"
    threshold: float = 0.30

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        # yfinance .info doesn't always expose goodwill — return None silently
        gw = info.get("goodWill") or info.get("goodwill")
        ta = info.get("totalAssets")
        if not gw or not ta or ta <= 0:
            return None
        ratio = gw / ta
        if ratio >= self.threshold:
            return _flag(symbol, self.code, "WARN",
                         f"Goodwill {ratio*100:.0f}% of assets — acquisition-heavy")
        return None


@dataclass
class PriceCollapseRule:
    """Price down sharply with no recovery — possible falling knife."""
    code: str = "PRICE_COLLAPSE"
    drawdown_threshold: float = -0.30

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        if history is None or len(history) < 60:
            return None
        try:
            close = history["Close"]
            high = float(close.max())
            current = float(close.iloc[-1])
            dd = current / high - 1
            if dd <= self.drawdown_threshold:
                # check no recovery — current is near 60-day min
                last_60 = close.tail(60)
                if current <= last_60.min() * 1.05:
                    return _flag(symbol, self.code, "WARN",
                                 f"Down {dd*100:.0f}% from highs, no recovery")
        except Exception:
            return None
        return None


@dataclass
class VolumeDropRule:
    """Sustained volume collapse — institutional exit signal."""
    code: str = "VOLUME_COLLAPSE"
    ratio_threshold: float = 0.50

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        if history is None or len(history) < 60:
            return None
        try:
            v = history["Volume"]
            recent = v.tail(20).mean()
            prior = v.iloc[-60:-20].mean()
            if prior > 0 and recent / prior <= self.ratio_threshold:
                return _flag(symbol, self.code, "INFO",
                             f"Volume {recent/prior:.0%} of 60-day average")
        except Exception:
            return None
        return None


@dataclass
class HighPERule:
    code: str = "EXTREME_VALUATION"
    threshold: float = 100.0

    def check(self, symbol: str, info: dict, history) -> Optional[RedFlag]:
        pe = info.get("trailingPE")
        if pe is None:
            return None
        if pe > self.threshold:
            return _flag(symbol, self.code, "INFO", f"P/E {pe:.0f} > {self.threshold}")
        return None


# ── Scanner ──────────────────────────────────────────────────────────────────
class RedFlagScanner:
    """Runs all rules, aggregates flags. Stateless."""

    DEFAULT_RULES = [
        HighDebtRule(),
        NegativeFCFRule(),
        LowCashConversionRule(),
        HighGoodwillRule(),
        PriceCollapseRule(),
        VolumeDropRule(),
        HighPERule(),
    ]

    def __init__(self, rules: list | None = None) -> None:
        self.rules = rules if rules is not None else list(self.DEFAULT_RULES)

    def scan(self, symbol: str, info: dict, history: pd.DataFrame | None) -> list[RedFlag]:
        flags: list[RedFlag] = []
        for rule in self.rules:
            try:
                f = rule.check(symbol, info or {}, history)
                if f is not None:
                    flags.append(f)
            except Exception:
                continue
        return flags

    def critical_count(self, flags: list[RedFlag]) -> int:
        """Number of WARN+CRITICAL flags (used for thesis-break override)."""
        return sum(1 for f in flags if f.severity in ("WARN", "CRITICAL"))
