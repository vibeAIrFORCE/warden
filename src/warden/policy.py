"""The policy wallet — the layer that refuses.

Every order passes `PolicyWallet.admit()` BEFORE the executor ever sees it.
A refusal here is terminal for that order: no retry-crafting, no prompt
negotiation, no "just this once". This is Warden's thesis: spending limits
belong in code between the brain and the exchange, checked at decision
time — the same place NEAR chain-signature policy keys enforce them at
MPC time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import Decision, Policy, PositionState, Signal


@dataclass(frozen=True)
class AdmitVerdict:
    allowed: bool
    reason: str
    effective_size_usd: Optional[float] = None
    effective_leverage: float = 1.0


class PolicyWallet:
    """Hard limits, checked in a fixed order, every one a full stop.

    The admit order is the audit order: identity → coin → size → leverage →
    concurrency → deployed capital → daily fuse. First refusal wins.
    """

    def __init__(self, policy: Policy) -> None:
        self._p = policy

    async def admit(
        self, signal: Signal, decision: Decision, state: PositionState
    ) -> AdmitVerdict:
        p = self._p

        if not decision.follow:
            return AdmitVerdict(False, "brain said no")

        size = decision.size_usd
        if size is None or size <= 0:
            return AdmitVerdict(False, "brain proposed no size")

        if signal.coin in state.held_coins:
            return AdmitVerdict(
                False, f"already holding {signal.coin} (one-way mode)"
            )

        if p.allowed_coins is not None and signal.coin not in p.allowed_coins:
            return AdmitVerdict(False, f"{signal.coin} not on the coin allowlist")

        if size > p.max_trade_usd:
            return AdmitVerdict(
                False,
                f"size ${size:.2f} > max_trade_usd ${p.max_trade_usd:.2f}",
            )

        leverage = max(1.0, float(decision.leverage or 1.0))
        if leverage > p.max_leverage:
            leverage = p.max_leverage  # clamp, never refuse — the market's
            # own max clamps again at the executor

        if state.open_positions >= p.max_open_positions:
            return AdmitVerdict(
                False,
                f"open positions {state.open_positions} >= cap "
                f"{p.max_open_positions}",
            )

        if state.deployed_usd + size > p.max_total_deployed_usd:
            return AdmitVerdict(
                False,
                f"deployed ${state.deployed_usd:.2f} + ${size:.2f} > cap "
                f"${p.max_total_deployed_usd:.2f}",
            )

        if state.spent_today_usd + size > p.daily_budget_usd:
            return AdmitVerdict(
                False,
                f"daily fuse: ${state.spent_today_usd:.2f} spent + ${size:.2f} "
                f"> ${p.daily_budget_usd:.2f}/day",
            )

        return AdmitVerdict(
            True, "admitted", effective_size_usd=size, effective_leverage=leverage
        )
