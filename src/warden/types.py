"""Warden's shared vocabulary — small, frozen, typed.

Everything crossing a layer boundary is one of these. Frozen dataclasses
over dicts: a layer cannot mutate what another layer handed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

Side = Literal["LONG", "SHORT"]
AssetClass = Literal["perp"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Signal:
    """One entry the signal book delivered.

    `entry_price` is the book's OWN recorded basis — followers mirror the
    delivery, never the current mark. `signal_id` must be stable and unique
    per delivery: the agent's exactly-once guarantee is keyed on it.
    """

    signal_id: str
    coin: str
    side: Side
    entry_price: Optional[float] = None
    conviction: Optional[float] = None
    created_at: datetime = field(default_factory=utcnow)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """The brain's verdict on one signal.

    follow_size_usd is NOTIONAL (the position size, not margin) — leverage
    multiplies margin efficiency only, so "$25 follow" is a $25 position at
    any leverage.

    `probabilities` is the brain's parsed action probabilities
    ({"enter", "hold", "exit"} → float in [0, 1]) — filled by NearBrain
    wherever the three values validated, None when they never parsed
    (parse/transport refusals). Presentation data only: no layer reads it
    to decide anything.
    """

    follow: bool
    size_usd: Optional[float] = None
    leverage: float = 1.0
    reason: str = ""
    source: str = "brain"
    probabilities: Optional[dict] = None


@dataclass(frozen=True)
class Policy:
    """The wallet's constitution. Every field is a hard refusal, not a warn."""

    max_trade_usd: float = 25.0
    max_total_deployed_usd: float = 100.0
    max_open_positions: int = 5
    daily_budget_usd: float = 50.0
    max_leverage: float = 2.0
    allowed_coins: frozenset[str] | None = None  # None = any coin


@dataclass(frozen=True)
class PositionState:
    """What the wallet currently holds (read-side snapshot)."""

    open_positions: int = 0
    deployed_usd: float = 0.0
    spent_today_usd: float = 0.0
    held_coins: frozenset[str] = frozenset()
