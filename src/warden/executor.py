"""The Hyperliquid executor — typed actions or nothing.

Warden's teeth, ported from the pattern we run in production:

  * actions are CONSTRUCTED in code (three frozen types below) and the
    signing dispatch accepts ONLY those types — a raw dict can never reach
    the signer, so withdrawals / USD-sends / staking / builder-fee
    approvals are structurally unsendable. Not a permission check: a type
    check.
  * the executor is INERT until an explicit two-key arm (a flag + a
    recorded testnet fire-test), and every cycle re-checks both.
  * every order rides the policy verdict the caller already obtained —
    the executor re-checks nothing; it refuses to be the policy layer.

Signing uses hyperliquid-python-sdk when installed (extras: `hl`); without
it the executor builds actions and raises on post — dry mode works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .types import Side

ORDER_ACTION_TYPES = ("OrderAction", "CancelAction", "UpdateLeverageAction")


class ActionNotAllowed(TypeError):
    """Raised BEFORE any signing for anything off the allowlist."""


class NotArmed(RuntimeError):
    """Raised when either arming key is absent — the executor is inert."""


@dataclass(frozen=True)
class OrderAction:
    coin: str
    asset: int
    is_buy: bool
    size: float
    limit_px: float
    order_type: str = "ioc"          # "ioc" | "gtc"
    reduce_only: bool = False
    leverage: float = 1.0            # isolated margin, clamped by the policy

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "order",
            "coin": self.coin,
            "asset": self.asset,
            "is_buy": self.is_buy,
            "size": self.size,
            "limit_px": self.limit_px,
            "order_type": self.order_type,
            "reduce_only": self.reduce_only,
            "leverage": self.leverage,
        }


@dataclass(frozen=True)
class CancelAction:
    coin: str
    asset: int
    oid: int

    def to_wire(self) -> dict[str, Any]:
        return {"type": "cancel", "coin": self.coin, "asset": self.asset,
                "oid": self.oid}


@dataclass(frozen=True)
class UpdateLeverageAction:
    coin: str
    asset: int
    leverage: float
    is_cross: bool = False           # isolated only — never cross margin

    def to_wire(self) -> dict[str, Any]:
        return {"type": "updateLeverage", "coin": self.coin,
                "asset": self.asset, "leverage": self.leverage,
                "is_cross": self.is_cross}


ALLOWED = (OrderAction, CancelAction, UpdateLeverageAction)

ENV_FLAG = "WARDEN_ENABLED"
ENV_FIRE_TEST = "WARDEN_TESTNET_FIRE_TEST"
BASE_URLS = {"prod": "https://api.hyperliquid.xyz",
             "testnet": "https://api.hyperliquid-testnet.xyz"}


class HyperliquidExecutor:
    def __init__(self, env: str = "testnet", private_key_env: str = "WARDEN_PRIVATE_KEY") -> None:
        if env not in BASE_URLS:
            raise ValueError(f"unknown env {env!r} — use prod|testnet")
        self.env = env
        self.base_url = BASE_URLS[env]
        self._key_env = private_key_env

    @classmethod
    def from_env(cls) -> "HyperliquidExecutor":
        return cls(env=os.getenv("WARDEN_ENV", "testnet"))

    def assert_armed(self) -> None:
        flag = os.getenv(ENV_FLAG, "").strip().lower()
        fire = os.getenv(ENV_FIRE_TEST, "").strip().lower()
        if flag not in ("1", "true", "on"):
            raise NotArmed(f"{ENV_FLAG} not set — the executor is inert")
        if fire != "pass":
            raise NotArmed(
                f"{ENV_FIRE_TEST} != 'pass' — run the testnet fire-test first"
            )

    def build_open(self, coin: str, asset: int, side: Side, size: float,
                   mark: float, leverage: float) -> OrderAction:
        """IOC market-shaped order at an aggressive limit (mark ±1%) — the
        only way an open exists. Sizing is NOTIONAL: qty = size / mark."""
        if size <= 0 or mark <= 0:
            raise ValueError("size and mark must be positive")
        aggressive = mark * (1.01 if side == "LONG" else 0.99)
        return OrderAction(
            coin=coin, asset=asset, is_buy=(side == "LONG"),
            size=round(size / mark, 6), limit_px=round(aggressive, 6),
            order_type="ioc", leverage=max(1.0, leverage),
        )

    async def sign_and_post(self, action: Any) -> dict[str, Any]:
        """The ONLY dispatch. Anything not on the allowlist raises before a
        key is touched; the two-key arm is re-checked every call."""
        if not isinstance(action, ALLOWED):
            raise ActionNotAllowed(
                f"{type(action).__name__} is not on the allowlist — "
                f"withdrawals and transfers are structurally unsendable"
            )
        self.assert_armed()
        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils.signing import sign_l1_action  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "hyperliquid-python-sdk not installed — `uv sync --extra hl`"
            ) from exc
        key = os.getenv(self._key_env, "")
        if not key:
            raise NotArmed(f"{self._key_env} not set")
        # The SDK wiring (wallet load, L1 signature, POST /exchange) lands
        # with the live-fire step — the typed-action boundary above is the
        # security boundary and is final; see README Status.
        raise RuntimeError("live signing ships with the fire-test — README Status")
