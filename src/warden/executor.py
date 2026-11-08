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
  * dispatch is clamped to the MARKET's own physics (exchange physics, not
    policy): the action's coin is resolved against the live universe and
    the effective leverage is clamped to maxLeverage, order size is rounded
    to szDecimals and the limit px to the HL tick rule — a resolver failure
    propagates BEFORE any SDK call, so an unclamped or unrounded action is
    never dispatched. Clamp-only: no second action is ever auto-issued.

Signing uses hyperliquid-python-sdk when installed (extras: `hl`); without
it the executor builds actions and raises on post — dry mode works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Optional

from .assets import AssetMeta, AssetResolver, AssetUnresolved
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
    def __init__(
        self,
        env: str = "testnet",
        private_key_env: str = "WARDEN_PRIVATE_KEY",
        resolver: AssetResolver | None = None,
    ) -> None:
        if env not in BASE_URLS:
            raise ValueError(f"unknown env {env!r} — use prod|testnet")
        self.env = env
        self.base_url = BASE_URLS[env]
        self._key_env = private_key_env
        self.resolver = resolver if resolver is not None else AssetResolver(self.base_url)

    async def resolve_asset(self, coin: str) -> AssetMeta:
        """Coin symbol -> universe metadata (asset id / maxLeverage /
        szDecimals). Cached by the resolver; unknown coins raise
        AssetUnresolved — fail closed, never a guessed id."""
        return await self.resolver.resolve(coin)

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

    async def _clamp_to_market(self, action: Any) -> Any:
        """Resolve the action's coin and clamp it to the market's own
        physics — closes policy.py's promise that "the market's own max
        clamps again at the executor". Exchange physics, not policy: the
        policy layer's text is untouched.

        Fail closed: a resolver failure (AssetUnresolved or any HTTP/parse
        failure) propagates — nothing is dispatched, an unclamped or
        unrounded action never reaches the wire. Clamp-only: this never
        issues a second action (no auto-updateLeverage before an order);
        for OrderAction the clamped leverage lives on the action (metadata
        + to_wire) — the SDK's order() carries no leverage field, HL
        leverage is account-level and wire-effective only via
        UpdateLeverageAction's int(leverage). Isolated-only is preserved:
        is_cross is carried through unchanged (warden constructs False).
        """
        if isinstance(action, CancelAction):
            return action  # a cancel carries no leverage/size to clamp
        meta = await self.resolve_asset(action.coin)
        if isinstance(action, OrderAction):
            return replace(
                action,
                leverage=min(action.leverage, meta.max_leverage),
                size=round(action.size, meta.sz_decimals),
                limit_px=round(
                    float(f"{action.limit_px:.5g}"), 6 - meta.sz_decimals
                ),
            )
        # UpdateLeverageAction
        return replace(action, leverage=min(action.leverage, meta.max_leverage))

    async def sign_and_post(self, action: Any) -> dict[str, Any]:
        """The ONLY dispatch. Anything not on the allowlist raises before a
        key is touched; the two-key arm is re-checked every call; the
        market clamp runs after the arm gate and before any SDK call."""
        if not isinstance(action, ALLOWED):
            raise ActionNotAllowed(
                f"{type(action).__name__} is not on the allowlist — "
                f"withdrawals and transfers are structurally unsendable"
            )
        self.assert_armed()
        try:
            import eth_account  # noqa: F401

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
        # Market clamp FIRST — a resolver failure propagates before the
        # wallet is even loaded, so nothing unclamped/unrounded dispatches.
        # The typed-action boundary above is the security boundary and is
        # final; see README Status.
        action = await self._clamp_to_market(action)
        wallet = eth_account.Account.from_key(key)
        exchange = Exchange(wallet, self.base_url)
        if isinstance(action, OrderAction):
            return exchange.order(
                action.coin,
                action.is_buy,
                action.size,
                action.limit_px,
                {"limit": {"tif": "Ioc" if action.order_type == "ioc" else "Gtc"}},
                reduce_only=action.reduce_only,
            )
        if isinstance(action, CancelAction):
            return exchange.cancel(action.coin, action.oid)
        # UpdateLeverageAction — int(leverage) is the wire-effective clamp.
        return exchange.update_leverage(
            int(action.leverage), action.coin, is_cross=action.is_cross
        )
