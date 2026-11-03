"""Settlement — deposit from any chain into your Hyperliquid balance.

NEAR Intents 1Click underneath: quote an EXACT_INPUT swap into HyperCore
USDC, show the solver deposit address, poll the status until terminal.
The user sends from their own wallet — this module orchestrates quotes and
status reads and NEVER moves funds. Your EVM address IS your HL account:
the funds land as perps collateral directly, not spot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

BASE_URL = "https://1click.chaindefuser.com"
# HyperCore USDC (HIP-1) — lands as perps balance, not spot.
DESTINATION_ASSET = "1cs_v1:hypercore:hip1:0x6d1e7cde53ba9467b783cb7c530ce054"

# USDC origin allowlist (verified against 1Click's /v0/tokens registry —
# omnifold ids carry the .omft.near suffix).
ORIGIN_ASSETS = {
    "base": "nep141:base-0x833589fcd6edb6e08f4c7c32d4f71b54bda02913.omft.near",
    "ethereum": "nep141:eth-0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48.omft.near",
    "arbitrum": "nep141:arb-0xaf88d065e77c8cc2239327c5edb3a432268e5831.omft.near",
    "solana": "nep141:sol-5ce3bf3a31af18be40ba30f721101b4341690186.omft.near",
}


class SettlementError(RuntimeError):
    pass


async def quote(origin_chain: str, amount_base_units: int, recipient: str,
                http: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    """EXACT_INPUT quote into HyperCore USDC. The user must send EXACTLY
    `amountIn` to the returned depositAddress — any other amount stalls
    upstream in INCOMPLETE_DEPOSIT."""
    origin = ORIGIN_ASSETS.get((origin_chain or "").strip().lower())
    if origin is None:
        raise SettlementError(
            f"unknown origin {origin_chain!r} — allowed: {sorted(ORIGIN_ASSETS)}"
        )
    if amount_base_units <= 0:
        raise SettlementError("amount must be positive")
    if not recipient or not recipient.strip():
        raise SettlementError("recipient (your EVM address) is required")
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    body = {
        "dry": False,
        "swapType": "EXACT_INPUT",
        "slippageTolerance": 100,
        "originAsset": origin,
        "destinationAsset": DESTINATION_ASSET,
        "amount": str(int(amount_base_units)),
        "depositType": "ORIGIN_CHAIN",
        "recipient": recipient.strip(),
        "recipientType": "DESTINATION_CHAIN",
        "refundTo": recipient.strip(),
        "refundType": "ORIGIN_CHAIN",
        "deadline": deadline,
    }
    own = http is None
    http = http or httpx.AsyncClient(timeout=20.0)
    try:
        res = await http.post(f"{BASE_URL}/v0/quote", json=body)
        if res.status_code >= 400:
            raise SettlementError(f"1Click quote HTTP {res.status_code}: {res.text[:200]}")
        q = (res.json() or {}).get("quote") or {}
        addr = q.get("depositAddress")
        amount_in = q.get("amountIn")
        if not addr or not amount_in:
            raise SettlementError("1Click quote missing depositAddress/amountIn")
        return {"deposit_address": addr, "amount_in": str(amount_in),
                "origin_asset": origin}
    finally:
        if own:
            await http.aclose()


async def status(deposit_address: str,
                 http: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    """Poll a deposit to terminal. `outcome` ∈ in_progress | completed |
    refunded (INCOMPLETE_DEPOSIT stays in_progress — the user can still
    complete it upstream)."""
    own = http is None
    http = http or httpx.AsyncClient(timeout=15.0)
    try:
        res = await http.get(f"{BASE_URL}/v0/status",
                             params={"depositAddress": deposit_address})
        if res.status_code >= 400:
            raise SettlementError(f"1Click status HTTP {res.status_code}")
        history = (res.json() or {}).get("history") or []
        latest = history[0].get("status") if history else None
        if latest in ("COMPLETED",):
            outcome = "completed"
        elif latest in ("REFUNDED", "REFUNDING", "REFUND_CREATED"):
            outcome = "refunded"
        else:
            outcome = "in_progress"
        return {"outcome": outcome, "latest_status": latest,
                "history": history}
    finally:
        if own:
            await http.aclose()
