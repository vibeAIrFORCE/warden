"""The testnet fire-test — resolve / probe / order.

Three modes against Hyperliquid TESTNET ONLY (this script never talks to
prod; it constructs the executor with env="testnet" unconditionally):

  resolve  — network-only, NO SDK, NO key: prints each coin's universe
             metadata (the asset IDS are the pinned stable values;
             maxLeverage/szDecimals print the LIVE resolved values — never
             hardcode them here, HL adjusts leverage tiers).
             Runs under bare `uv run` (no `--extra hl` needed).
  probe    — zero-cost signature proof: self-arms IN-PROCESS ONLY, builds
             an UpdateLeverageAction at 1x and drives sign_and_post.
  order    — THE gate leg: a real testnet order via build_open/sign_and_post
             with the $10-minimum floor asserted before dispatch.

Self-arming is IN-PROCESS ONLY (os.environ below): it exercises
sign_and_post without pre-owning the marker — the REAL exported
WARDEN_TESTNET_FIRE_TEST=pass is set by the operator only after a
successful run of this script.

Secret-safe by construction: this script prints addresses, verdicts,
exchange responses and oids — NEVER key material (the key is read from
WARDEN_PRIVATE_KEY and never echoed).

Operator preconditions (probe/order): `uv sync --extra hl` done once and
WARDEN_PRIVATE_KEY set to a NEW FUNDED HL TESTNET wallet key (created for
Warden, faucet-funded via https://app.hyperliquid-testnet.xyz — never a
production or house key). An unfunded wallet exits nonzero naming the
faucet — never a fake pass.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys

import httpx

from warden.assets import AssetMeta, AssetResolver, AssetUnresolved
from warden.executor import HyperliquidExecutor, NotArmed

TESTNET_BASE_URL = "https://api.hyperliquid-testnet.xyz"
FAUCET = "https://app.hyperliquid-testnet.xyz"
HL_MIN_ORDER_USD = 10.0
RESOLVE_COINS = ("BTC", "ETH", "SOL", "INJ")
_INFO_TIMEOUT_S = 15.0


def _self_arm_in_process() -> None:
    """Set BOTH arm keys in this process only (documented above)."""
    os.environ["WARDEN_ENABLED"] = "1"
    os.environ["WARDEN_TESTNET_FIRE_TEST"] = "pass"


def _testnet_executor() -> HyperliquidExecutor:
    """The fire test is testnet-ONLY — env is never taken from WARDEN_ENV."""
    return HyperliquidExecutor(env="testnet")


async def _info_post(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=_INFO_TIMEOUT_S) as client:
        resp = await client.post(f"{TESTNET_BASE_URL}/info", json=payload)
        resp.raise_for_status()
        return resp.json()


async def _mid(coin: str) -> float:
    data = await _info_post({"type": "allMids"})
    raw = data.get(coin)
    if raw is None:
        raise RuntimeError(f"no {coin} mid in /info allMids (got "
                           f"{len(data)} mids) — is {coin} live on testnet?")
    return float(raw)


# ------------------------------------------------------------------ resolve


async def _mode_resolve() -> int:
    resolver = AssetResolver(TESTNET_BASE_URL)
    failed = False
    for coin in RESOLVE_COINS:
        try:
            meta = await resolver.resolve(coin)
        except AssetUnresolved as exc:
            print(f"{coin} UNRESOLVED: {exc}")
            failed = True
            continue
        print(f"{coin} asset={meta.asset} maxLeverage={meta.max_leverage} "
              f"szDecimals={meta.sz_decimals}")
    return 1 if failed else 0


# -------------------------------------------------------------------- probe


async def _mode_probe(coin: str) -> int:
    _self_arm_in_process()
    executor = _testnet_executor()
    try:
        meta = await executor.resolve_asset(coin)
    except AssetUnresolved as exc:
        print(f"PROBE_FAILED (resolve): {exc}")
        return 1
    if not os.getenv("WARDEN_PRIVATE_KEY"):
        print(f"PROBE_FAILED: WARDEN_PRIVATE_KEY not set — create a NEW "
              f"testnet wallet and fund it at the faucet: {FAUCET}")
        return 1
    action = _probe_action(coin, meta)
    try:
        resp = await executor.sign_and_post(action)
    except NotArmed as exc:  # defensive — the in-process arm cannot miss
        print(f"PROBE_FAILED (arm): {exc}")
        return 1
    except Exception as exc:  # SDK ClientError/ServerError, transport, ...
        # Raised = the exchange rejected at HTTP >= 400 or the transport
        # failed — the signature was NOT verified end-to-end. Fail closed
        # on the evidence marker: this probe is a convenience leg, the
        # order leg's ORDER_OK is the gate.
        print(f"PROBE_FAILED (raised {type(exc).__name__}): {exc}")
        return 1
    # ERR SHAPE (SDK 0.24 verified): the SDK raises only on HTTP >= 400 —
    # an exchange-level refusal arrives as a RETURNED dict
    # {"status": "err", "response": "..."} at HTTP 200, so a returned err
    # dict IS the signature-verified refusal -> transport OK.
    print(f"probe response: {resp}")
    print("PROBE_TRANSPORT_OK")
    return 0


def _probe_action(coin: str, meta: AssetMeta):
    from warden.executor import UpdateLeverageAction

    return UpdateLeverageAction(coin=coin, asset=meta.asset, leverage=1,
                                is_cross=False)


# -------------------------------------------------------------------- order


async def _mode_order(coin: str, notional: float, side: str = "LONG") -> int:
    _self_arm_in_process()
    executor = _testnet_executor()
    try:
        meta = await executor.resolve_asset(coin)
    except AssetUnresolved as exc:
        print(f"ORDER_FAILED (resolve): {exc}")
        return 1
    try:
        mid = await _mid(coin)
    except Exception as exc:
        print(f"ORDER_FAILED (mid): {exc}")
        return 1

    # Round-UP reference size: ceil(notional/mid * 10**sz) / 10**sz. A plain
    # round() can land the effective notional BELOW HL's $10 minimum order
    # value and the order exchange-errs (at the 2026-11-08 testnet BTC mid
    # 75865.5, round(10/mid, 5) = 0.00013 = $9.86 < $10).
    scale = 10 ** meta.sz_decimals
    effective_size = math.ceil(notional / mid * scale) / scale

    # build_open's `size` parameter is NOTIONAL (its sizing is
    # notional/mark) — pass the notional and let the executor's sz_decimals
    # clamp produce the final wired size. The aggressive limit px
    # (mid * 1.01) is rounded to the HL tick rule inside the executor.
    action = executor.build_open(
        coin, asset=meta.asset, side=side, size=notional, mark=mid,
        leverage=1.0,
    )
    wired_size = round(action.size, meta.sz_decimals)
    wired_notional = wired_size * mid
    print(f"{coin} asset={meta.asset} mid={mid} szDecimals={meta.sz_decimals}")
    print(f"ceil-reference size={effective_size} "
          f"(round UP, never plain round)")
    print(f"wired size={wired_size} effective notional=${wired_notional:.2f} "
          f"(HL floor ${HL_MIN_ORDER_USD:.0f})")
    if wired_notional < HL_MIN_ORDER_USD:
        print(f"ORDER_FAILED: wired effective notional ${wired_notional:.2f} "
              f"is below HL's ${HL_MIN_ORDER_USD:.0f} minimum — bump "
              f"--notional (the executor's sz_decimals rounding rounded the "
              f"order under the floor)")
        return 1

    if not os.getenv("WARDEN_PRIVATE_KEY"):
        print(f"ORDER_FAILED: WARDEN_PRIVATE_KEY not set — create a NEW "
              f"testnet wallet and fund it at the faucet: {FAUCET}")
        return 1
    try:
        resp = await executor.sign_and_post(action)
    except Exception as exc:
        print(f"ORDER_FAILED (raised {type(exc).__name__}): {exc}")
        print(f"faucet (fund the testnet wallet): {FAUCET}")
        return 1

    if not isinstance(resp, dict) or resp.get("status") != "ok":
        # Exchange-level refusal (HTTP 200 {"status": "err", ...}) —
        # typically an unfunded wallet / insufficient margin.
        print(f"ORDER_FAILED (exchange err): {resp}")
        print(f"faucet (fund the testnet wallet): {FAUCET}")
        return 1
    statuses = (resp.get("response", {}).get("data", {}) or {}).get(
        "statuses", [])
    oid = _extract_oid(statuses)
    if oid is None:
        print(f"ORDER_FAILED: no resting/filled oid in statuses {statuses}")
        print(f"faucet (fund the testnet wallet): {FAUCET}")
        return 1
    print(f"ORDER_OK")
    print(f"ORDER_OID={oid}")
    print("WARDEN_TESTNET_FIRE_TEST=pass")
    return 0


def _extract_oid(statuses: list) -> int | None:
    for st in statuses:
        if not isinstance(st, dict):
            continue
        for key in ("resting", "filled"):
            inner = st.get(key)
            if isinstance(inner, dict) and inner.get("oid") is not None:
                return int(inner["oid"])
    return None


# --------------------------------------------------------------------- main


async def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fire_test",
        description="Warden testnet fire-test (Hyperliquid TESTNET only).",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("resolve", help="print universe metadata for the demo "
                                   "coins (no SDK, no key)")

    p_probe = sub.add_parser("probe", help="zero-cost signature proof")
    p_probe.add_argument("--coin", default="BTC")

    p_order = sub.add_parser("order", help="the gate leg: a real testnet order")
    p_order.add_argument("--coin", default="BTC")
    p_order.add_argument("--notional", type=float, default=15.0)
    p_order.add_argument("--side", default="LONG", choices=["LONG", "SHORT"])

    args = ap.parse_args(argv)
    if args.mode == "resolve":
        return await _mode_resolve()
    if args.mode == "probe":
        return await _mode_probe(args.coin)
    return await _mode_order(args.coin, args.notional, args.side)


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
