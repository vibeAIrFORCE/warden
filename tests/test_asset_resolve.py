"""Asset-resolution teeth — the resolver, the fail-closed surface, the cache.

Proves:
  * MappingAssetResolver returns the exact AssetMeta it was given;
  * AssetResolver against a stubbed httpx transport resolves the pinned
    testnet universe ids — BTC=3, ETH=4, SOL=0, INJ=26 with maxLeverage
    40/25/10/10 (LIVE-VERIFIED 2026-11-08 against the 213-asset testnet
    universe; these pins live HERE against the stubbed transport, never in
    an executor gate — HL adjusts leverage tiers);
  * an unknown coin raises AssetUnresolved — never a guessed asset id;
  * HTTP/parse failures fail closed (AssetUnresolved, never a guess);
  * the cache: a second resolve makes NO second HTTP call.

Network-free: httpx.AsyncClient is stubbed with a MockTransport.
(The agent fail-closed case — a signal whose coin cannot resolve records
exec_reason asset_unresolved, builds nothing, posts nothing — is carried by
the NEARBRAIN-ADAPTER goal per the single-writer partition; this file pins
the resolver surface only.)
"""

import json

import httpx
import pytest

import warden.assets as assets_mod
from warden.assets import (
    AssetMeta,
    AssetResolver,
    AssetUnresolved,
    MappingAssetResolver,
)

BTC = AssetMeta(asset=3, max_leverage=40, sz_decimals=5)
ETH = AssetMeta(asset=4, max_leverage=25, sz_decimals=4)
SOL = AssetMeta(asset=0, max_leverage=10, sz_decimals=2)
INJ = AssetMeta(asset=26, max_leverage=10, sz_decimals=2)


def _entry(name: str, sz: int, ml: int) -> dict:
    return {"name": name, "szDecimals": sz, "maxLeverage": ml, "isDelisted": False}


def _universe_payload() -> dict:
    """A universe fixture with the pinned ids at their pinned indices.
    Filler entries between them prove the resolver uses the ARRAY INDEX as
    the asset id (skipping entries would shift every later id)."""
    names = ["ADA", "AVAX", "LINK", "DOT", "LTC", "APT", "ARB", "OP",
             "ATOM", "FIL", "NEAR", "SEI", "TIA", "WIF", "BONK", "PENGU",
             "ARBX", "OPX", "ATOM2", "FILX", "NEARX"]
    universe = [_entry("SOL", 2, 10)]          # 0
    universe += [_entry(n, 2, 5) for n in names]  # 1..21
    universe += [_entry("FILLER_A", 1, 3),     # 22
                 _entry("FILLER_B", 1, 3),     # 23
                 _entry("FILLER_C", 1, 3),     # 24
                 _entry("FILLER_D", 1, 3)]     # 25
    universe.append(_entry("INJ", 2, 10))      # 26
    universe[3] = _entry("BTC", 5, 40)         # 3
    universe[4] = _entry("ETH", 4, 25)         # 4
    return {"universe": universe, "marginTables": []}


def _stub_transport(monkeypatch, responder) -> list[httpx.Request]:
    """Route AssetResolver's AsyncClient through a MockTransport; returns
    the list of captured requests."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responder(request)

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler),
                           timeout=kwargs.get("timeout"))

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return calls


# --------------------------------------------------- MappingAssetResolver


async def test_mapping_resolver_returns_exact_meta():
    r = MappingAssetResolver({"BTC": BTC, "INJ": INJ})
    got = await r.resolve("BTC")
    assert type(got) is AssetMeta
    assert got == BTC
    assert (got.asset, got.max_leverage, got.sz_decimals) == (3, 40, 5)
    assert await r.resolve("INJ") == INJ


async def test_mapping_resolver_unknown_coin_raises():
    r = MappingAssetResolver({"BTC": BTC})
    with pytest.raises(AssetUnresolved) as ei:
        await r.resolve("PEPE")
    assert "PEPE" in str(ei.value)


# -------------------------------------------- AssetResolver (stubbed httpx)


async def test_asset_resolver_resolves_the_pinned_universe_ids(monkeypatch):
    """BTC->asset 3, ETH->4, SOL->0, INJ->26, maxLeverage 40/25/10/10 —
    the LIVE-VERIFIED testnet values, pinned HERE against the stubbed
    transport (the array index is the asset id)."""
    calls = _stub_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_universe_payload()),
    )
    r = AssetResolver("https://api.hyperliquid-testnet.xyz")
    got = {coin: await r.resolve(coin) for coin in ("BTC", "ETH", "SOL", "INJ")}
    assert got == {"BTC": BTC, "ETH": ETH, "SOL": SOL, "INJ": INJ}
    assert calls, "the resolver must have made its meta POST"


async def test_resolver_posts_meta_to_info(monkeypatch):
    """ONE httpx POST {"type": "meta"} to {base_url}/info."""
    seen: dict = {}

    def responder(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_universe_payload())

    _stub_transport(monkeypatch, responder)
    r = AssetResolver("https://api.hyperliquid-testnet.xyz")
    await r.resolve("BTC")
    assert seen["url"] == "https://api.hyperliquid-testnet.xyz/info"
    assert seen["body"] == {"type": "meta"}


async def test_unknown_coin_raises_asset_unresolved(monkeypatch):
    _stub_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_universe_payload()),
    )
    r = AssetResolver()
    with pytest.raises(AssetUnresolved) as ei:
        await r.resolve("NOTACOIN")
    assert "NOTACOIN" in str(ei.value)


async def test_http_failure_fails_closed(monkeypatch):
    """An HTTP failure raises AssetUnresolved — never a guess."""

    def responder(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("testnet unreachable", request=req)

    _stub_transport(monkeypatch, responder)
    r = AssetResolver()
    with pytest.raises(AssetUnresolved) as ei:
        await r.resolve("BTC")
    assert "meta fetch failed" in str(ei.value)


async def test_malformed_payload_fails_closed(monkeypatch):
    """A drifted payload shape fails closed (AssetUnresolved), never a
    silent partial map."""

    def responder(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"universe": [{"nope": True}]})

    _stub_transport(monkeypatch, responder)
    r = AssetResolver()
    with pytest.raises(AssetUnresolved) as ei:
        await r.resolve("BTC")
    assert "parse failed" in str(ei.value)


async def test_cache_second_resolve_makes_no_second_http_call(monkeypatch):
    calls = _stub_transport(
        monkeypatch,
        lambda req: httpx.Response(200, json=_universe_payload()),
    )
    r = AssetResolver()
    first = await r.resolve("BTC")
    second = await r.resolve("BTC")
    third = await r.resolve("INJ")
    assert len(calls) == 1                    # one meta POST for the process
    assert first is second                    # cached map, same object
    assert third == INJ
    with pytest.raises(AssetUnresolved):
        await r.resolve("PEPE")               # an unknown coin costs no HTTP
    assert len(calls) == 1


# ------------------------------- Agent fail-closed (NEARBRAIN-ADAPTER goal)
#
# Carried by the NEARBRAIN-ADAPTER goal per the single-writer partition
# (agent.py + tests/test_agent_tick.py are its edit sites): a signal whose
# coin cannot resolve records exec_reason asset_unresolved with
# admitted=true / executed=false — and builds/posts NOTHING. The case rides
# the RecordingExecutor double from test_agent_tick, whose resolve_asset is
# MappingAssetResolver-backed (an unknown coin raises AssetUnresolved
# through the same interface the live executor exposes).


async def test_agent_fail_closed_on_unresolved_coin(tmp_path, monkeypatch):
    from warden.ledger import JsonlLedger

    from test_agent_tick import (
        ListProvider,
        RecordingExecutor,
        agent,
        ledger_rows,
        sig,
    )

    monkeypatch.setenv("WARDEN_DRY", "0")     # fail-closed in LIVE mode too
    ex = RecordingExecutor()
    a = agent(ListProvider([sig("s1", coin="NOPE")]), executor=ex,
              ledger=JsonlLedger(tmp_path / "ledger.jsonl"))

    outs = await a.tick()

    assert len(outs) == 1
    assert outs[0]["verdict"].allowed is True          # the policy DID admit
    assert ex.opens == [] and ex.posted == []          # built and posted NOTHING
    rows = ledger_rows(tmp_path / "ledger.jsonl")
    assert len(rows) == 1
    assert rows[0]["exec_reason"] == "asset_unresolved"
    assert rows[0]["admitted"] is True
    assert rows[0]["executed"] is False
    assert rows[0]["dry"] is False
    assert rows[0]["decision"] is True                 # the brain's own key intact
    assert "asset" not in rows[0]["reason"]            # never clobbered


async def test_agent_fail_closed_on_unresolved_coin_in_dry(tmp_path, monkeypatch):
    from warden.ledger import JsonlLedger

    from test_agent_tick import (
        ListProvider,
        RecordingExecutor,
        agent,
        ledger_rows,
        sig,
    )

    monkeypatch.delenv("WARDEN_DRY", raising=False)    # the default IS dry
    ex = RecordingExecutor()
    a = agent(ListProvider([sig("s1", coin="NOPE")]), executor=ex,
              ledger=JsonlLedger(tmp_path / "ledger.jsonl"))

    outs = await a.tick()

    assert len(outs) == 1
    assert ex.opens == [] and ex.posted == []          # no action in dry either
    rows = ledger_rows(tmp_path / "ledger.jsonl")
    assert rows[0]["exec_reason"] == "asset_unresolved"
    assert rows[0]["dry"] is True
    assert rows[0]["executed"] is False
    assert rows[0]["admitted"] is True
