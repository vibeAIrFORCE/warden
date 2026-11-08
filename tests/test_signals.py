"""Signal providers — the pluggable feed contract.

FileSignalProvider over a tmp JSONL (valid / malformed / wrong-side /
missing-coin) and the priced VibeSignalProvider over httpx.MockTransport:
the seven-fixture honesty battery (happy parse / 200-empty / 402 honest
raise / 429 honest raise / credential tri-state / error propagation /
parse-drop loud failure). Network-free.
"""

import json

import httpx
import pytest

from warden.signals import (
    FileSignalProvider,
    SignalsUnavailable,
    VibeSignalProvider,
)


def _write(tmp_path, rows) -> object:
    """Rows: dicts are JSON-dumped; strings land verbatim (malformed lines)."""
    p = tmp_path / "signals.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows)
        + "\n"
    )
    return p


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------ FileSignalProvider


async def test_file_provider_valid_row(tmp_path):
    p = _write(tmp_path, [{
        "signal_id": "s1", "coin": "BTC", "side": "LONG",
        "entry_price": "100.5", "conviction": 72,
    }])
    out = await FileSignalProvider(p).latest()
    assert len(out) == 1
    s = out[0]
    assert s.signal_id == "s1"
    assert s.coin == "BTC"
    assert s.side == "LONG"
    assert s.entry_price == 100.5
    assert s.conviction == 72.0
    assert s.meta == {"source": "file"}


async def test_file_provider_filters_malformed_wrong_side_missing_coin(tmp_path):
    rows = [
        "this is not json {{{",                                       # malformed
        "",                                                           # blank
        {"signal_id": "s2", "coin": "ETH", "side": "buy"},            # wrong side
        {"signal_id": "s3", "coin": "SOL", "side": "FLAT"},           # wrong side
        {"signal_id": "s4", "side": "LONG"},                          # missing coin
        {"coin": "DOGE", "side": "LONG"},                             # missing signal_id
        {"signal_id": "s5", "coin": "BTC", "side": "SHORT",
         "entry_price": "abc"},                                       # valid; price unparsable
    ]
    out = await FileSignalProvider(_write(tmp_path, rows)).latest()
    assert [s.signal_id for s in out] == ["s5"]
    s = out[0]
    assert s.side == "SHORT"
    assert s.entry_price is None
    assert s.conviction is None


async def test_file_provider_missing_file_returns_empty(tmp_path):
    assert await FileSignalProvider(tmp_path / "nope.jsonl").latest() == []


async def test_file_provider_zero_or_negative_prices_are_none(tmp_path):
    rows = [
        {"signal_id": "z1", "coin": "BTC", "side": "LONG", "entry_price": 0},
        {"signal_id": "z2", "coin": "ETH", "side": "LONG", "conviction": -3},
    ]
    out = await FileSignalProvider(_write(tmp_path, rows)).latest()
    assert [s.signal_id for s in out] == ["z1", "z2"]
    assert out[0].entry_price is None
    assert out[1].conviction is None


# ------------------------------------------------------ VibeSignalProvider


def _delivery(sid="v1", coin="BTC", side="LONG", exited_at=None, **extra):
    """One SKU wire row per the goal-3 pinned contract (verbatim fields)."""
    row = {
        "id": sid,
        "token_symbol": coin,
        "side": side,
        "conviction": 80,
        "entry_price": 100.5,
        "entered_at": "2026-11-08T10:00:00+00:00",
        "status": "resolved" if exited_at else "open",
        "exit_price": 110.0 if exited_at else None,
        "exited_at": exited_at,
        "reason": "pending" if exited_at else "open",
        "return_pct": 9.5 if exited_at else None,
    }
    row.update(extra)
    return row


async def test_vibe_provider_happy_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={
            "success": True,
            "data": {"deliveries": [
                _delivery("v1", "BTC", "LONG"),
                _delivery("v2", "ETH", side="short",
                          entry_price=0, conviction=None),
                _delivery("v3", "SOL", side="FLAT"),               # dropped
                _delivery("v4", "BASE"),                           # no prices
                _delivery("", "XRP"),                              # no id -> dropped
            ], "count": 5},
            "meta": {"source": "supabase:top_perps_tg_signals",
                     "price_usdc": 0.10, "payment_mode": "credit"},
        })

    async with _client(handler) as http:
        prov = VibeSignalProvider(base_url="https://mock.test/api/",
                                  http=http, limit=7)
        out = await prov.latest()

    assert seen["url"].startswith("https://mock.test/api/x402/alpha/jev/deliveries")
    assert seen["params"]["limit"] == "7"
    assert seen["params"]["status"] == "open"
    assert [s.signal_id for s in out] == ["v1", "v2", "v4"]
    v1, v2 = out[0], out[1]
    assert v1.coin == "BTC" and v1.side == "LONG"
    assert v1.entry_price == 100.5 and v1.conviction == 80.0
    assert v1.meta == {"source": "vibe", "payment_mode": "credit"}
    assert v2.side == "SHORT"
    assert v2.entry_price is None and v2.conviction is None


async def test_vibe_provider_rows_key_fallback():
    def handler(request):
        return httpx.Response(200, json={
            "success": True,
            "data": [
                {"id": "r1", "coin": "BTC", "side": "LONG",
                 "entry_price": 10.0, "exited_at": None},
            ],
        })

    async with _client(handler) as http:
        out = await VibeSignalProvider(http=http).latest()
    assert len(out) == 1
    assert out[0].signal_id == "r1" and out[0].entry_price == 10.0


async def test_vibe_provider_empty_payload_returns_empty():
    def handler(request):
        return httpx.Response(200, json={
            "success": True, "data": {"deliveries": [], "count": 0},
        })

    async with _client(handler) as http:
        assert await VibeSignalProvider(http=http).latest() == []


async def test_vibe_provider_http_500_propagates():
    """402/429 map to SignalsUnavailable; EVERY other non-200 propagates —
    [] is returned ONLY on HTTP 200."""
    def handler(request):
        return httpx.Response(500, text="boom")

    async with _client(handler) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await VibeSignalProvider(http=http).latest()


async def test_vibe_provider_402_raises_signals_unavailable():
    """Fixture 3 — INSUFFICIENT_CREDITS / x402 payment challenge is honest
    unavailability, NEVER a silent empty book."""
    def handler(request):
        return httpx.Response(402, json={
            "success": False,
            "error": {"code": "INSUFFICIENT_CREDITS",
                      "message": "credit balance too low"},
        })

    async with _client(handler) as http:
        with pytest.raises(SignalsUnavailable):
            await VibeSignalProvider(http=http).latest()


async def test_vibe_provider_402_payment_challenge_raises():
    """The keyless x402 payment-required challenge answers 402 too — same
    honest raise."""
    def handler(request):
        return httpx.Response(402, text="payment required", headers={
            "payment-required": "eyJ4NDAyVmVyc2lvbiI6Mn0="})

    async with _client(handler) as http:
        with pytest.raises(SignalsUnavailable):
            await VibeSignalProvider(http=http).latest()


async def test_vibe_provider_429_raises_signals_unavailable():
    """Fixture 4 — trial budget exhausted (429) is honest unavailability."""
    def handler(request):
        return httpx.Response(429, json={
            "success": False,
            "error": {"code": "TRIAL_DAILY_LIMIT",
                      "message": "trial token exhausted"},
        })

    async with _client(handler) as http:
        with pytest.raises(SignalsUnavailable):
            await VibeSignalProvider(http=http).latest()


def _header_provider(monkeypatch, *, api_env=None, trial_env=None,
                     **kwargs) -> VibeSignalProvider:
    monkeypatch.delenv("VIBE_API_KEY", raising=False)
    monkeypatch.delenv("VIBE_TRIAL_TOKEN", raising=False)
    if api_env is not None:
        monkeypatch.setenv("VIBE_API_KEY", api_env)
    if trial_env is not None:
        monkeypatch.setenv("VIBE_TRIAL_TOKEN", trial_env)
    return VibeSignalProvider(**kwargs)


async def test_vibe_provider_credential_tri_state(monkeypatch):
    """Fixture 5 — the request carries the header per env: key lane wins
    over trial; trial only when no key; keyless otherwise."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x_api_key"] = request.headers.get("x-api-key")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "success": True, "data": {"deliveries": [], "count": 0}})

    async with _client(handler) as http:
        # key wins when both env vars are set
        await _header_provider(monkeypatch, api_env="pk_a:sk_b", trial_env="trial_t",
                               http=http).latest()
        assert seen["x_api_key"] == "pk_a:sk_b"
        assert seen["authorization"] is None
        # trial lane
        await _header_provider(monkeypatch, trial_env="trial_t", http=http).latest()
        assert seen["x_api_key"] is None
        assert seen["authorization"] == "Bearer trial_t"
        # keyless fallback
        await _header_provider(monkeypatch, http=http).latest()
        assert seen["x_api_key"] is None
        assert seen["authorization"] is None

    async with _client(handler) as http:
        # explicit constructor args behave identically to the env lanes
        await VibeSignalProvider(api_key="pk_a:sk_b", trial_token="trial_t",
                                 http=http).latest()
        assert seen["x_api_key"] == "pk_a:sk_b"
        assert seen["authorization"] is None
        await VibeSignalProvider(trial_token="trial_t", http=http).latest()
        assert seen["x_api_key"] is None
        assert seen["authorization"] == "Bearer trial_t"
        await VibeSignalProvider(http=http).latest()
        assert seen["x_api_key"] is None
        assert seen["authorization"] is None


async def test_vibe_provider_parse_dropped_all_rows_raises():
    """Fixture 7 — rows present and open, but every side wrong-shaped:
    payload drift must be LOUD (SignalsUnavailable), never []."""
    def handler(request):
        return httpx.Response(200, json={
            "success": True,
            "data": {"deliveries": [
                _delivery("w1", "BTC", side="buy"),
                _delivery("w2", "ETH", side="FLAT"),
                _delivery("w3", "SOL", side=""),
            ], "count": 3},
        })

    async with _client(handler) as http:
        with pytest.raises(SignalsUnavailable) as ei:
            await VibeSignalProvider(http=http).latest()
    assert ei.value.reason == "parse_dropped_all_rows"


async def test_vibe_provider_exited_rows_all_skipped_returns_empty():
    """Only resolved rows arrive (status=open defensively filtered
    client-side too): every row hits the exited_at skip -> honest [] (no
    candidates survived, so no parse-drift raise)."""
    def handler(request):
        return httpx.Response(200, json={
            "success": True,
            "data": {"deliveries": [
                _delivery("d1", exited_at="2026-11-08T11:00:00+00:00"),
                _delivery("d2", exited_at="2026-11-08T12:00:00+00:00"),
            ], "count": 2},
        })

    async with _client(handler) as http:
        out = await VibeSignalProvider(http=http).latest()
    assert out == []


async def test_vibe_provider_meta_payment_mode_absent_when_unmarked():
    """preview/keyless lanes carry no meta.payment_mode — meta stays honest."""
    def handler(request):
        return httpx.Response(200, json={
            "success": True,
            "data": {"deliveries": [_delivery()], "count": 1},
            "meta": {"source": "supabase:top_perps_tg_signals",
                     "price_usdc": 0.10},
        })

    async with _client(handler) as http:
        out = await VibeSignalProvider(http=http).latest()
    assert out[0].meta == {"source": "vibe", "payment_mode": None}
