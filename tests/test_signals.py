"""Signal providers — the pluggable feed contract.

FileSignalProvider over a tmp JSONL (valid / malformed / wrong-side /
missing-coin) and VibeSignalProvider over httpx.MockTransport (happy parse
+ HTTP failures propagating). Network-free.
"""

import json

import httpx
import pytest

from warden.signals import FileSignalProvider, VibeSignalProvider


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


async def test_vibe_provider_happy_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"signals": [
            {"id": "v1", "token_symbol": "BTC", "side": "LONG",
             "entry_price": 100.5, "conviction": 80},
            {"signal_id": "v2", "coin": "ETH", "side": "short",
             "entry_price": 0, "conviction": None},
            {"id": "v3", "coin": "SOL", "side": "FLAT"},          # dropped
            {"id": "v4", "coin": "BASE", "side": "LONG"},         # no prices
            {"id": "", "coin": "XRP", "side": "LONG"},            # no id -> dropped
        ]})

    async with _client(handler) as http:
        prov = VibeSignalProvider(base_url="https://mock.test/api/",
                                  http=http, limit=7)
        out = await prov.latest()

    assert seen["url"].startswith("https://mock.test/api/top-picks/signals/history")
    assert seen["params"]["limit"] == "7"
    assert [s.signal_id for s in out] == ["v1", "v2", "v4"]
    v1, v2 = out[0], out[1]
    assert v1.coin == "BTC" and v1.side == "LONG"
    assert v1.entry_price == 100.5 and v1.conviction == 80.0
    assert v1.meta == {"source": "vibe"}
    assert v2.side == "SHORT"
    assert v2.entry_price is None and v2.conviction is None


async def test_vibe_provider_rows_key_fallback():
    def handler(request):
        return httpx.Response(200, json={"rows": [
            {"id": "r1", "coin": "BTC", "side": "LONG", "entry_price": 10.0},
        ]})

    async with _client(handler) as http:
        out = await VibeSignalProvider(http=http).latest()
    assert len(out) == 1
    assert out[0].signal_id == "r1" and out[0].entry_price == 10.0


async def test_vibe_provider_empty_payload_returns_empty():
    def handler(request):
        return httpx.Response(200, json={})

    async with _client(handler) as http:
        assert await VibeSignalProvider(http=http).latest() == []


async def test_vibe_provider_http_500_propagates():
    def handler(request):
        return httpx.Response(500, text="boom")

    async with _client(handler) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await VibeSignalProvider(http=http).latest()
