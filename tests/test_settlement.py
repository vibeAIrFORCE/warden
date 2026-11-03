"""Settlement — 1Click quote/status orchestration over httpx.MockTransport.

Pins the quote BODY (dry:False — never a dry-run quote; the destination is
the HyperCore USDC asset; recipientType DESTINATION_CHAIN; refundTo = the
recipient) plus every failure path and the status-outcome mapping.
Network-free.
"""

import json

import httpx
import pytest

from warden.settlement import (
    DESTINATION_ASSET,
    ORIGIN_ASSETS,
    SettlementError,
    quote,
    status,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------- quote


async def test_quote_happy_body_and_result():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"quote": {
            "depositAddress": "0xsolver123", "amountIn": 25_000_000,
        }})

    async with _client(handler) as http:
        out = await quote("base", 25_000_000, "0xme", http=http)

    assert captured["url"] == "https://1click.chaindefuser.com/v0/quote"
    b = captured["body"]
    assert b["dry"] is False                       # a REAL quote, never dry
    assert b["swapType"] == "EXACT_INPUT"
    assert b["originAsset"] == ORIGIN_ASSETS["base"]
    assert b["destinationAsset"] == DESTINATION_ASSET   # HyperCore USDC
    assert b["amount"] == "25000000"
    assert b["depositType"] == "ORIGIN_CHAIN"
    assert b["recipient"] == "0xme"
    assert b["recipientType"] == "DESTINATION_CHAIN"
    assert b["refundTo"] == "0xme"
    assert b["refundType"] == "ORIGIN_CHAIN"
    assert b["deadline"].endswith("Z")
    assert out == {"deposit_address": "0xsolver123",
                   "amount_in": "25000000",
                   "origin_asset": ORIGIN_ASSETS["base"]}


async def test_quote_origin_chain_is_case_and_space_insensitive():
    def handler(request):
        return httpx.Response(200, json={"quote": {
            "depositAddress": "0xa", "amountIn": "1"}})

    async with _client(handler) as http:
        out = await quote(" BASE ", 1, "0xme", http=http)
    assert out["origin_asset"] == ORIGIN_ASSETS["base"]


async def test_quote_http_400_is_settlement_error():
    def handler(request):
        return httpx.Response(400, text="bad quote")

    async with _client(handler) as http:
        with pytest.raises(SettlementError) as ei:
            await quote("base", 100, "0xme", http=http)
    assert "HTTP 400" in str(ei.value)
    assert "bad quote" in str(ei.value)            # upstream body carried


async def test_quote_missing_deposit_address_is_settlement_error():
    def handler(request):
        return httpx.Response(200, json={"quote": {"amountIn": "1"}})

    async with _client(handler) as http:
        with pytest.raises(SettlementError) as ei:
            await quote("base", 100, "0xme", http=http)
    assert "depositAddress" in str(ei.value)


async def test_quote_unknown_origin_is_settlement_error():
    with pytest.raises(SettlementError) as ei:
        await quote("polygon", 100, "0xme")
    assert "polygon" in str(ei.value)
    assert "base" in str(ei.value)                 # the allowed list is named


async def test_quote_rejects_nonpositive_amount_and_blank_recipient():
    with pytest.raises(SettlementError):
        await quote("base", 0, "0xme")
    with pytest.raises(SettlementError):
        await quote("base", -5, "0xme")
    with pytest.raises(SettlementError):
        await quote("base", 100, "   ")


# ------------------------------------------------------------------ status


def _status_handler(latest):
    def handler(request: httpx.Request) -> httpx.Response:
        history = [{"status": latest}] if latest else []
        return httpx.Response(200, json={"history": history})

    return handler


async def test_status_completed():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"history": [{"status": "COMPLETED"}]})

    async with _client(handler) as http:
        out = await status("0xdep", http=http)

    assert seen["params"]["depositAddress"] == "0xdep"
    assert out["outcome"] == "completed"
    assert out["latest_status"] == "COMPLETED"
    assert out["history"] == [{"status": "COMPLETED"}]


@pytest.mark.parametrize("latest", ["REFUNDED", "REFUNDING", "REFUND_CREATED"])
async def test_status_refunded_family(latest):
    async with _client(_status_handler(latest)) as http:
        out = await status("0xdep", http=http)
    assert out["outcome"] == "refunded"
    assert out["latest_status"] == latest


async def test_status_incomplete_deposit_stays_in_progress():
    """INCOMPLETE_DEPOSIT is deliberately NOT terminal — the user can still
    complete it upstream, so the poll keeps it in_progress."""
    async with _client(_status_handler("INCOMPLETE_DEPOSIT")) as http:
        out = await status("0xdep", http=http)
    assert out["outcome"] == "in_progress"
    assert out["latest_status"] == "INCOMPLETE_DEPOSIT"


async def test_status_empty_history_is_in_progress_with_null_latest():
    async with _client(_status_handler(None)) as http:
        out = await status("0xdep", http=http)
    assert out["outcome"] == "in_progress"
    assert out["latest_status"] is None
    assert out["history"] == []


async def test_status_http_500_is_settlement_error():
    def handler(request):
        return httpx.Response(500, text="nope")

    async with _client(handler) as http:
        with pytest.raises(SettlementError) as ei:
            await status("0xdep", http=http)
    assert "HTTP 500" in str(ei.value)
