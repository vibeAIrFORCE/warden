"""Brains — the decision layer contract.

HeuristicBrain's verdicts (follow at floor+, exactly-at-floor, refuse below,
refuse None) and ModelBrain's fail-closed construction + inference stub.
Network-free.
"""

import json

import httpx
import pytest

from warden.brain import (
    HeuristicBrain,
    ModelBrain,
    ModelNotReady,
    NearBrain,
    NearBrainUnavailable,
)
from warden.types import PositionState, Signal


def sig(conviction=70.0) -> Signal:
    return Signal(signal_id="s1", coin="BTC", side="LONG",
                  entry_price=100.0, conviction=conviction)


# ---------------------------------------------------------- HeuristicBrain


async def test_heuristic_follows_above_floor():
    d = await HeuristicBrain(min_conviction=60.0, size_usd=10.0).decide(
        sig(70.0), PositionState())
    assert d.follow is True
    assert d.size_usd == 10.0
    assert d.leverage == 1.0
    assert d.source == "heuristic"
    assert ">=" in d.reason


async def test_heuristic_follows_exactly_at_floor():
    d = await HeuristicBrain(min_conviction=60.0).decide(
        sig(60.0), PositionState())
    assert d.follow is True


async def test_heuristic_refuses_below_floor():
    d = await HeuristicBrain(min_conviction=60.0).decide(
        sig(59.0), PositionState())
    assert d.follow is False
    assert d.size_usd is None
    assert "conviction 59 < floor 60" in d.reason


async def test_heuristic_refuses_missing_conviction():
    d = await HeuristicBrain().decide(sig(conviction=None), PositionState())
    assert d.follow is False
    assert "no conviction on signal" in d.reason


# -------------------------------------------------------------- ModelBrain


def test_model_brain_missing_dir_fails_closed(tmp_path):
    with pytest.raises(ModelNotReady) as ei:
        ModelBrain(tmp_path / "absent")
    assert "does not exist" in str(ei.value)


def test_model_brain_dir_without_config_fails_closed(tmp_path):
    (tmp_path / "my-model").mkdir()
    with pytest.raises(ModelNotReady) as ei:
        ModelBrain(tmp_path / "my-model")
    assert "warden_model.json" in str(ei.value)


async def test_model_brain_inference_not_ready_until_weights(tmp_path):
    """A valid model dir constructs — but decide() is a hard stop until the
    weights ship. The agent fails closed instead of guessing."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "warden_model.json").write_text('{"arch": "4b-gate"}')
    brain = ModelBrain(d)
    with pytest.raises(ModelNotReady) as ei:
        await brain.decide(sig(), PositionState())
    assert "inference ships with the weights" in str(ei.value)


# --------------------------------------------------------------- NearBrain
#
# The remote veto brain over httpx.MockTransport — network-free, the same
# handler-injection pattern the signal-provider tests use. Every test
# injects the client (NearBrain never constructs a live one in tests) and
# passes api_key explicitly, because the missing-key refusal path otherwise
# fires before the mock transport is ever reached (test 13 omits it
# deliberately).


def _clean_env(monkeypatch) -> None:
    """Neutralize the operator env — the key var NearBrain reads at call
    time (an os.environ name) and the brain URL. Never assume unset."""
    monkeypatch.delenv("NEAR_API_KEY", raising=False)  # os.environ name, never a value
    monkeypatch.delenv("WARDEN_BRAIN_URL", raising=False)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _reply(content: str | None) -> dict:
    return {"choices": [{"message": {"content": content}}]}


_ENTER = '{"action":"enter","probabilities":{"enter":0.7,"hold":0.2,"exit":0.1}}'


def _brain(http, **kw) -> NearBrain:
    kw.setdefault("base_url", "https://mock.test/v1")
    kw.setdefault("api_key", "test-key")
    return NearBrain(http=http, **kw)


async def test_nearbrain_enter_with_margin_follows(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_reply(_ENTER))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is True
    assert d.size_usd == 10.0
    assert d.leverage == 1.0
    assert d.source == "nearbrain"
    assert len(calls) == 1


async def test_nearbrain_argmax_exit_refuses_as_exit_veto(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"exit","probabilities":{"enter":0.1,"hold":0.2,"exit":0.7}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "nearbrain exit-veto" in d.reason


async def test_nearbrain_argmax_hold_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"hold","probabilities":{"enter":0.1,"hold":0.7,"exit":0.2}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert d.size_usd is None
    assert "hold" in d.reason


async def test_nearbrain_low_margin_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"enter","probabilities":{"enter":0.45,"hold":0.35,"exit":0.2}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "margin" in d.reason


async def test_nearbrain_unparseable_content_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply("I am not JSON at all"))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "not valid JSON" in d.reason


async def test_nearbrain_missing_probabilities_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply('{"action":"enter"}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "probabilities" in d.reason


async def test_nearbrain_extra_probability_key_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"enter","probabilities":'
            '{"enter":0.7,"hold":0.2,"exit":0.1,"other":0.0}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "exactly enter/hold/exit" in d.reason


async def test_nearbrain_nan_probability_refuses(monkeypatch):
    # json.loads tolerates NaN literals; a NaN also passes every '<'
    # comparison downstream — only the isfinite gate catches it.
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"enter","probabilities":'
            '{"enter":NaN,"hold":0.4,"exit":0.1}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "not in [0, 1]" in d.reason


async def test_nearbrain_bool_probability_refuses(monkeypatch):
    # isinstance(True, int) is True in Python — the bool check must fire
    # BEFORE the number check.
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"enter","probabilities":'
            '{"enter":true,"hold":0.2,"exit":0.1}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "not a JSON number" in d.reason


async def test_nearbrain_integral_probabilities_accepted(monkeypatch):
    # JSON integral numbers are VALID replies — float() coercion, never a
    # bare isinstance(v, float).
    def handler(request):
        return httpx.Response(200, json=_reply(
            '{"action":"enter","probabilities":{"enter":1,"hold":0,"exit":0}}'))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is True
    assert d.size_usd == 10.0


async def test_nearbrain_http_500_retries_once_then_refuses(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="boom")

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert len(calls) == 2          # exactly ONE immediate retry
    assert "500" in d.reason


async def test_nearbrain_http_429_retry_then_200_parses(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_reply(_ENTER))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is True
    assert len(calls) == 2


async def test_nearbrain_timeout_single_attempt_no_retry(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.TimeoutException("timed out")

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert len(calls) == 1          # NO retry on timeout
    assert "TimeoutException" in d.reason


async def test_nearbrain_missing_operator_key_refuses_before_any_http(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_reply(_ENTER))

    _clean_env(monkeypatch)         # the key env is deleted — no api_key arg
    async with _client(handler) as http:
        d = await _brain(http, api_key=None).decide(sig(), PositionState())

    assert d.follow is False
    assert calls == []              # refusal fired before the wire
    assert "key not set" in d.reason


def test_nearbrain_no_base_url_and_no_env_raises_unavailable(monkeypatch):
    _clean_env(monkeypatch)         # DEL the env var — never assume unset
    with pytest.raises(NearBrainUnavailable) as ei:
        NearBrain(api_key="test-key")
    assert "WARDEN_BRAIN_URL not set" in str(ei.value)


async def test_nearbrain_wire_shape_bearer_header_and_url(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_reply(_ENTER))

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        await _brain(http, base_url="https://mock.test/v1/").decide(
            sig(), PositionState())

    req = calls[0]
    assert str(req.url) == "https://mock.test/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer test-key"
    assert "test-key" not in str(req.url)   # the key rides the header only
    body = json.loads(req.content)
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


async def test_nearbrain_response_missing_choices_refuses(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={})

    _clean_env(monkeypatch)
    async with _client(handler) as http:
        d = await _brain(http).decide(sig(), PositionState())

    assert d.follow is False
    assert "choices" in d.reason
