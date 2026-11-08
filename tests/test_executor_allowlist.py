"""The executor's teeth — the typed-action allowlist, the two-key arm, and
the market clamp.

Proves, in order:
  * a raw dict can never reach signing — ActionNotAllowed fires BEFORE any
    env read (proven on an unarmed executor: if the arm gate, which reads
    env, ran first, we would see NotArmed instead);
  * the arm gate is a two-key AND, re-checked every call — full matrix;
  * build_open's sizing/limit math is exact (notional / mark, mark ±1%);
  * the allowlist is a TYPE check: UpdateLeverage/Cancel pass isinstance and
    die at the arm gate instead — asserted by exception TYPE;
  * the fully-armed path dispatches through the SDK's Exchange — recorded
    by a callable fake (constructible as Exchange(wallet, base_url));
  * the market clamp: leverage above the resolved max is clamped, at/below
    passes through, isolated-only is preserved, order size/limit px are
    rounded to the market's szDecimals / HL tick rule, and a resolver
    failure propagates BEFORE any Exchange call (fail-closed).

Network-free: the optional SDK import is stubbed (including eth_account,
which ships only with the `hl` extra) and every dispatch/clamp test
injects a MappingAssetResolver — the default AssetResolver makes a real
network call inside a unit test.
"""

import sys
import types

import pytest

from warden.assets import AssetMeta, AssetUnresolved, MappingAssetResolver
from warden.executor import (
    ActionNotAllowed,
    CancelAction,
    HyperliquidExecutor,
    NotArmed,
    OrderAction,
    UpdateLeverageAction,
)

# Syntactically VALID 32-byte hex key (the public anvil throwaway) —
# eth_account.Account.from_key would reject anything shorter before any
# dispatch; the key never signs anything because the Exchange is the fake.
VALID_TEST_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

BTC = AssetMeta(asset=3, max_leverage=40, sz_decimals=5)
ETH = AssetMeta(asset=4, max_leverage=25, sz_decimals=4)
SOL = AssetMeta(asset=0, max_leverage=10, sz_decimals=2)


def ex(env: str = "testnet") -> HyperliquidExecutor:
    return HyperliquidExecutor(env=env)


def ex_resolved(**metas: AssetMeta) -> HyperliquidExecutor:
    """Executor with a MappingAssetResolver (network-free) — default BTC."""
    mapping = {"BTC": BTC, "ETH": ETH, "SOL": SOL}
    mapping.update(metas)
    return HyperliquidExecutor(env="testnet", resolver=MappingAssetResolver(mapping))


def _order() -> OrderAction:
    return OrderAction(coin="BTC", asset=0, is_buy=True, size=0.1,
                       limit_px=101.0)


class FakeWallet:
    def __init__(self, key: str) -> None:
        self.key = key


class FakeExchange:
    """Callable recorder fake — constructible as Exchange(wallet, base_url);
    order/cancel/update_leverage record their args and return a parsed
    response dict shaped like the SDK's."""

    instances: list["FakeExchange"] = []

    def __init__(self, wallet, base_url=None) -> None:
        self.wallet = wallet
        self.base_url = base_url
        self.calls: list[tuple] = []
        FakeExchange.instances.append(self)

    def order(self, coin, is_buy, size, limit_px, order_type, reduce_only=False):
        self.calls.append(("order", coin, is_buy, size, limit_px,
                           order_type, reduce_only))
        return {"status": "ok", "response": {"type": "order", "data": {
            "statuses": [{"resting": {"oid": 1}}]}}}

    def cancel(self, coin, oid):
        self.calls.append(("cancel", coin, oid))
        return {"status": "ok", "response": {"type": "cancel"}}

    def update_leverage(self, leverage, coin, is_cross=False):
        self.calls.append(("update_leverage", leverage, coin, is_cross))
        return {"status": "ok", "response": {"type": "updateLeverage"}}


def _install_fake_sdk(monkeypatch) -> type[FakeExchange]:
    """hyperliquid-python-sdk is an optional extra — stub the imports
    sign_and_post makes so the full arming chain is testable network-free.
    eth_account ships only with the `hl` extra, so it is stubbed too (a
    minimal fake whose Account.from_key returns a stub wallet)."""
    hl = types.ModuleType("hyperliquid")
    info = types.ModuleType("hyperliquid.info")
    info.Info = object
    exch = types.ModuleType("hyperliquid.exchange")
    exch.Exchange = FakeExchange
    utils = types.ModuleType("hyperliquid.utils")
    signing = types.ModuleType("hyperliquid.utils.signing")
    signing.sign_l1_action = lambda *a, **k: b""
    hl.info, hl.exchange, hl.utils = info, exch, utils
    eth = types.ModuleType("eth_account")

    class FakeAccount:
        @staticmethod
        def from_key(key: str) -> FakeWallet:
            return FakeWallet(key)

    eth.Account = FakeAccount
    for name, mod in [
        ("hyperliquid", hl), ("hyperliquid.info", info),
        ("hyperliquid.exchange", exch), ("hyperliquid.utils", utils),
        ("hyperliquid.utils.signing", signing), ("eth_account", eth),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    FakeExchange.instances.clear()
    return FakeExchange


def _arm(monkeypatch) -> type[FakeExchange]:
    """Full arm + fake SDK + valid key — the dispatch-ready test rig."""
    monkeypatch.setenv("WARDEN_ENABLED", "1")
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", "pass")
    monkeypatch.setenv("WARDEN_PRIVATE_KEY", VALID_TEST_KEY)
    return _install_fake_sdk(monkeypatch)


# ---------------------------------------------------------------- allowlist


async def test_dict_action_refused_before_any_env_read(monkeypatch):
    """Unarmed executor + a raw dict: the TYPE gate must fire first. If the
    arm check (an env read) ran before it, this would raise NotArmed."""
    monkeypatch.delenv("WARDEN_ENABLED", raising=False)
    monkeypatch.delenv("WARDEN_TESTNET_FIRE_TEST", raising=False)
    with pytest.raises(ActionNotAllowed) as ei:
        await ex().sign_and_post({"type": "withdraw", "amount": "1000000"})
    assert type(ei.value) is ActionNotAllowed
    assert "not on the allowlist" in str(ei.value)
    assert "withdrawals" in str(ei.value)


async def test_withdraw_shaped_class_still_refused():
    """A lookalike class is not on the allowlist either — the check is on
    the exact frozen action types, not on shape."""

    class WithdrawAction:
        coin = "USDC"

    with pytest.raises(ActionNotAllowed) as ei:
        await ex().sign_and_post(WithdrawAction())
    assert "WithdrawAction" in str(ei.value)


async def test_order_action_passes_allowlist_then_hits_arm_gate(monkeypatch):
    """The happy-type case dies at the arm gate, not the type gate — the
    exception TYPE proves the allowlist let it through."""
    monkeypatch.delenv("WARDEN_ENABLED", raising=False)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert type(ei.value) is NotArmed


async def test_update_leverage_passes_allowlist_then_arm_gate(monkeypatch):
    monkeypatch.delenv("WARDEN_ENABLED", raising=False)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(
            UpdateLeverageAction(coin="BTC", asset=0, leverage=2))
    assert type(ei.value) is NotArmed


async def test_cancel_passes_allowlist_then_arm_gate(monkeypatch):
    monkeypatch.delenv("WARDEN_ENABLED", raising=False)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(CancelAction(coin="BTC", asset=0, oid=42))
    assert type(ei.value) is NotArmed


# ------------------------------------------------------------ the arm gate


async def test_notarmed_when_no_env(monkeypatch):
    monkeypatch.delenv("WARDEN_ENABLED", raising=False)
    monkeypatch.delenv("WARDEN_TESTNET_FIRE_TEST", raising=False)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert "WARDEN_ENABLED not set" in str(ei.value)


@pytest.mark.parametrize("flag", ["0", "false", "off", "", "no"])
async def test_notarmed_flag_invalid_values(monkeypatch, flag):
    monkeypatch.setenv("WARDEN_ENABLED", flag)
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", "pass")
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert "WARDEN_ENABLED" in str(ei.value)


@pytest.mark.parametrize("flag", ["1", "true", "on", "TRUE", "On "])
async def test_flag_accepts_truthy_spellings_then_fire_gate(monkeypatch, flag):
    monkeypatch.setenv("WARDEN_ENABLED", flag)
    monkeypatch.delenv("WARDEN_TESTNET_FIRE_TEST", raising=False)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert "WARDEN_TESTNET_FIRE_TEST" in str(ei.value)


@pytest.mark.parametrize("fire", ["fail", "pass1", "", "pending"])
async def test_notarmed_fire_test_not_pass(monkeypatch, fire):
    monkeypatch.setenv("WARDEN_ENABLED", "1")
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", fire)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert "fire-test" in str(ei.value)


async def test_armed_but_no_private_key(monkeypatch):
    monkeypatch.setenv("WARDEN_ENABLED", "1")
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", "pass")
    monkeypatch.delenv("WARDEN_PRIVATE_KEY", raising=False)
    _install_fake_sdk(monkeypatch)
    with pytest.raises(NotArmed) as ei:
        await ex().sign_and_post(_order())
    assert "WARDEN_PRIVATE_KEY not set" in str(ei.value)


async def test_fully_armed_dispatches_through_a_stubbed_exchange(monkeypatch):
    """Both arm keys + a key env + the SDK import resolving: the fully-armed
    path dispatches through the SDK's Exchange (stubbed) — the action's
    coin is resolved, the wallet is loaded from the key env via
    eth_account.Account.from_key, and the parsed response dict returns."""
    fake = _arm(monkeypatch)
    resp = await ex_resolved().sign_and_post(_order())
    assert resp == {"status": "ok", "response": {"type": "order", "data": {
        "statuses": [{"resting": {"oid": 1}}]}}}
    assert len(fake.instances) == 1
    sent = fake.instances[0]
    assert sent.wallet.key == VALID_TEST_KEY          # eth_account loaded the env key
    assert sent.base_url == "https://api.hyperliquid-testnet.xyz"
    assert sent.calls == [("order", "BTC", True, 0.1, 101.0,
                           {"limit": {"tif": "Ioc"}}, False)]


async def test_armed_cancel_dispatches_through_a_stubbed_exchange(monkeypatch):
    """A CancelAction on the fully-armed path dispatches to exchange.cancel
    (no clamp — a cancel carries no leverage/size to resolve)."""
    fake = _arm(monkeypatch)
    await ex_resolved().sign_and_post(CancelAction(coin="BTC", asset=3, oid=42))
    assert fake.instances[0].calls == [("cancel", "BTC", 42)]


async def test_armed_update_leverage_dispatches_through_a_stubbed_exchange(
        monkeypatch):
    fake = _arm(monkeypatch)
    await ex_resolved().sign_and_post(
        UpdateLeverageAction(coin="BTC", asset=3, leverage=2))
    assert fake.instances[0].calls == [("update_leverage", 2, "BTC", False)]


# --------------------------------------------------------- the market clamp


@pytest.mark.parametrize("asked,wired", [(50, 40), (41, 40), (40, 40), (10, 10)])
async def test_leverage_clamped_to_resolved_max(monkeypatch, asked, wired):
    """The market clamp: leverage above the resolved maxLeverage is clamped
    (BTC max 40 -> 50 asks 40 on the wire); at/below max passes through."""
    fake = _arm(monkeypatch)
    await ex_resolved().sign_and_post(
        UpdateLeverageAction(coin="BTC", asset=3, leverage=asked))
    assert fake.instances[0].calls == [("update_leverage", wired, "BTC", False)]


async def test_clamp_uses_each_market_own_max(monkeypatch):
    """The clamp is per-market: SOL's resolved max is 10, not BTC's 40."""
    fake = _arm(monkeypatch)
    await ex_resolved().sign_and_post(
        UpdateLeverageAction(coin="SOL", asset=0, leverage=50))
    assert fake.instances[0].calls == [("update_leverage", 10, "SOL", False)]


async def test_clamp_preserves_isolated_only(monkeypatch):
    """Isolated-only preserved: the clamp carries is_cross through unchanged
    (warden constructs False — never cross margin)."""
    fake = _arm(monkeypatch)
    await ex_resolved().sign_and_post(
        UpdateLeverageAction(coin="BTC", asset=3, leverage=50, is_cross=False))
    assert fake.instances[0].calls == [("update_leverage", 40, "BTC", False)]


async def test_clamp_is_clamp_only_no_second_action(monkeypatch):
    """Clamp-only: an order whose leverage exceeds the max is dispatched as
    ONE order call — no auto-updateLeverage is issued before it."""
    fake = _arm(monkeypatch)
    action = OrderAction(coin="BTC", asset=3, is_buy=True, size=0.1,
                         limit_px=101.0, leverage=50)
    await ex_resolved().sign_and_post(action)
    calls = fake.instances[0].calls
    assert [c[0] for c in calls] == ["order"]


async def test_order_size_and_limit_px_rounded_to_market_physics(monkeypatch):
    """OrderAction rounding: size to szDecimals decimals, limit px via the
    HL rule round(float(f"{px:.5g}"), 6 - szDecimals) — an unrounded wire
    value exchange-errs."""
    fake = _arm(monkeypatch)
    action = OrderAction(coin="BTC", asset=3, is_buy=True,
                         size=0.123456789, limit_px=75865.51234, leverage=1)
    await ex_resolved().sign_and_post(action)
    (_, _, _, wired_size, wired_px, _, _) = fake.instances[0].calls[0]
    assert wired_size == round(0.123456789, 5)              # szDecimals=5
    assert wired_px == round(float(f"{75865.51234:.5g}"), 1)  # 6 - 5
    assert wired_px == 75866.0                              # the 5-sig-fig rule, pinned literally


async def test_resolver_failure_is_fail_closed_before_any_exchange_call(
        monkeypatch):
    """The clamp is fail-closed: a resolver that raises propagates
    AssetUnresolved and NO Exchange call is recorded — nothing unclamped or
    unrounded is ever dispatched."""

    class ExplodingResolver:
        async def resolve(self, coin):
            raise AssetUnresolved(f"meta fetch failed for {coin}")

    fake = _arm(monkeypatch)
    boom = HyperliquidExecutor(env="testnet", resolver=ExplodingResolver())
    with pytest.raises(AssetUnresolved):
        await boom.sign_and_post(_order())
    assert fake.instances == []


async def test_sdk_absent_is_a_runtime_error_not_an_import_error(monkeypatch):
    """Without the `hl` extra the missing SDK surfaces as an explicit
    RuntimeError after arming — never a bare ImportError at import time."""
    monkeypatch.setenv("WARDEN_ENABLED", "1")
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", "pass")
    monkeypatch.setitem(sys.modules, "hyperliquid", None)  # import -> ImportError
    with pytest.raises(RuntimeError) as ei:
        await ex().sign_and_post(_order())
    assert "hyperliquid-python-sdk not installed" in str(ei.value)


# ------------------------------------------------------------- build_open


def test_build_open_long_math():
    a = ex().build_open("BTC", asset=0, side="LONG", size=25.0, mark=100.0,
                        leverage=1.0)
    assert type(a) is OrderAction
    assert a.size == 0.25            # NOTIONAL / mark
    assert a.limit_px == 101.0       # mark * 1.01 — aggressive IOC limit
    assert a.is_buy is True
    assert a.order_type == "ioc"
    assert a.reduce_only is False
    assert a.leverage == 1.0


def test_build_open_short_math():
    a = ex().build_open("ETH", asset=1, side="SHORT", size=25.0, mark=100.0,
                        leverage=1.0)
    assert a.size == 0.25
    assert a.limit_px == 99.0        # mark * 0.99
    assert a.is_buy is False


def test_build_open_leverage_floored_at_one():
    a = ex().build_open("BTC", asset=0, side="LONG", size=10.0, mark=50.0,
                        leverage=0.5)
    assert a.leverage == 1.0


def test_build_open_wire_shape():
    w = ex().build_open("BTC", asset=0, side="LONG", size=25.0, mark=100.0,
                        leverage=2.0).to_wire()
    assert w == {"type": "order", "coin": "BTC", "asset": 0, "is_buy": True,
                 "size": 0.25, "limit_px": 101.0, "order_type": "ioc",
                 "reduce_only": False, "leverage": 2.0}


@pytest.mark.parametrize(
    "kwargs",
    [dict(size=0.0, mark=100.0), dict(size=-1.0, mark=100.0),
     dict(size=10.0, mark=0.0), dict(size=10.0, mark=-5.0)],
)
def test_build_open_rejects_nonpositive(kwargs):
    with pytest.raises(ValueError):
        ex().build_open("BTC", asset=0, side="LONG", leverage=1.0, **kwargs)


# ------------------------------------------------------------------- misc


def test_unknown_env_valueerror():
    with pytest.raises(ValueError) as ei:
        HyperliquidExecutor(env="mainnet")
    assert "unknown env" in str(ei.value)


def test_from_env_reads_warden_env(monkeypatch):
    monkeypatch.setenv("WARDEN_ENV", "prod")
    e = HyperliquidExecutor.from_env()
    assert e.env == "prod"
    assert e.base_url == "https://api.hyperliquid.xyz"


def test_default_env_is_testnet(monkeypatch):
    monkeypatch.delenv("WARDEN_ENV", raising=False)
    e = HyperliquidExecutor.from_env()
    assert e.env == "testnet"
    assert e.base_url == "https://api.hyperliquid-testnet.xyz"
