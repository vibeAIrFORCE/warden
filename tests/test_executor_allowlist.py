"""The executor's teeth — the typed-action allowlist and the two-key arm.

Proves, in order:
  * a raw dict can never reach signing — ActionNotAllowed fires BEFORE any
    env read (proven on an unarmed executor: if the arm gate, which reads
    env, ran first, we would see NotArmed instead);
  * the arm gate is a two-key AND, re-checked every call — full matrix;
  * build_open's sizing/limit math is exact (notional / mark, mark ±1%);
  * the allowlist is a TYPE check: UpdateLeverage/Cancel pass isinstance and
    die at the arm gate instead — asserted by exception TYPE.

Network-free: the optional SDK import is stubbed, no extra needed.
"""

import sys
import types

import pytest

from warden.executor import (
    ActionNotAllowed,
    CancelAction,
    HyperliquidExecutor,
    NotArmed,
    OrderAction,
    UpdateLeverageAction,
)


def ex(env: str = "testnet") -> HyperliquidExecutor:
    return HyperliquidExecutor(env=env)


def _order() -> OrderAction:
    return OrderAction(coin="BTC", asset=0, is_buy=True, size=0.1,
                       limit_px=101.0)


def _install_fake_sdk(monkeypatch) -> None:
    """hyperliquid-python-sdk is an optional extra — stub the three imports
    sign_and_post makes so the full arming chain is testable network-free."""
    hl = types.ModuleType("hyperliquid")
    info = types.ModuleType("hyperliquid.info")
    info.Info = object
    exch = types.ModuleType("hyperliquid.exchange")
    exch.Exchange = object
    utils = types.ModuleType("hyperliquid.utils")
    signing = types.ModuleType("hyperliquid.utils.signing")
    signing.sign_l1_action = lambda *a, **k: b""
    hl.info, hl.exchange, hl.utils = info, exch, utils
    for name, mod in [
        ("hyperliquid", hl), ("hyperliquid.info", info),
        ("hyperliquid.exchange", exch), ("hyperliquid.utils", utils),
        ("hyperliquid.utils.signing", signing),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


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


async def test_fully_armed_reaches_the_live_ship_boundary(monkeypatch):
    """Both arm keys + a key env + the SDK import resolving: the only thing
    left is the not-yet-shipped signing path — still a hard stop."""
    monkeypatch.setenv("WARDEN_ENABLED", "1")
    monkeypatch.setenv("WARDEN_TESTNET_FIRE_TEST", "pass")
    monkeypatch.setenv("WARDEN_PRIVATE_KEY", "0xdeadbeef")
    _install_fake_sdk(monkeypatch)
    with pytest.raises(RuntimeError) as ei:
        await ex().sign_and_post(_order())
    assert "live signing ships with the fire-test" in str(ei.value)


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
