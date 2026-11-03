"""The teeth battery — every refusal branch in PolicyWallet.admit, in the
fixed audit order, plus the leverage CLAMP and the happy path.

Network-free: pure dataclasses, no I/O.
"""

from warden.policy import PolicyWallet
from warden.types import Decision, Policy, PositionState, Signal


def sig(coin="BTC", side="LONG", conviction=70.0, entry_price=100.0):
    return Signal(
        signal_id=f"{coin}-{side}-1", coin=coin, side=side,
        entry_price=entry_price, conviction=conviction,
    )


def dec(follow=True, size=10.0, leverage=1.0, reason="ok"):
    return Decision(follow=follow, size_usd=size, leverage=leverage, reason=reason)


def state(open_positions=0, deployed=0.0, spent_today=0.0, held=()):
    return PositionState(
        open_positions=open_positions, deployed_usd=deployed,
        spent_today_usd=spent_today, held_coins=frozenset(held),
    )


def wallet(**overrides) -> PolicyWallet:
    base = dict(
        max_trade_usd=25.0, max_total_deployed_usd=100.0,
        max_open_positions=5, daily_budget_usd=50.0, max_leverage=2.0,
    )
    base.update(overrides)
    return PolicyWallet(Policy(**base))


async def test_happy_path_admitted():
    v = await wallet().admit(sig(), dec(size=10.0), state())
    assert v.allowed is True
    assert v.reason == "admitted"
    assert v.effective_size_usd == 10.0
    assert v.effective_leverage == 1.0


async def test_refuse_brain_said_no():
    v = await wallet().admit(sig(), dec(follow=False, size=None), state())
    assert v.allowed is False
    assert v.reason == "brain said no"


async def test_refuse_no_size():
    v = await wallet().admit(sig(), dec(size=None), state())
    assert v.allowed is False
    assert v.reason == "brain proposed no size"


async def test_refuse_zero_size():
    v = await wallet().admit(sig(), dec(size=0.0), state())
    assert v.allowed is False
    assert v.reason == "brain proposed no size"


async def test_refuse_held_coin():
    v = await wallet().admit(sig(coin="BTC"), dec(size=10.0),
                             state(held=("BTC",)))
    assert v.allowed is False
    assert "already holding BTC" in v.reason


async def test_refuse_not_on_allowlist():
    w = wallet(allowed_coins=frozenset({"BTC", "ETH"}))
    v = await w.admit(sig(coin="SOL"), dec(size=10.0), state())
    assert v.allowed is False
    assert "SOL not on the coin allowlist" in v.reason


async def test_allowlist_none_means_any_coin():
    v = await wallet().admit(sig(coin="DOGE"), dec(size=10.0), state())
    assert v.allowed is True


async def test_refuse_size_over_cap():
    v = await wallet().admit(sig(), dec(size=30.0), state())
    assert v.allowed is False
    assert "$30.00 > max_trade_usd $25.00" in v.reason


async def test_size_at_cap_is_allowed():
    v = await wallet().admit(sig(), dec(size=25.0), state())
    assert v.allowed is True


async def test_leverage_clamped_not_refused():
    v = await wallet().admit(sig(), dec(size=10.0, leverage=9.0), state())
    assert v.allowed is True
    assert v.effective_leverage == 2.0
    assert v.effective_size_usd == 10.0


async def test_leverage_below_one_clamped_to_one():
    v = await wallet().admit(sig(), dec(size=10.0, leverage=0.0), state())
    assert v.allowed is True
    assert v.effective_leverage == 1.0


async def test_refuse_concurrency_cap():
    v = await wallet().admit(sig(), dec(size=10.0),
                             state(open_positions=5))
    assert v.allowed is False
    assert "open positions 5 >= cap 5" in v.reason


async def test_refuse_deployed_cap():
    v = await wallet().admit(sig(), dec(size=10.0),
                             state(deployed=95.0))
    assert v.allowed is False
    assert "deployed $95.00 + $10.00 > cap $100.00" in v.reason


async def test_deployed_exactly_at_cap_is_allowed():
    v = await wallet().admit(sig(), dec(size=10.0),
                             state(deployed=90.0))
    assert v.allowed is True


async def test_refuse_daily_fuse():
    v = await wallet().admit(sig(), dec(size=10.0),
                             state(spent_today=45.0))
    assert v.allowed is False
    assert "daily fuse" in v.reason
    assert "$45.00 spent + $10.00" in v.reason


async def test_daily_fuse_exactly_at_budget_is_allowed():
    v = await wallet().admit(sig(), dec(size=10.0),
                             state(spent_today=40.0))
    assert v.allowed is True


async def test_held_coin_beats_size_cap_first_refusal_wins():
    """Pins the audit order: held-coin (coin stage) fires before the size
    stage — first refusal in the fixed order wins."""
    w = wallet(max_trade_usd=25.0)
    v = await w.admit(sig(coin="BTC"), dec(size=999.0), state(held=("BTC",)))
    assert v.allowed is False
    assert "already holding BTC" in v.reason
