"""The follow loop — one tick, end to end, with fakes.

Exactly-once per signal_id, dry-by-default never posts, WARDEN_DRY=0 posts
exactly once per admitted signal, and refused decisions never reach the
executor even live. Network-free.
"""

import json

from warden.agent import (
    Agent,
    _load_signal_preferences,
    _pref_side_param,
    _pref_tokens_param,
)
from warden.assets import AssetMeta, MappingAssetResolver
from warden.brain import HeuristicBrain
from warden.executor import OrderAction
from warden.ledger import JsonlLedger
from warden.policy import PolicyWallet
from warden.types import Decision, Policy, Signal


class ListProvider:
    def __init__(self, signals):
        self.signals = list(signals)

    async def latest(self):
        return list(self.signals)


# The universe slice the recording double resolves against — same shape the
# live resolver produces (array-index asset ids, testnet values pinned in
# tests/test_asset_resolve.py). Any coin outside it raises AssetUnresolved,
# which is exactly what the agent's fail-closed path must handle.
_EXECUTOR_ASSET_MAP = {
    "BTC": AssetMeta(asset=3, max_leverage=40, sz_decimals=5),
    "ETH": AssetMeta(asset=4, max_leverage=25, sz_decimals=4),
    "SOL": AssetMeta(asset=0, max_leverage=10, sz_decimals=2),
}


class RecordingExecutor:
    """Duck-typed stand-in: records every build_open and every post."""

    def __init__(self):
        self.opens = []
        self.posted = []
        self._resolver = MappingAssetResolver(_EXECUTOR_ASSET_MAP)

    async def resolve_asset(self, coin):
        return await self._resolver.resolve(coin)

    def build_open(self, coin, asset, side, size, mark, leverage):
        action = OrderAction(coin=coin, asset=asset,
                             is_buy=(side == "LONG"), size=size,
                             limit_px=mark, leverage=leverage)
        self.opens.append(action)
        return action

    async def sign_and_post(self, action):
        self.posted.append(action)
        return {"status": "filled"}


def sig(sid, coin="BTC", side="LONG", entry=100.0, conviction=80.0) -> Signal:
    return Signal(signal_id=sid, coin=coin, side=side,
                  entry_price=entry, conviction=conviction)


def agent(signals, executor=None, ledger=None) -> Agent:
    return Agent(
        signals=signals,
        brain=HeuristicBrain(min_conviction=60.0, size_usd=10.0),
        policy=PolicyWallet(Policy(max_trade_usd=25, daily_budget_usd=50)),
        executor=executor,
        ledger=ledger,
    )


def ledger_rows(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


async def test_dry_default_never_posts_and_marks_executed_false(
        tmp_path, monkeypatch):
    monkeypatch.delenv("WARDEN_DRY", raising=False)   # the default IS dry
    ex = RecordingExecutor()
    a = agent(ListProvider([sig("s1")]), executor=ex,
              ledger=JsonlLedger(tmp_path / "ledger.jsonl"))

    outs = await a.tick()

    assert len(outs) == 1
    assert outs[0]["verdict"].allowed is True
    assert ex.posted == []                       # nothing sent
    assert len(ex.opens) == 1                    # the action IS constructed
    assert ex.opens[0].coin == "BTC"
    assert ex.opens[0].size == 10.0              # the policy's effective size
    assert ex.opens[0].leverage == 1.0           # the policy's effective lev
    rows = ledger_rows(tmp_path / "ledger.jsonl")
    assert len(rows) == 1
    assert rows[0]["executed"] is False
    assert rows[0]["dry"] is True
    assert rows[0]["admitted"] is True


async def test_warden_dry_zero_posts_once_per_admitted_signal(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WARDEN_DRY", "0")
    ex = RecordingExecutor()
    a = agent(ListProvider([
        sig("s1", coin="BTC", entry=100.0),
        sig("s2", coin="ETH", side="SHORT", entry=50.0),
        sig("s3", coin="SOL", entry=10.0, conviction=30.0),   # brain refuses
    ]), executor=ex, ledger=JsonlLedger(tmp_path / "ledger.jsonl"))

    outs = await a.tick()

    assert len(outs) == 3
    assert len(ex.posted) == 2                   # once per ADMITTED signal
    assert ex.posted[0].coin == "BTC" and ex.posted[0].is_buy is True
    assert ex.posted[1].coin == "ETH" and ex.posted[1].is_buy is False
    rows = ledger_rows(tmp_path / "ledger.jsonl")
    assert len(rows) == 3
    by_sid = {r["signal_id"]: r for r in rows}
    assert by_sid["s1"]["executed"] is True and by_sid["s1"]["dry"] is False
    assert by_sid["s2"]["executed"] is True
    assert by_sid["s3"]["decision"] is False
    assert by_sid["s3"]["admitted"] is False
    assert by_sid["s3"]["executed"] is False


async def test_exactly_once_second_tick_empty_and_no_double_post(
        tmp_path, monkeypatch):
    """The live-mode exactly-once: the second tick over the SAME feed does
    nothing — no outcome, no post, no ledger row."""
    monkeypatch.setenv("WARDEN_DRY", "0")
    ex = RecordingExecutor()
    path = tmp_path / "ledger.jsonl"
    a = agent(ListProvider([sig("s1"), sig("s2", coin="ETH")]),
              executor=ex, ledger=JsonlLedger(path))

    first = await a.tick()
    second = await a.tick()

    assert len(first) == 2
    assert second == []
    assert len(ex.posted) == 2                   # first tick only
    assert len(ex.opens) == 2
    assert len(ledger_rows(path)) == 2           # exactly-once, on the wire


async def test_oversize_refusal_never_reaches_executor_even_live(
        tmp_path, monkeypatch):
    monkeypatch.setenv("WARDEN_DRY", "0")
    ex = RecordingExecutor()

    class BigBrain:
        async def decide(self, signal, state):
            return Decision(follow=True, size_usd=30.0, reason="big",
                            source="test")

    a = Agent(signals=ListProvider([sig("s1")]), brain=BigBrain(),
              policy=PolicyWallet(Policy(max_trade_usd=25,
                                         daily_budget_usd=50)),
              executor=ex, ledger=JsonlLedger(tmp_path / "ledger.jsonl"))

    outs = await a.tick()

    assert outs[0]["verdict"].allowed is False
    assert "max_trade_usd" in outs[0]["verdict"].reason
    assert ex.opens == [] and ex.posted == []
    rows = ledger_rows(tmp_path / "ledger.jsonl")
    assert rows[0]["admitted"] is False and rows[0]["executed"] is False


async def test_no_executor_still_records_and_never_crashes(
        tmp_path, monkeypatch):
    """The observation-only configuration (the CLI default wires no
    executor): the loop records decisions and never sends."""
    monkeypatch.setenv("WARDEN_DRY", "0")        # live loop, no executor wired
    path = tmp_path / "ledger.jsonl"
    a = agent(ListProvider([sig("s1")]), executor=None, ledger=JsonlLedger(path))

    outs = await a.tick()

    assert outs[0]["verdict"].allowed is True
    rows = ledger_rows(path)
    assert rows[0]["executed"] is False


# ------------------------------------------------ your preferences (env)


def _prefs_env(monkeypatch, **vals) -> None:
    """Set the three preference vars per kwargs; clear the rest."""
    for var in ("WARDEN_TICKERS", "WARDEN_MIN_CONVICTION", "WARDEN_SIDES"):
        if var in vals:
            monkeypatch.setenv(var, vals[var])
        else:
            monkeypatch.delenv(var, raising=False)


def test_prefs_absent_means_no_preference(monkeypatch):
    """All three vars absent → every field None (the unfiltered agent)."""
    _prefs_env(monkeypatch)
    prefs = _load_signal_preferences()
    assert prefs.tickers is None
    assert prefs.min_conviction is None
    assert prefs.sides is None


def test_prefs_valid_values_parse(monkeypatch):
    _prefs_env(monkeypatch, WARDEN_TICKERS="USELESS,BTC",
               WARDEN_MIN_CONVICTION="60", WARDEN_SIDES="SHORT")
    prefs = _load_signal_preferences()
    assert prefs.tickers == frozenset({"USELESS", "BTC"})
    assert prefs.min_conviction == 60.0
    assert prefs.sides == frozenset({"SHORT"})


def test_prefs_tickers_uppercase_normalization(monkeypatch):
    """Lowercase env input normalizes to the book's uppercase symbols."""
    _prefs_env(monkeypatch, WARDEN_TICKERS="useless,btc")
    assert _load_signal_preferences().tickers == frozenset({"USELESS", "BTC"})


def test_prefs_invalid_conviction_fails_closed_with_one_warning(
        monkeypatch, capsys):
    _prefs_env(monkeypatch, WARDEN_MIN_CONVICTION="not-a-number")
    prefs = _load_signal_preferences()
    assert prefs.min_conviction is None
    err = capsys.readouterr().err
    assert "WARDEN_MIN_CONVICTION" in err
    assert len(err.strip().splitlines()) == 1      # ONE warning line


def test_prefs_conviction_bounds(monkeypatch, capsys):
    """0 is a valid floor; anything outside [0, 100] fails closed + warns."""
    _prefs_env(monkeypatch, WARDEN_MIN_CONVICTION="0")
    assert _load_signal_preferences().min_conviction == 0.0
    _prefs_env(monkeypatch, WARDEN_MIN_CONVICTION="150")
    prefs = _load_signal_preferences()
    assert prefs.min_conviction is None
    assert "WARDEN_MIN_CONVICTION" in capsys.readouterr().err


def test_prefs_sides_invalid_token_ignored(monkeypatch, capsys):
    """A bad token among good ones is dropped (var named in ONE warning),
    the valid rest still applies."""
    _prefs_env(monkeypatch, WARDEN_SIDES="SHORT,FLAT")
    prefs = _load_signal_preferences()
    assert prefs.sides == frozenset({"SHORT"})
    err = capsys.readouterr().err
    assert "WARDEN_SIDES" in err
    assert len(err.strip().splitlines()) == 1


def test_prefs_sides_all_invalid_fails_closed(monkeypatch, capsys):
    _prefs_env(monkeypatch, WARDEN_SIDES="FLAT")
    prefs = _load_signal_preferences()
    assert prefs.sides is None
    assert "WARDEN_SIDES" in capsys.readouterr().err


def test_prefs_side_and_tokens_param_conversion():
    """frozenset → the SKU's wire shapes: one side (both = no filter) and a
    sorted comma-joined tokens string (byte-stable query)."""
    assert _pref_side_param(frozenset({"SHORT"})) == "SHORT"
    assert _pref_side_param(frozenset({"LONG"})) == "LONG"
    assert _pref_side_param(frozenset({"LONG", "SHORT"})) is None
    assert _pref_side_param(None) is None
    assert _pref_tokens_param(frozenset({"USELESS", "BTC"})) == "BTC,USELESS"
    assert _pref_tokens_param(None) is None


async def test_prefs_policy_allowed_coins_refusal_still_fires(
        monkeypatch, tmp_path):
    """The double gate's local half: WARDEN_TICKERS → Policy.allowed_coins.
    A coin outside the allowlist is refused at admit time even when the
    feed delivers it (server filter bypassed or drifted) — and every other
    Policy field stays at its default."""
    _prefs_env(monkeypatch, WARDEN_TICKERS="BTC")
    prefs = _load_signal_preferences()

    class AllowAllBrain:
        async def decide(self, signal, state):
            return Decision(follow=True, size_usd=10.0, reason="wants in",
                            source="test")

    a = Agent(
        signals=ListProvider([sig("s1", coin="USELESS")]),
        brain=AllowAllBrain(),
        policy=PolicyWallet(Policy(allowed_coins=prefs.tickers)),
        ledger=JsonlLedger(tmp_path / "ledger.jsonl"),
    )

    outs = await a.tick()

    assert outs[0]["verdict"].allowed is False
    assert "not on the coin allowlist" in outs[0]["verdict"].reason
    assert ledger_rows(tmp_path / "ledger.jsonl")[0]["admitted"] is False
