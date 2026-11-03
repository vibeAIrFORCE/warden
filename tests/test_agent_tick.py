"""The follow loop — one tick, end to end, with fakes.

Exactly-once per signal_id, dry-by-default never posts, WARDEN_DRY=0 posts
exactly once per admitted signal, and refused decisions never reach the
executor even live. Network-free.
"""

import json

from warden.agent import Agent
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


class RecordingExecutor:
    """Duck-typed stand-in: records every build_open and every post."""

    def __init__(self):
        self.opens = []
        self.posted = []

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
