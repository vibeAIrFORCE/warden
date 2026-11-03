"""The follow loop — signals in, policy-checked orders out.

Exactly-once per signal_id (the agent never re-follows a delivery it has
already acted on), the brain proposes, the policy disposes, the executor
executes — in that order, always. Dry mode is the default: the loop runs
end-to-end and prints every decision without sending anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

from .brain import BrainAdapter, HeuristicBrain
from .executor import HyperliquidExecutor
from .ledger import JsonlLedger
from .policy import PolicyWallet
from .signals import VibeSignalProvider
from .types import Policy, PositionState

POLL_SECONDS = 60


class Agent:
    def __init__(self, *, signals, brain: BrainAdapter, policy: PolicyWallet,
                 executor: HyperliquidExecutor | None = None,
                 ledger: JsonlLedger | None = None,
                 positions: PositionState | None = None) -> None:
        self.signals = signals
        self.brain = brain
        self.policy = policy
        self.executor = executor
        self.ledger = ledger or JsonlLedger()
        self.state = positions or PositionState()

    async def tick(self) -> list[dict]:
        """One pass over the open signals. Returns the per-signal outcomes."""
        outcomes: list[dict] = []
        for signal in await self.signals.latest():
            if self.ledger.seen(signal.signal_id):
                continue
            decision = await self.brain.decide(signal, self.state)
            verdict = await self.policy.admit(signal, decision, self.state)
            executed = False
            if verdict.allowed and self.executor is not None:
                # mark: in live mode the mark comes from the exchange's
                # allMids; dry mode uses the signal's own recorded basis —
                # the honest mirror-the-book entry.
                mark = signal.entry_price or 0.0
                action = self.executor.build_open(
                    signal.coin, asset=0, side=signal.side,
                    size=verdict.effective_size_usd or 0.0, mark=mark,
                    leverage=verdict.effective_leverage,
                )
                if not _dry():
                    await self.executor.sign_and_post(action)
                    executed = True
            self.ledger.record({
                "signal_id": signal.signal_id, "coin": signal.coin,
                "side": signal.side, "decision": decision.follow,
                "reason": decision.reason, "admitted": verdict.allowed,
                "admit_reason": verdict.reason, "executed": executed,
                "dry": _dry(),
            })
            outcomes.append({"signal": signal, "verdict": verdict})
        return outcomes

    async def run(self, poll_seconds: int = POLL_SECONDS) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:  # the loop never dies on one bad cycle
                print(f"[warden] cycle error: {exc}", file=sys.stderr)
            await asyncio.sleep(poll_seconds)


def _dry() -> bool:
    import os
    return os.getenv("WARDEN_DRY", "1") not in ("0", "false", "off")


async def _main() -> None:
    ap = argparse.ArgumentParser(prog="warden")
    ap.add_argument("--dry", action="store_true", default=True)
    ap.add_argument("--min-conviction", type=float, default=60.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    args = ap.parse_args()

    agent = Agent(
        signals=VibeSignalProvider(),
        brain=HeuristicBrain(min_conviction=args.min_conviction,
                             size_usd=args.size),
        policy=PolicyWallet(Policy()),
    )
    if _dry():
        print("[warden] DRY — decisions print, nothing sends")
    while True:
        for out in await agent.tick():
            v = out["verdict"]
            s = out["signal"]
            print(f"[warden] {s.coin} {s.side} "
                  f"{'FOLLOW' if v.allowed else 'skip'} — {v.reason}")
        await asyncio.sleep(args.poll)


if __name__ == "__main__":
    asyncio.run(_main())
