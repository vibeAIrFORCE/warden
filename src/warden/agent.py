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

from .assets import AssetUnresolved
from .brain import BrainAdapter, HeuristicBrain, NearBrain, NearBrainUnavailable
from .executor import HyperliquidExecutor
from .ledger import JsonlLedger
from .policy import PolicyWallet
from .signals import SignalsUnavailable, VibeSignalProvider
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
                try:
                    meta = await self.executor.resolve_asset(signal.coin)
                except AssetUnresolved:
                    # Fail closed in dry runs too: an unresolvable coin
                    # never becomes an action (the silent-asset-0 class is
                    # dead everywhere). The row still records — a skipped
                    # signal_id is not retried — and the brain's decision
                    # reason keeps its own key; the asset failure names
                    # itself in exec_reason.
                    self.ledger.record({
                        "signal_id": signal.signal_id, "coin": signal.coin,
                        "side": signal.side, "decision": decision.follow,
                        "reason": decision.reason,
                        "admitted": verdict.allowed,
                        "admit_reason": verdict.reason, "executed": False,
                        "dry": _dry(), "exec_reason": "asset_unresolved",
                    })
                    outcomes.append({"signal": signal, "verdict": verdict})
                    continue
                action = self.executor.build_open(
                    signal.coin, asset=meta.asset, side=signal.side,
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


def _guided_unavailable(reason: str) -> None:
    """Print the caller-actionable next steps for a priced-feed failure.

    SignalsUnavailable is honesty, not a bug — the feed is pay/trial-gated
    and THIS caller's supply failed. Present it as guidance and keep the
    loop alive (the next cycle succeeds once the env var is set).
    """
    if reason == "http_402":
        head = "→ mint a FREE trial token (no account needed):"
        extra = "  → or get a funded key: https://www.vibe.airforce/developers"
    elif reason == "http_429":
        head = "→ trial budget exhausted — mint a fresh token (they're capped):"
        extra = "  → or get a funded key: https://www.vibe.airforce/developers"
    elif reason == "parse_dropped_all_rows":
        head = "→ payload shape drifted — report this: https://github.com/vibeAIrFORCE/warden"
        extra = ""
    else:
        head = "→ check your credentials / try again shortly"
        extra = "  → or get a funded key: https://www.vibe.airforce/developers"
    print(f"[warden] signal feed unavailable ({reason})", file=sys.stderr)
    print(f"  {head}", file=sys.stderr)
    if reason in ("http_402", "http_429"):
        print("      curl -s -X POST https://api.vibe.airforce/api/vibe-tools/trial-auth",
              file=sys.stderr)
        print("  → then (copy the WHOLE token — no dots):", file=sys.stderr)
        print("      export VIBE_TRIAL_TOKEN=trial_xxxxxxxxxx", file=sys.stderr)
        print("      uv run python -m warden.agent --dry", file=sys.stderr)
    if extra:
        print(extra, file=sys.stderr)


def _print_setup_path() -> None:
    """The one-time startup checklist — what's configured, what's next.
    The flow: BRAIN first (NEAR), then SIGNALS (VIBE), then (optionally)
    the executor arm. Printed once at startup; every step is skippable —
    the agent runs degraded-but-honest without any of them."""
    import os

    print("[warden] setup checklist:")
    # 1. BRAIN — the veto decision model
    if os.getenv("WARDEN_BRAIN_URL") and os.getenv("NEAR_API_KEY"):
        print("  ✓ brain: NearBrain configured (WARDEN_BRAIN_URL + NEAR_API_KEY)")
    else:
        print("  · brain: heuristic fallback (no veto model configured)")
        print("    → for the Jev veto brain, set TWO vars:")
        print("        WARDEN_BRAIN_URL=https://cloud-api.near.ai/v1")
        print("        NEAR_API_KEY=<your NEAR AI Cloud key — cloud-api.near.ai>")
    # 2. SIGNALS — where the follow list comes from
    if os.getenv("VIBE_API_KEY"):
        print("  ✓ signals: funded VIBE API key (per-call credits)")
    elif os.getenv("VIBE_TRIAL_TOKEN"):
        print("  ✓ signals: trial token (free, ~3 calls — then mint another)")
    else:
        print("  · signals: NOT configured — the feed will refuse (402)")
        print("    → free trial (no account):")
        print("        curl -s -X POST https://api.vibe.airforce/api/vibe-tools/trial-auth")
        print("      then: export VIBE_TRIAL_TOKEN=trial_xxxxxxxxxx   # the WHOLE token")
        print("    → or a funded key: https://www.vibe.airforce/developers")
    # 3. EXECUTOR — real orders (optional; dry is the default)
    if os.getenv("WARDEN_ENABLED") == "1" and os.getenv("WARDEN_TESTNET_FIRE_TEST") == "pass":
        print("  ✓ executor: ARMED (real orders when WARDEN_DRY=0)")
    else:
        print("  · executor: inert (dry-only) — arm later, see README 'Arm the executor'")
    print()


async def _main() -> None:
    ap = argparse.ArgumentParser(prog="warden")
    ap.add_argument("--dry", action="store_true", default=True)
    ap.add_argument("--min-conviction", type=float, default=60.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    args = ap.parse_args()

    # Construction-time brain selection ONLY: the remote veto brain first
    # (NearBrain, configured via WARDEN_BRAIN_URL), falling back to the
    # heuristic when no endpoint is configured. A NearBrain runtime failure
    # refuses that one signal — the brain never swaps mid-flight, which
    # would mask outages and break the ledger's decision.source honesty.
    # --min-conviction stays HeuristicBrain-only: NearBrain's confidence
    # mechanism is the parsed probability margin, not a conviction floor.
    try:
        brain: BrainAdapter = NearBrain(size_usd=args.size)
    except NearBrainUnavailable:
        brain = HeuristicBrain(min_conviction=args.min_conviction,
                               size_usd=args.size)
    agent = Agent(
        signals=VibeSignalProvider(),
        brain=brain,
        policy=PolicyWallet(Policy()),
    )
    if _dry():
        print("[warden] DRY — decisions print, nothing sends")
    _print_setup_path()
    while True:
        try:
            for out in await agent.tick():
                v = out["verdict"]
                s = out["signal"]
                print(f"[warden] {s.coin} {s.side} "
                      f"{'FOLLOW' if v.allowed else 'skip'} — {v.reason}")
        except SignalsUnavailable as exc:
            # Guided presentation only — every other exception still
            # propagates loudly (a real bug must crash). The loop keeps
            # polling: set the env var and the next cycle succeeds.
            _guided_unavailable(exc.reason)
        await asyncio.sleep(args.poll)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        # Ctrl-C is the DESIGNED stop for an infinite poll loop — exit
        # silently (exit code 130 = SIGINT), never a CancelledError dump.
        raise SystemExit(130)
