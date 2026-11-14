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
import os
import sys
from pathlib import Path
from typing import Sequence

from .assets import AssetUnresolved
from .brain import BrainAdapter, HeuristicBrain, NearBrain, NearBrainUnavailable
from .executor import HyperliquidExecutor
from .ledger import JsonlLedger
from .policy import AdmitVerdict, PolicyWallet
from .signals import SignalsUnavailable, VibeSignalProvider
from .types import Decision, Policy, PositionState, Signal

POLL_SECONDS = 60


# --- ANSI presentation layer -------------------------------------------------
# Color is presentation ONLY — no layer reads it. Hard-disabled when NO_COLOR
# is set, WARDEN_PLAIN=1, or stdout is not a TTY: piped/captured runs stay
# byte-plain, which keeps README captured-fence outputs honest and pipe-safe.
_ANSI = {
    "green": "\033[32m", "amber": "\033[33m", "red": "\033[31m",
    "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
}


def _plain() -> bool:
    if os.getenv("NO_COLOR") is not None or os.getenv("WARDEN_PLAIN") == "1":
        return True
    return not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """The one colorizer — every colorized print goes through this."""
    if _plain():
        return text
    return f"{_ANSI[code]}{text}{_ANSI['reset']}"


# --- decision-line presentation ----------------------------------------------
# tick() reports as it goes: FOLLOW lines green-bold, nearbrain vetoes and
# policy refusals print individually in dim, and pure brain-said-no skips are
# counted — a flood of them becomes one dim summary line after the pass.

def _probabilities_summary(probs: dict) -> tuple[str, float, float]:
    """(top action, its probability, top1 − top2 margin) from a parsed
    NearBrain probability dict — presentation only, never a decision."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    (top, p), (_, runner_up) = ranked[0], ranked[1]
    return top, p, p - runner_up


def _decision_line(signal: Signal, decision: Decision,
                   verdict: AdmitVerdict) -> tuple[str, str]:
    """Render one outcome's decision line.

    Returns (kind, line) where kind is "follow", "veto" (a nearbrain
    refusal — always interesting), "policy" (the brain followed and the
    policy refused) or "counted" (a pure brain-said-no skip — eligible for
    the end-of-tick counter instead of one print each).
    """
    head = f"[warden] {signal.coin} {signal.side}"
    if verdict.allowed:
        detail = decision.reason
        if decision.reason.startswith("nearbrain") and decision.probabilities:
            who = decision.reason.split(" (")[0]        # "nearbrain enter"
            top, p, _ = _probabilities_summary(decision.probabilities)
            detail = f"{who} ({top} {p:.2f})"
        size = (decision.size_usd if decision.size_usd is not None
                else verdict.effective_size_usd)
        coin = _c("bold", _c("green", signal.coin))
        line = (f"[warden] {coin} {signal.side} FOLLOW — {detail}"
                f" · ${(size or 0.0):.2f} @ {decision.leverage:g}x")
        return "follow", line
    if decision.follow:
        # the brain followed, the POLICY refused — verdict.reason names the cap
        return "policy", _c("dim", f"{head} skip — {verdict.reason}")
    if decision.reason.startswith("nearbrain"):
        # the veto itself — with parsed probabilities, show its shape
        if decision.probabilities:
            top, p, margin = _probabilities_summary(decision.probabilities)
            detail = f"nearbrain veto ({top} {p:.2f} · margin {margin:.2f})"
        else:
            detail = decision.reason
        return "veto", _c("dim", f"{head} skip — {detail}")
    return "counted", _c("dim", f"{head} skip — {decision.reason}")


def _report_outcome(signal: Signal, decision: Decision, verdict: AdmitVerdict,
                    counted: list[str]) -> None:
    """Print one decision line, or buffer it into the skip counter."""
    kind, line = _decision_line(signal, decision, verdict)
    if kind == "counted":
        counted.append(line)
        return
    print(line)


def _flush_counted_skips(counted: list[str]) -> None:
    """End-of-tick skip counter — a flood of brain-said-no skips becomes one
    dim summary; the first three still print individually."""
    if not counted:
        return
    if len(counted) > 3:
        for line in counted[:3]:
            print(line)
        print(_c("dim", f"[warden] · {len(counted)} skipped (signals checked this cycle)"))
        return
    for line in counted:
        print(line)


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
        """One pass over the open signals. Returns the per-signal outcomes.

        Decision lines print here (so the poll loop and `--once` report
        identically): FOLLOW lines, nearbrain vetoes and policy refusals
        individually, pure brain-said-no skips counted and summarized.
        """
        outcomes: list[dict] = []
        counted_skips: list[str] = []
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
                    _report_outcome(signal, decision, verdict, counted_skips)
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
            _report_outcome(signal, decision, verdict, counted_skips)
        _flush_counted_skips(counted_skips)
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


def _executor_armed() -> bool:
    import os
    return (os.getenv("WARDEN_ENABLED") == "1"
            and os.getenv("WARDEN_TESTNET_FIRE_TEST") == "pass")


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


def _print_checklist_line(line: str) -> None:
    """✓ lines green, · lines dim, → action lines amber kept dim — the
    body text is untouched; presentation only."""
    stripped = line.lstrip()
    if stripped.startswith("✓"):
        print(_c("green", line))
    elif stripped.startswith("·"):
        print(_c("dim", line))
    elif stripped.startswith("→"):
        print(_c("dim", _c("amber", line)))
    else:
        print(line)


def _print_setup_path() -> None:
    """The one-time startup checklist — what's configured, what's next.
    The flow: BRAIN first (NEAR), then SIGNALS (VIBE), then (optionally)
    the executor arm. Printed once at startup; every step is skippable —
    the agent runs degraded-but-honest without any of them."""
    import os

    print("[warden] setup checklist:")
    # 1. BRAIN — the veto decision model
    if os.getenv("WARDEN_BRAIN_URL") and os.getenv("NEAR_API_KEY"):
        _print_checklist_line(
            "  ✓ brain: NearBrain configured (WARDEN_BRAIN_URL + NEAR_API_KEY)")
    else:
        _print_checklist_line(
            "  · brain: heuristic fallback (no veto model configured)")
        _print_checklist_line("    → for the Jev veto brain, set TWO vars:")
        _print_checklist_line(
            "        WARDEN_BRAIN_URL=https://cloud-api.near.ai/v1")
        _print_checklist_line(
            "        NEAR_API_KEY=<your NEAR AI Cloud key — cloud-api.near.ai>")
    # 2. SIGNALS — where the follow list comes from
    if os.getenv("VIBE_API_KEY"):
        _print_checklist_line(
            "  ✓ signals: funded VIBE API key (per-call credits)")
    elif os.getenv("VIBE_TRIAL_TOKEN"):
        _print_checklist_line(
            "  ✓ signals: trial token (free, ~3 calls — then mint another)")
    else:
        _print_checklist_line(
            "  · signals: NOT configured — the feed will refuse (402)")
        _print_checklist_line("    → free trial (no account):")
        _print_checklist_line(
            "        curl -s -X POST https://api.vibe.airforce/api/vibe-tools/trial-auth")
        _print_checklist_line(
            "      then: export VIBE_TRIAL_TOKEN=trial_xxxxxxxxxx   # the WHOLE token")
        _print_checklist_line(
            "    → or a funded key: https://www.vibe.airforce/developers")
    # 3. EXECUTOR — real orders (optional; dry is the default)
    if _executor_armed():
        _print_checklist_line(
            "  ✓ executor: ARMED (real orders when WARDEN_DRY=0)")
    else:
        _print_checklist_line(
            "  · executor: inert (dry-only) — arm later, see README 'Arm the executor'")
    print()


def _print_banner() -> None:
    """The identity line — bold version + dim mode tagline, one compact
    line, printed first (before the DRY banner)."""
    from . import __version__
    mode = ("dry mode: decisions print, nothing sends" if _dry()
            else "live mode: real orders")
    print(_c("bold", f"WARDEN v{__version__} — policy-capped trading agent")
          + _c("dim", f"  brain → policy → executor · {mode}"))


def _read_ledger_rows(path: Path) -> list[dict]:
    """Read-only ledger scan for the status screen — never writes."""
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _ledger_row_line(row: dict) -> str:
    """One compact ledger row for the status screen."""
    ts = str(row.get("ts", "?"))[:19]
    word = "FOLLOW" if row.get("decision") is True else "skip"
    if row.get("executed") is True:
        state = "executed"
    elif row.get("dry") is True:
        state = "dry"
    else:
        state = "not-executed"
    return (f"[warden]   {ts} {row.get('coin', '?')} {row.get('side', '?')}"
            f" {word} {state}")


def _print_status() -> None:
    """The `status` mode — one read-only screen (checklist, ledger stats,
    executor arm state), no polling, exit 0."""
    ledger = JsonlLedger()
    rows = _read_ledger_rows(ledger._path)   # the ledger's own path, read-only
    follows = sum(1 for r in rows if r.get("decision") is True)
    print(_c("dim", f"[warden] ledger: {len(rows)} rows · {follows} FOLLOW"
                    f" — {ledger._path}"))
    for row in rows[-3:]:
        print(_c("dim", _ledger_row_line(row)))
    if not rows:
        print(_c("dim", "[warden]   (no rows yet — the ledger fills as signals arrive)"))
    if _executor_armed():
        print(_c("green", "[warden] executor arm: ARMED (real orders when WARDEN_DRY=0)"))
    else:
        print(_c("dim", "[warden] executor arm: inert (dry-only)"))


async def _main() -> None:
    ap = argparse.ArgumentParser(prog="warden")
    ap.add_argument("mode", nargs="?", choices=("dry", "status"), default="dry",
                    help="dry = follow loop (default) · "
                         "status = one read-only screen, no polling")
    ap.add_argument("--dry", action="store_true", default=True)
    ap.add_argument("--once", action="store_true",
                    help="run exactly one tick, then exit 0")
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
    _print_banner()
    if args.mode == "status":
        _print_setup_path()
        _print_status()
        return
    if _dry():
        print("[warden] DRY — decisions print, nothing sends")
    _print_setup_path()
    if args.once:
        try:
            await agent.tick()
        except SignalsUnavailable as exc:
            # one tick, the same guided honesty as the loop
            _guided_unavailable(exc.reason)
        return
    while True:
        try:
            await agent.tick()
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
