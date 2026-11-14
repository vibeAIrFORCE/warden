"""The follow loop — signals in, policy-checked orders out.

Exactly-once per signal_id (the agent never re-follows a delivery it has
already acted on), the brain proposes, the policy disposes, the executor
executes — in that order, always. Dry mode is the default: the loop runs
end-to-end and prints every decision without sending anything.

`--setup` is the operator on-ramp: it collects and LIVE-verifies the brain
key and the trial-signal token, then prints a masked summary — every step
Enter-skippable. Without a TTY (or with --non-interactive) it degrades to
the old fire-and-forget token mint, so pipes and CI never block.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from .assets import AssetUnresolved
from .brain import BrainAdapter, HeuristicBrain, NearBrain, NearBrainUnavailable
from .executor import HyperliquidExecutor
from .ledger import JsonlLedger
from .policy import AdmitVerdict, PolicyWallet
from .signals import SignalsUnavailable, VibeSignalProvider
from .types import Decision, Policy, PositionState, Signal, SignalPreferences

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


_brain_401_printed = False   # module latch: the 401 hint prints ONCE


def _brain_401_hint(decision: Decision) -> None:
    """Nearbrain refused with an http 401 — the operator key itself was
    REJECTED (not absent, not rate-limited). Say so once, plainly, with
    where to fix it; identical vetoes afterwards stay silent."""
    global _brain_401_printed
    if _brain_401_printed or "401" not in decision.reason:
        return
    _brain_401_printed = True
    print(_c("dim", "[warden] NEAR_API_KEY was rejected (401) — check it at "
                    "cloud-api.near.ai"), file=sys.stderr)


def _report_outcome(signal: Signal, decision: Decision, verdict: AdmitVerdict,
                    counted: list[str]) -> None:
    """Print one decision line, or buffer it into the skip counter."""
    _brain_401_hint(decision)
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
                 positions: PositionState | None = None,
                 prefs: SignalPreferences | None = None) -> None:
        self.signals = signals
        self.brain = brain
        self.policy = policy
        self.executor = executor
        self.ledger = ledger or JsonlLedger()
        self.state = positions or PositionState()
        self._prefs = prefs or SignalPreferences()

    def _book_counts(self, fetched: list[Signal]) -> tuple[int, int, int]:
        """The funnel the status line reports, computed honestly:
        (fetched, matching prefs locally approximated, above the effective
        floor). The floor is the operator's pref when set (the server
        already filtered on it — arrivals below it are drift), else the
        heuristic brain's floor when that's the brain; no floor → every
        matched signal counts."""
        prefs = self._prefs
        matched = [s for s in fetched
                   if (prefs.tickers is None or s.coin in prefs.tickers)
                   and (prefs.sides is None or s.side in prefs.sides)]
        floor = prefs.min_conviction
        if floor is None:
            floor = getattr(self.brain, "min_conviction", None)
            if not isinstance(floor, (int, float)):
                floor = None
        if floor is None:
            above = matched
        else:
            above = [s for s in matched
                     if s.conviction is not None and s.conviction >= floor]
        return len(fetched), len(matched), len(above)

    async def tick(self) -> list[dict]:
        """One pass over the open signals. Returns the per-signal outcomes.

        Decision lines print here (so the poll loop and `--once` report
        identically): FOLLOW lines, nearbrain vetoes and policy refusals
        individually, pure brain-said-no skips counted and summarized —
        then ONE dim book-status line (always, even on an empty book).
        """
        outcomes: list[dict] = []
        counted_skips: list[str] = []
        fetched = list(await self.signals.latest())
        for signal in fetched:
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
        new_count = len(outcomes)
        fetched_n, matched_n, above_n = self._book_counts(fetched)
        print(_c("dim", f"[warden] book: {fetched_n} open · {matched_n} match "
                        f"prefs · {above_n} above floor · {new_count} new"))
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


# --- your preferences (env-driven, all optional) ------------------------------
# WARDEN_TICKERS / WARDEN_MIN_CONVICTION / WARDEN_SIDES narrow the book to
# the operator's slice. Absent = no preference (None). INVALID values fail
# closed to absent with ONE warning line naming the var — never a crash,
# never a half-parsed preference.

_PREF_SIDES = ("LONG", "SHORT")


def _warn_pref(var: str, raw: str, hint: str) -> None:
    print(f"[warden] {var}={raw!r} ignored — {hint}", file=sys.stderr)


def _pref_tickers(raw: str | None) -> frozenset[str] | None:
    """WARDEN_TICKERS — comma-separated coin symbols, uppercase-normalized."""
    if raw is None or not raw.strip():
        return None
    tickers = frozenset(t.strip().upper() for t in raw.split(",") if t.strip())
    if not tickers:
        _warn_pref("WARDEN_TICKERS", raw, "no coin symbols found")
        return None
    return tickers


def _pref_min_conviction(raw: str | None) -> float | None:
    """WARDEN_MIN_CONVICTION — a float in [0, 100]."""
    if raw is None or not raw.strip():
        return None
    try:
        v = float(raw)
    except ValueError:
        _warn_pref("WARDEN_MIN_CONVICTION", raw, "not a number (0-100)")
        return None
    if not (0.0 <= v <= 100.0):
        _warn_pref("WARDEN_MIN_CONVICTION", raw, "outside 0-100")
        return None
    return v


def _pref_sides(raw: str | None) -> frozenset[str] | None:
    """WARDEN_SIDES — comma-separated LONG/SHORT. Invalid tokens are ignored
    (one warning naming the var); nothing valid → absent."""
    if raw is None or not raw.strip():
        return None
    sides = frozenset(t.strip().upper() for t in raw.split(",") if t.strip())
    bad = sorted(sides - set(_PREF_SIDES))
    if bad:
        _warn_pref("WARDEN_SIDES", raw,
                   f"invalid side(s) {bad} ignored (LONG/SHORT only)")
    return (sides & set(_PREF_SIDES)) or None


def _load_signal_preferences() -> SignalPreferences:
    """The three preference env vars → SignalPreferences (all optional)."""
    return SignalPreferences(
        tickers=_pref_tickers(os.getenv("WARDEN_TICKERS")),
        min_conviction=_pref_min_conviction(os.getenv("WARDEN_MIN_CONVICTION")),
        sides=_pref_sides(os.getenv("WARDEN_SIDES")),
    )


def _pref_side_param(sides: frozenset[str] | None) -> str | None:
    """frozenset → the SKU's single-side query param. The backend accepts
    exactly one side (^(LONG|SHORT)$); both sides = no filter — send nothing."""
    if sides and len(sides) == 1:
        return next(iter(sides))
    return None


def _pref_tokens_param(tickers: frozenset[str] | None) -> str | None:
    """frozenset → the SKU's comma-separated tokens param (sorted for a
    byte-stable query string)."""
    if not tickers:
        return None
    return ",".join(sorted(tickers))


_guided_printed = False   # module latch: the FULL guidance block prints ONCE


def _guided_unavailable(reason: str) -> None:
    """Print the caller-actionable next steps for a priced-feed failure.

    SignalsUnavailable is honesty, not a bug — the feed is pay/trial-gated
    and THIS caller's supply failed. Present it as guidance and keep the
    loop alive (the next cycle succeeds once the env var is set).

    Dampened: the FULL block prints ONCE per process; every subsequent
    failure is one dim line (a 60s poll flooding identical setup steps is
    noise, not guidance).
    """
    global _guided_printed
    if _guided_printed:
        print(_c("dim", f"[warden] feed still unavailable ({reason}) — "
                        "see the steps printed above"), file=sys.stderr)
        return
    _guided_printed = True
    if reason == "http_402":
        head = "→ mint a FREE trial token (no account needed):"
        extra = "  → or get a funded key: https://www.vibe.airforce/developers"
    elif reason == "http_429":
        head = "→ trial budget exhausted — mint a fresh token (they're capped):"
        extra = ("  → or get a funded key: https://www.vibe.airforce/developers\n"
                 "  → token-frugal alternative: uv run python -m warden.agent --once")
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
            "  ✓ signals: trial token (free, ~3 calls — then mint another; "
            "the loop polls every 60s — each cycle costs one call; ~3 min "
            "of watching per token)")
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


def _print_ledger_hint(ledger: JsonlLedger) -> None:
    """The ledger trap, defused at startup: a populated ledger silently eats
    every already-seen signal — the operator watches a quiet loop and
    wonders why. One dim line names it and the two ways out."""
    try:
        n = sum(1 for _ in _read_ledger_rows(ledger._path))
    except OSError:
        return
    if n > 0:
        print(_c("dim", f"[warden] ledger: {n} decisions recorded — "
                        "already-seen signals print nothing; delete the file "
                        "or run --fresh to re-decide"))


def _fresh_ledger(ledger: JsonlLedger) -> bool:
    """--fresh: rename the ledger to <name>.bak-<epoch> so the loop starts
    deciding from zero. Returns whether a rename happened."""
    if not ledger._path.exists():
        return False
    backup = ledger._path.with_name(
        f"{ledger._path.name}.bak-{int(time.time())}")
    ledger._path.rename(backup)
    print(f"[warden] ledger moved to {backup.name} — fresh decisions from here")
    return True


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


# --- --setup: guided, live-verified onboarding --------------------------------
#
# Three steps, every one Enter-skippable (the agent runs degraded-but-honest
# without any of them — the checklist at startup says which are missing):
#   ① BRAIN — paste a NEAR AI Cloud key; ONE chat/completions "Reply with
#     OK" proves the key before anything is written (401 → not written).
#   ② SIGNALS — mint the free trial token and verify it against the
#     deliveries SKU (402/429 → honest note; the token itself is fine).
#   ③ EXECUTOR — inert by design; one pointer line, never a prompt.
# A populated-then-masked .env summary closes the flow. Non-interactive
# shells (pipes, CI) degrade to the legacy mint-only path — never a hang.

_SETUP_URL = "https://api.vibe.airforce/api/vibe-tools/trial-auth"
_BRAIN_URL_DEFAULT = "https://cloud-api.near.ai/v1"
_BRAIN_MODEL = "z-ai/glm-5.3-flash"          # NearBrain's wire default


def _mask_token(token: str) -> str:
    """trial_<first6>… — the token never prints whole."""
    return f"trial_{token[6:12]}…"


def _mask_secret(secret: str) -> str:
    """first4+last4 with the middle elided — for NEAR_API_KEY lines.
    Degenerates honestly on short values (never fabricates middle chars)."""
    if len(secret) <= 8:
        return "…"
    return f"{secret[:4]}…{secret[-4:]}"


def _write_env_var(env_path: Path, key: str, value: str) -> bool:
    """Append-or-replace `<key>=<value>` in .env.

    A `<key>=` line gets its VALUE replaced in place (every other line
    byte-identical); no such line → appended. A missing file is created
    with the template's header line. Returns True when the file was
    created fresh."""
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                env_path.write_text("\n".join(lines) + "\n")
                return False
        env_path.write_text("\n".join(lines) + "\n%s=%s\n" % (key, value))
        return False
    env_path.write_text(
        "# Warden configuration — see .env.example for every field.\n"
        "%s=%s\n" % (key, value))
    return True


def _write_env_token(env_path: Path, token: str) -> bool:
    """The trial-token spelling of `_write_env_var` (the legacy name kept
    so existing greps and call sites land somewhere real)."""
    return _write_env_var(env_path, "VIBE_TRIAL_TOKEN", token)


async def _mint_trial_token(http) -> str:
    """ONE mint POST → the `trial_…` token. Raises RuntimeError with a
    caller-printable message on every failure shape (transport, non-200,
    unexpected body, missing token). The injected client is never closed."""
    import httpx

    try:
        res = await http.post(_SETUP_URL)
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        raise RuntimeError(f"trial mint failed: {type(exc).__name__}") from exc
    if res.status_code != 200:
        raise RuntimeError(f"trial mint failed: http {res.status_code}")
    try:
        token = res.json()["data"]["token"]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("trial mint failed: unexpected response shape")
    if not (isinstance(token, str) and token.startswith("trial_")):
        raise RuntimeError("trial mint failed: token missing from response")
    return token


async def _verify_brain(http, api_key: str,
                        base_url: str = _BRAIN_URL_DEFAULT) -> tuple[bool, str]:
    """ONE chat/completions probe in the NearBrain wire shape — Bearer key,
    tiny messages payload. 200 with ANY body = verified (the veto brain's
    real contract is parse-fail-closed; here we only prove the endpoint and
    the key). Returns (ok, kind) where kind ∈ {"verified", "rejected",
    "unreachable"} — 401/403 are a KEY problem (rejected), timeouts and
    5xx are an UPSTREAM problem (unreachable). Never constructs a NearBrain
    (its constructor requires env config); ~the same wire, written out."""
    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": _BRAIN_MODEL,
        "messages": [{"role": "user", "content": "Reply with OK"}],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        res = await http.post(url, json=payload, headers=headers)
    except (httpx.TimeoutException, httpx.TransportError):
        return False, "unreachable"
    if res.status_code in (401, 403):
        return False, "rejected"
    if res.status_code != 200:
        return False, "unreachable"
    return True, "verified"


async def _verify_feed(http, token: str) -> tuple[bool, int, str]:
    """Verify the freshly minted trial token against the deliveries SKU —
    the SAME request the loop's first cycle will make. Returns
    (ok, open_signals, kind): 200 → the open-signal count (every parsed
    row the provider returns); 402/429 → valid-but-capped (honest note,
    the token itself is fine and IS written); anything else → unknown."""
    try:
        provider = VibeSignalProvider(trial_token=token, http=http)
        signals = list(await provider.latest())
    except SignalsUnavailable as exc:
        return False, 0, exc.reason        # "http_402" / "http_429"
    return True, len(signals), "verified"


def _ask(prompt: str, *, hidden: bool = False) -> str:
    """One interactive read. getpass for secrets (no terminal echo);
    EOFError / closed stdin → empty string (= skip), matching bare Enter.
    The test seam: monkeypatch warden.agent._ask."""
    if hidden:
        import getpass
        try:
            return getpass.getpass(prompt)
        except EOFError:
            return ""
    try:
        return input(prompt)
    except EOFError:
        return ""


def _interactive() -> bool:
    """A real operator terminal — BOTH stdin and stdout must be TTYs.
    A piped stdout (`--setup | tee`) is presentation-only and still fine to
    prompt on; a piped STDIN is what must never block."""
    return sys.stdin.isatty() and sys.stdout.isatty()


async def _setup_step_brain(http, env_path: Path) -> bool:
    """① BRAIN — env pre-check → paste (hidden) → live verify → write
    WARDEN_BRAIN_URL + NEAR_API_KEY TOGETHER, only on a verified key.
    401 → not written, retry-or-skip; timeout/5xx → one retry offer."""
    if os.getenv("NEAR_API_KEY"):
        print(_c("green", "✓ brain already configured in environment"))
        return True
    key = _ask("Paste NEAR_API_KEY (Enter = skip): ", hidden=True).strip()
    if not key:
        print(_c("dim", "  · skipped — the heuristic fallback brain stays "
                        "active (set NEAR_API_KEY anytime)"))
        return False
    while True:
        print("  → verifying… ", end="", flush=True)
        ok, kind = await _verify_brain(http, key)
        if ok:
            fresh = not env_path.exists()
            _write_env_var(env_path, "WARDEN_BRAIN_URL", _BRAIN_URL_DEFAULT)
            _write_env_var(env_path, "NEAR_API_KEY", key)
            print(_c("green",
                     f"✓ brain verified and written to {env_path}"))
            if fresh:
                print(_c("dim", "  (.env created)"))
            return True
        if kind == "rejected":
            # the key itself is wrong — writing it would ship a brain that
            # refuses every signal; ask retry-or-skip instead
            print(_c("red",
                     "✗ key rejected — check it at cloud-api.near.ai "
                     "(nothing written)"))
            again = _ask("Paste a different key (Enter = skip): ",
                         hidden=True).strip()
            if not again:
                print(_c("dim", "  · skipped — brain not configured"))
                return False
            key = again
            continue
        # unreachable (timeout / 5xx): ONE retry offer, then skip
        print(_c("red", "✗ NEAR unreachable right now"))
        if _ask("Retry verify? [y/N]: ").strip().lower() in ("y", "yes"):
            continue
        print(_c("dim", "  · skipped — verify later: export NEAR_API_KEY=… "
                        "and re-run --setup"))
        return False


async def _setup_step_signals(http, env_path: Path) -> str | None:
    """② SIGNALS — mint (offering a refresh when one is already set), then
    verify against the deliveries SKU. Returns the minted token or None.
    VIBE_API_KEY in the environment wins over any trial token (the
    provider's own tri-state), so minting would be waste."""
    if os.getenv("VIBE_API_KEY"):
        print(_c("green", "✓ signals already configured in environment "
                          "(funded VIBE_API_KEY)"))
        return None
    if os.getenv("VIBE_TRIAL_TOKEN"):
        ans = _ask("VIBE_TRIAL_TOKEN already set — mint a fresh one? "
                   "[Y/n]: ").strip().lower()
        if ans in ("n", "no"):
            print(_c("dim", "  · keeping the existing token"))
            return None
    print("  ✓ minting… ", end="", flush=True)
    try:
        token = await _mint_trial_token(http)
    except RuntimeError as exc:
        print(_c("red", f"✗ {exc}"))
        return None
    created = _write_env_token(env_path, token)
    verb = "created" if created else "updated"
    print(_c("green", f"✓ token written to {env_path} ({verb}) — "
                      f"VIBE_TRIAL_TOKEN={_mask_token(token)}"))
    print("  → verifying… ", end="", flush=True)
    ok, count, kind = await _verify_feed(http, token)
    if ok:
        print(_c("green", f"✓ feed reachable — {count} open signal(s) on "
                          "the book right now"))
    elif kind in ("http_402", "http_429"):
        # honest note: the mint succeeded and the token IS written — the
        # cap/mint lane just refused THIS probe (cooldown, budget shape)
        print(_c("amber", f"✗ feed said {kind} — the token IS written and "
                          "will work after the cooldown "
                          "(trial tokens are capped)"))
    else:
        print(_c("amber", f"✗ feed unreachable ({kind}) — token written; "
                          "the loop will tell you more on first run"))
    return token


def _print_setup_summary(env_path: Path, wrote_brain: bool) -> None:
    """═══ DONE — your .env ═══ — masked values only, then the next run.
    Reads the vars back out of the FILE we just wrote (the truth of the
    artifact, not the process env — exports may differ)."""
    def _env_value(key: str) -> str | None:
        if not env_path.exists():
            return None
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1]
        return None

    print()
    print(_c("bold", "═══ DONE — your .env ═══"))
    if wrote_brain:
        print(f"  WARDEN_BRAIN_URL={_env_value('WARDEN_BRAIN_URL')}")
        key = _env_value("NEAR_API_KEY") or ""
        print(f"  NEAR_API_KEY={_mask_secret(key)}")
    trial = _env_value("VIBE_TRIAL_TOKEN")
    if trial:
        print(f"  VIBE_TRIAL_TOKEN={_mask_token(trial)}")
    print()
    print("▶ Next: uv run python -u -m warden.agent --dry")


async def _run_setup(http=None, non_interactive: bool = False) -> int:
    """The --setup entrypoint.

    Interactive (TTY + no --non-interactive): the guided ①②③ flow with a
    live brain verify, a feed verify and the masked summary — total wall
    time is two HTTP round trips. Non-interactive (no TTY or the flag):
    the legacy fire-and-forget mint — token written, one line saying why,
    exit. Pipes and CI must never block on input()."""
    import httpx

    env_path = Path(os.getenv("WARDEN_ENV_FILE", ".env"))
    own = http is None
    client = http or httpx.AsyncClient(timeout=15.0)
    try:
        if non_interactive or not _interactive():
            if not non_interactive:
                print("[warden] non-interactive shell — minted trial token "
                      "only (run --setup in a terminal for the guided flow)")
            return await _setup_mint_only(client, env_path)

        from . import __version__
        print(_c("bold", f"WARDEN v{__version__} — setup"))
        print()
        print(_c("bold", "① BRAIN — the veto decision model (NEAR AI Cloud)"))
        print("   Get a key: https://cloud-api.near.ai")
        wrote_brain = await _setup_step_brain(client, env_path)
        print()
        print(_c("bold",
                 "② SIGNALS — the follow feed (free trial, no account)"))
        await _setup_step_signals(client, env_path)
        print()
        print(_c("bold", "③ EXECUTOR — real orders (advanced; Enter = skip, "
                         "dry mode is default)"))
        print(_c("dim", "  · arm later via the fire-test ritual — see README "
                        "'Arm the executor'"))
        _print_setup_summary(env_path, wrote_brain)
        return 0
    finally:
        if own:
            await client.aclose()


async def _setup_mint_only(http, env_path: Path) -> int:
    """The legacy fire-and-forget path (pipes, CI, --non-interactive):
    mint, write, print the next command. Shares the mint coroutine with
    the guided path — one wire shape, two presentations."""
    try:
        token = await _mint_trial_token(http)
    except RuntimeError as exc:
        print(f"[warden] {exc}", file=sys.stderr)
        return 1
    created = _write_env_token(env_path, token)
    verb = "created" if created else "updated"
    print(f"[warden] {env_path} {verb} — VIBE_TRIAL_TOKEN={_mask_token(token)}")
    print("[warden] next: uv run python -u -m warden.agent --dry")
    return 0


async def _main() -> None:
    from dotenv import load_dotenv
    # The cwd .env loads FIRST (override=False): a value already in the real
    # environment always wins — .env fills the gaps, never clobbers exports.
    load_dotenv(override=False)

    ap = argparse.ArgumentParser(prog="warden")
    ap.add_argument("mode", nargs="?", choices=("dry", "status"), default="dry",
                    help="dry = follow loop (default) · "
                         "status = one read-only screen, no polling")
    ap.add_argument("--dry", action="store_true", default=True)
    ap.add_argument("--setup", action="store_true",
                    help="guided setup — collects + live-verifies the brain "
                         "key and trial token, writes .env (Enter skips any "
                         "step; non-interactive shells mint a token only)")
    ap.add_argument("--non-interactive", action="store_true",
                    help="with --setup: skip the guided prompts entirely "
                         "(pipes, CI) — mint a trial token and exit")
    ap.add_argument("--fresh", action="store_true",
                    help="rename the ledger (<name>.bak-<epoch>) and start "
                         "deciding from zero")
    ap.add_argument("--once", action="store_true",
                    help="run exactly one tick, then exit 0")
    ap.add_argument("--min-conviction", type=float, default=60.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--poll", type=int, default=POLL_SECONDS)
    args = ap.parse_args()

    if args.setup:
        raise SystemExit(await _run_setup(
            non_interactive=args.non_interactive))

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
    # Your preferences wire TWICE (the double gate): the provider sends the
    # filters server-side (fewer rows over the wire) and the policy refuses
    # any non-allowlisted coin locally — a feed-side drift can never widen
    # what the wallet follows. Every other Policy field stays at its default.
    prefs = _load_signal_preferences()
    ledger = JsonlLedger()
    agent = Agent(
        signals=VibeSignalProvider(
            min_conviction=prefs.min_conviction,
            side=_pref_side_param(prefs.sides),
            tokens=_pref_tokens_param(prefs.tickers),
        ),
        brain=brain,
        policy=PolicyWallet(Policy(allowed_coins=prefs.tickers)),
        ledger=ledger,
        prefs=prefs,
    )
    _print_banner()
    if args.mode == "status":
        _print_setup_path()
        _print_status()
        return
    if args.fresh:
        _fresh_ledger(ledger)
    if _dry():
        print("[warden] DRY — decisions print, nothing sends")
    _print_setup_path()
    _print_ledger_hint(ledger)
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
