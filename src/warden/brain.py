"""Brain adapters — the decision layer.

`BrainAdapter` is the whole contract: given a Signal (+ the wallet's
position state), return a Decision. Two adapters ship:

  HeuristicBrain — a tiny real rule (conviction floor + sizing), so the
                   repo runs end-to-end on day one.
  ModelBrain     — the local decision-model slot. Point it at a directory
                   holding a fine-tuned classifier (our 4B when it passes
                   its eval gate; any HF-compatible model otherwise) and it
                   becomes the brain. Stubbed until weights exist.
  NearBrain      — the remote veto brain (OpenAI chat/completions wire):
                   a model is asked enter/hold/exit with probabilities and
                   only a confident `enter` follows. Everything else —
                   hold, low margin, unparseable, transport failure,
                   missing key — refuses. The operator key is read from
                   the environment at call time; never baked, never logged.

The brain NEVER sizes beyond what the policy wallet will allow — it
proposes, the policy disposes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Protocol

import httpx

from .types import Decision, PositionState, Signal


class BrainAdapter(Protocol):
    async def decide(self, signal: Signal, state: PositionState) -> Decision: ...


class HeuristicBrain:
    """Follow signals whose conviction clears a floor, at a fixed size.

    Real rule, honestly dumb: it exists so the loop is complete without a
    model. Replace it — that's the point of the interface.
    """

    def __init__(self, min_conviction: float = 60.0, size_usd: float = 10.0,
                 leverage: float = 1.0) -> None:
        self.min_conviction = min_conviction
        self.size_usd = size_usd
        self.leverage = leverage

    async def decide(self, signal: Signal, state: PositionState) -> Decision:
        if signal.conviction is None:
            return Decision(False, reason="no conviction on signal",
                            source="heuristic")
        if signal.conviction < self.min_conviction:
            return Decision(
                False,
                reason=f"conviction {signal.conviction:.0f} < floor "
                       f"{self.min_conviction:.0f}",
                source="heuristic",
            )
        return Decision(
            True,
            size_usd=self.size_usd,
            leverage=self.leverage,
            reason=f"conviction {signal.conviction:.0f} >= floor",
            source="heuristic",
        )


class ModelBrain:
    """The local decision-model slot.

    Point `model_dir` at a directory with a fine-tuned classifier (our 4B
    once it clears its eval battery — weights are NOT shipped in this repo
    until then). Input features are the signal + position state rendered to
    the model's prompt/feature schema; the output must be a strict JSON
    Decision.

    Raises ModelNotReady on construction when the directory is absent —
    the agent fails closed to the fallback brain instead of guessing.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self._dir = Path(model_dir)
        self._config: dict = {}
        if not self._dir.exists():
            raise ModelNotReady(f"{self._dir} does not exist")
        cfg = self._dir / "warden_model.json"
        if not cfg.exists():
            raise ModelNotReady(f"{cfg} missing (not a warden model dir)")
        self._config = json.loads(cfg.read_text())

    async def decide(self, signal: Signal, state: PositionState) -> Decision:
        raise ModelNotReady(
            "inference ships with the weights — see README Status"
        )


class ModelNotReady(RuntimeError):
    pass


class NearBrainUnavailable(RuntimeError):
    """Raised at CONSTRUCTION when no brain endpoint is configured — the
    agent fails closed to the fallback brain instead of guessing."""


# The system half of the veto prompt. The user half is rendered per signal
# by _render_veto_prompt below. Both are pure strings — no I/O anywhere.
_VETO_SYSTEM_PROMPT = (
    "You are the veto brain of Warden, a policy-capped trading agent. You "
    "are given one trading signal and the wallet's current position state. "
    "Decide whether following that signal is a good idea right now. Reply "
    "with ONLY strict JSON — no prose, no markdown, no code fences — of "
    "the exact shape "
    '{"action":"enter"|"hold"|"exit",'
    '"probabilities":{"enter":<number>,"hold":<number>,"exit":<number>}} '
    "where the three probabilities are your confidence in each action and "
    "sum to roughly 1.0."
)


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:g}"


def _render_veto_prompt(signal: Signal, state: PositionState) -> str:
    """Pure render helper — deterministic, no I/O. The model sees the
    signal (coin/side/entry/conviction) and the wallet state; nothing else."""
    return (
        f"Signal: coin={signal.coin} side={signal.side} "
        f"entry_price={_num(signal.entry_price)} "
        f"conviction={_num(signal.conviction)}\n"
        f"Wallet state: open_positions={state.open_positions} "
        f"deployed_usd={state.deployed_usd:.2f} "
        f"spent_today_usd={state.spent_today_usd:.2f} "
        f"held_coins={', '.join(sorted(state.held_coins)) or 'none'}\n"
        "Answer with the strict JSON object described in your instructions."
    )


class NearBrain:
    """The remote veto brain over an OpenAI chat/completions wire (NEAR AI
    Cloud serves it today; any OpenAI-shaped endpoint works).

    Fail-closed by construction, in two layers:

      * construction — no endpoint configured raises NearBrainUnavailable
        and the agent falls back to HeuristicBrain (brain selection happens
        at construction ONLY; a runtime failure never swaps brains
        mid-flight, it refuses that one signal);
      * decide() — every unavailable/parse/margin failure returns a refusal
        Decision naming the cause. decide() NEVER raises and a
        low-confidence answer never vetoes (the repo's fail-closed law).

    The operator key is read at CALL time from the environment — there is
    no default, no literal, and it never appears in logs, exceptions, or
    Decision.reason. `http` is the test-injection seam (mirrors
    VibeSignalProvider): injected clients are never closed by the brain;
    only a client the brain constructed itself is.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "z-ai/glm-5.3-flash",
        timeout: float = 10.0,
        min_margin: float = 0.2,
        size_usd: float = 10.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        base = (base_url or os.getenv("WARDEN_BRAIN_URL", "")).rstrip("/")
        if not base:
            raise NearBrainUnavailable("WARDEN_BRAIN_URL not set")
        self._base = base
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._min_margin = min_margin
        self._size_usd = size_usd
        self._http = http

    async def decide(self, signal: Signal, state: PositionState) -> Decision:
        api_key = self._api_key or os.getenv("NEAR_API_KEY", "")
        if not api_key:
            return _refused("unavailable", "operator key not set at call time")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _VETO_SYSTEM_PROMPT},
                {"role": "user", "content": _render_veto_prompt(signal, state)},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{self._base}/chat/completions"

        own = self._http is None
        http = self._http or httpx.AsyncClient(timeout=self._timeout)
        try:
            try:
                res = await http.post(url, json=payload, headers=headers)
                if res.status_code == 429 or res.status_code >= 500:
                    # exactly ONE immediate retry on 429/5xx — no backoff
                    res = await http.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError,
                    httpx.InvalidURL) as exc:
                return _refused(
                    "unavailable",
                    f"{type(exc).__name__} (single attempt, no retry)",
                )
            if res.status_code != 200:
                return _refused("unavailable", f"http {res.status_code}")
            try:
                data = res.json()
            except ValueError:
                return _refused("refused", "response body is not JSON")
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return _refused(
                    "refused", "response missing choices[0].message.content"
                )
            if not isinstance(content, str):
                return _refused("refused", "content is not a JSON string")
            return _parse_verdict(content, self._min_margin, self._size_usd)
        finally:
            if own:
                await http.aclose()


def _refused(kind: str, cause: str) -> Decision:
    """The fail-closed verdict — follow=False, source nearbrain, the cause
    named in the reason. Never raises, never allows."""
    return Decision(False, reason=f"nearbrain {kind}: {cause}",
                    source="nearbrain")


def _parse_verdict(content: str, min_margin: float, size_usd: float) -> Decision:
    """The fail-closed parse table. Every malformed / low-confidence reply
    is a refusal; only a confident `enter` follows (policy disposes)."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _refused("refused", "content is not valid JSON")
    if not isinstance(parsed, dict):
        return _refused("refused", "content is not a JSON object")
    action = parsed.get("action")
    probs = parsed.get("probabilities")
    if not isinstance(action, str) or action not in ("enter", "hold", "exit"):
        return _refused("refused", "action not in {enter, hold, exit}")
    if not isinstance(probs, dict) or set(probs) != {"enter", "hold", "exit"}:
        return _refused(
            "refused", "probabilities must have exactly enter/hold/exit"
        )
    values: dict[str, float] = {}
    for key in ("enter", "hold", "exit"):
        v = probs[key]
        # bool is an int subclass — reject it BEFORE the number check;
        # NaN/Infinity parse through json.loads and must refuse too.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return _refused("refused", f"probability {key} is not a JSON number")
        f = float(v)
        if not math.isfinite(f) or not (0.0 <= f <= 1.0):
            return _refused("refused", f"probability {key} not in [0, 1]")
        values[key] = f
    ordered = sorted(values.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    if margin < min_margin:
        return _refused(
            "refused", f"margin {margin:.2f} < min_margin {min_margin}"
        )
    top = max(values, key=values.get)  # ties already refused on margin
    if top == "hold":
        return _refused("refused", "argmax hold — nothing to follow")
    if top == "exit":
        # v1 veto: refuse the ENTRY. Close-execution is out of scope (the
        # executor has no close builder; refusing entries IS the veto).
        return Decision(False,
                        reason=f"nearbrain exit-veto (margin {margin:.2f})",
                        source="nearbrain")
    return Decision(True, size_usd=size_usd, leverage=1.0,
                    reason=f"nearbrain enter (margin {margin:.2f})",
                    source="nearbrain")
