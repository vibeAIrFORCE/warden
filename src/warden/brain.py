"""Brain adapters — the decision layer.

`BrainAdapter` is the whole contract: given a Signal (+ the wallet's
position state), return a Decision. Two adapters ship:

  HeuristicBrain — a tiny real rule (conviction floor + sizing), so the
                   repo runs end-to-end on day one.
  ModelBrain     — the local decision-model slot. Point it at a directory
                   holding a fine-tuned classifier (our 4B when it passes
                   its eval gate; any HF-compatible model otherwise) and it
                   becomes the brain. Stubbed until weights exist.

The brain NEVER sizes beyond what the policy wallet will allow — it
proposes, the policy disposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

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
