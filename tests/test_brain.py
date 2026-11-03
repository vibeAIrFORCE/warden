"""Brains — the decision layer contract.

HeuristicBrain's verdicts (follow at floor+, exactly-at-floor, refuse below,
refuse None) and ModelBrain's fail-closed construction + inference stub.
Network-free.
"""

import pytest

from warden.brain import HeuristicBrain, ModelBrain, ModelNotReady
from warden.types import PositionState, Signal


def sig(conviction=70.0) -> Signal:
    return Signal(signal_id="s1", coin="BTC", side="LONG",
                  entry_price=100.0, conviction=conviction)


# ---------------------------------------------------------- HeuristicBrain


async def test_heuristic_follows_above_floor():
    d = await HeuristicBrain(min_conviction=60.0, size_usd=10.0).decide(
        sig(70.0), PositionState())
    assert d.follow is True
    assert d.size_usd == 10.0
    assert d.leverage == 1.0
    assert d.source == "heuristic"
    assert ">=" in d.reason


async def test_heuristic_follows_exactly_at_floor():
    d = await HeuristicBrain(min_conviction=60.0).decide(
        sig(60.0), PositionState())
    assert d.follow is True


async def test_heuristic_refuses_below_floor():
    d = await HeuristicBrain(min_conviction=60.0).decide(
        sig(59.0), PositionState())
    assert d.follow is False
    assert d.size_usd is None
    assert "conviction 59 < floor 60" in d.reason


async def test_heuristic_refuses_missing_conviction():
    d = await HeuristicBrain().decide(sig(conviction=None), PositionState())
    assert d.follow is False
    assert "no conviction on signal" in d.reason


# -------------------------------------------------------------- ModelBrain


def test_model_brain_missing_dir_fails_closed(tmp_path):
    with pytest.raises(ModelNotReady) as ei:
        ModelBrain(tmp_path / "absent")
    assert "does not exist" in str(ei.value)


def test_model_brain_dir_without_config_fails_closed(tmp_path):
    (tmp_path / "my-model").mkdir()
    with pytest.raises(ModelNotReady) as ei:
        ModelBrain(tmp_path / "my-model")
    assert "warden_model.json" in str(ei.value)


async def test_model_brain_inference_not_ready_until_weights(tmp_path):
    """A valid model dir constructs — but decide() is a hard stop until the
    weights ship. The agent fails closed instead of guessing."""
    d = tmp_path / "m"
    d.mkdir()
    (d / "warden_model.json").write_text('{"arch": "4b-gate"}')
    brain = ModelBrain(d)
    with pytest.raises(ModelNotReady) as ei:
        await brain.decide(sig(), PositionState())
    assert "inference ships with the weights" in str(ei.value)
