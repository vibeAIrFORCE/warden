"""Signal providers — where the follow list comes from.

`SignalProvider` is the whole contract: `latest()` returns the OPEN signals
the agent may follow. Two implementations ship; write your own for anything
else (a Telegram channel, an on-chain feed, your own model's outputs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import httpx

from .types import Signal, Side


class SignalProvider(Protocol):
    def latest(self) -> Sequence[Signal]: ...


class VibeSignalProvider:
    """The default feed — VIBE's public delivered-signals history.

    Our smart-money aggregation pipeline scores wallets across chains and
    delivers a curated signal book; this provider reads its PUBLIC history
    endpoint (no key, anonymous). The agent follows the book's own recorded
    entry price — same entry as the book, never the current mark.
    """

    def __init__(
        self,
        base_url: str = "https://api.vibe.airforce/api",
        http: httpx.AsyncClient | None = None,
        limit: int = 25,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = http
        self._limit = limit

    async def latest(self) -> Sequence[Signal]:
        own = self._http is None
        http = self._http or httpx.AsyncClient(timeout=15.0)
        try:
            res = await http.get(
                f"{self._base}/top-picks/signals/history",
                params={"limit": self._limit},
            )
            res.raise_for_status()
            rows = res.json().get("signals") or res.json().get("rows") or []
        finally:
            if own:
                await http.aclose()
        out: list[Signal] = []
        for r in rows:
            side = str(r.get("side") or "").strip().upper()
            if side not in ("LONG", "SHORT"):
                continue
            out.append(
                Signal(
                    signal_id=str(r.get("id") or r.get("signal_id") or ""),
                    coin=str(r.get("coin") or r.get("token_symbol") or ""),
                    side=side,  # type: ignore[arg-type]
                    entry_price=_f(r.get("entry_price")),
                    conviction=_f(r.get("conviction")),
                    meta={"source": "vibe"},
                )
            )
        return [s for s in out if s.signal_id and s.coin]


class FileSignalProvider:
    """A JSONL file of signals — the offline/pluggable path. One JSON object
    per line with at least {signal_id, coin, side}."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def latest(self) -> Sequence[Signal]:
        if not self._path.exists():
            return []
        out: list[Signal] = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            side = str(r.get("side") or "").strip().upper()
            if side not in ("LONG", "SHORT"):
                continue
            out.append(
                Signal(
                    signal_id=str(r.get("signal_id") or ""),
                    coin=str(r.get("coin") or ""),
                    side=side,  # type: ignore[arg-type]
                    entry_price=_f(r.get("entry_price")),
                    conviction=_f(r.get("conviction")),
                    meta={"source": "file"},
                )
            )
        return [s for s in out if s.signal_id and s.coin]


def _f(v) -> float | None:
    try:
        f = float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
    return f if f and f > 0 else None
