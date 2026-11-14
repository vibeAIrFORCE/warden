"""Signal providers — where the follow list comes from.

`SignalProvider` is the whole contract: `latest()` returns the OPEN signals
the agent may follow. Two implementations ship; write your own for anything
else (a Telegram channel, an on-chain feed, your own model's outputs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, Sequence

import httpx

from .types import Signal, Side


class SignalProvider(Protocol):
    def latest(self) -> Sequence[Signal]: ...


class SignalsUnavailable(RuntimeError):
    """The priced signal feed is honestly unavailable to THIS caller.

    Raised ONLY for payment/auth-supply failures the caller can act on:
    HTTP 402 (insufficient credits or an x402 payment challenge) and
    HTTP 429 (trial budget exhausted). Every other failure — invalid key,
    upstream 5xx, timeout, undecodable JSON — propagates unchanged; the
    provider NEVER converts a failure into a silent empty book ([] is
    returned only on HTTP 200 with zero open deliveries).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class VibeSignalProvider:
    """The default feed — VIBE's priced delivered-signals SKU.

    Reads GET {base_url}/x402/alpha/jev/deliveries — perps TG-signal
    delivery lifecycles with the binding entry_price the agent follows.
    Credential tri-state: VIBE_API_KEY env (or api_key=) wins and is sent
    as `x-api-key`; else VIBE_TRIAL_TOKEN env (or trial_token=) is sent as
    `Authorization: Bearer <token>`; else the request is keyless (the
    endpoint answers with its x402 payment challenge). The agent follows
    the book's own recorded entry price — same entry as the book, never
    the current mark.

    Optional preference filters (min_conviction / side / tokens) ride the
    query string — the backend filters server-side, so a narrowed book
    costs fewer rows over the wire. Only non-None params are sent: absent
    filters produce the exact unfiltered URL as before.
    """

    def __init__(
        self,
        base_url: str = "https://api.vibe.airforce/api",
        api_key: str | None = None,
        trial_token: str | None = None,
        http: httpx.AsyncClient | None = None,
        limit: int = 25,
        min_conviction: float | None = None,
        side: str | None = None,
        tokens: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else os.getenv("VIBE_API_KEY")
        self._trial = trial_token if trial_token is not None else os.getenv("VIBE_TRIAL_TOKEN")
        self._http = http
        self._limit = limit
        self._min_conviction = min_conviction
        self._side = side
        self._tokens = tokens

    def _headers(self) -> dict[str, str]:
        """Credential tri-state: key lane wins, then trial, else keyless.
        A malformed credential (placeholder dots, whitespace, non-ASCII —
        all seen in the wild) refuses HERE with one guided line instead of
        an httpx UnicodeEncodeError deep inside a header merge."""
        if self._api_key:
            key = self._api_key
            if not (key.startswith("pk_") and ":sk_" in key and key.isascii()):
                raise SignalsUnavailable(
                    reason="bad_api_key_format — expected pk_…:sk_… (get one "
                           "at https://www.vibe.airforce/developers); paste "
                           "the WHOLE pair, no dots or spaces"
                )
            return {"x-api-key": key}
        if self._trial:
            tok = self._trial
            if not (tok.startswith("trial_") and tok.isascii()
                    and len(tok) <= 128 and "…" not in tok):
                raise SignalsUnavailable(
                    reason="bad_trial_token_format — expected the WHOLE "
                           "trial_ token the curl prints (38 chars, no "
                           "dots/ellipsis); mint another:"
                           " curl -s -X POST "
                           "https://api.vibe.airforce/api/vibe-tools/trial-auth"
                )
            return {"Authorization": f"Bearer {tok}"}
        return {}

    async def latest(self) -> Sequence[Signal]:
        own = self._http is None
        http = self._http or httpx.AsyncClient(timeout=15.0)
        # Preference filters ride the query ONLY when set — insertion order
        # keeps status/limit first, so the no-preferences URL is byte-identical
        # to the unfiltered wire of every release before them.
        params: dict[str, object] = {"status": "open", "limit": self._limit}
        if self._min_conviction is not None:
            params["min_conviction"] = self._min_conviction
        if self._side is not None:
            params["side"] = self._side
        if self._tokens is not None:
            params["tokens"] = self._tokens
        try:
            res = await http.get(
                f"{self._base}/x402/alpha/jev/deliveries",
                params=params,
                headers=self._headers(),
            )
            if res.status_code in (402, 429):
                # Honest-unavailable: the caller's payment/credential supply
                # failed (INSUFFICIENT_CREDITS envelope, x402 payment
                # challenge, or exhausted trial budget). NEVER fake-empty.
                raise SignalsUnavailable(reason=f"http_{res.status_code}")
            res.raise_for_status()
            body = res.json()
            rows = (
                (body.get("data") or {}).get("deliveries")
                if isinstance(body.get("data"), dict)
                else body.get("data")
            ) or []
            if not isinstance(rows, list):
                rows = []
        finally:
            if own:
                await http.aclose()
        out: list[Signal] = []
        candidates = 0  # rows that SURVIVE the exited_at skip (open candidates)
        for r in rows:
            if not isinstance(r, dict) or r.get("exited_at"):
                continue  # SKIP any row with exited_at set
            candidates += 1
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
                    meta={
                        "source": "vibe",
                        "payment_mode": ((body.get("meta") or {}).get("payment_mode")
                                         if isinstance(body.get("meta"), dict) else None),
                    },
                )
            )
        out = [s for s in out if s.signal_id and s.coin]
        if candidates and not out:
            raise SignalsUnavailable(reason="parse_dropped_all_rows")
        return out


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
