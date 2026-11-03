"""The ledger — append-only JSONL, the source of truth for exactly-once."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class JsonlLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or os.getenv("WARDEN_LEDGER", "warden_ledger.jsonl"))

    def seen(self, signal_id: str) -> bool:
        if not self._path.exists():
            return False
        for line in self._path.read_text().splitlines():
            try:
                if json.loads(line).get("signal_id") == signal_id:
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def record(self, entry: dict) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
        with self._path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
