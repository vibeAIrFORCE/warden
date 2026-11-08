"""Asset resolution — coin symbol to Hyperliquid universe metadata.

The Hyperliquid ``/info`` meta endpoint returns the tradable universe as an
ARRAY; the array index IS the asset id wire actions carry, and each entry
carries the market's own ``maxLeverage`` and ``szDecimals``. The resolver
fetches once per resolver lifetime and caches the name->meta map — every
later resolve is a dict lookup.

Fail closed: an unknown coin, an HTTP failure, or an unparseable payload
raises :class:`AssetUnresolved` — callers must never guess an asset id (the
silent-asset-0 class is dead everywhere). httpx is a base dependency; no
new package is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.hyperliquid-testnet.xyz"
_META_TIMEOUT_S = 15.0


class AssetUnresolved(RuntimeError):
    """A coin could not be resolved to universe metadata — fail closed."""


@dataclass(frozen=True)
class AssetMeta:
    """One universe entry: array-index asset id + the market's own physics."""

    asset: int
    max_leverage: int
    sz_decimals: int


class AssetResolver:
    """Resolves coin symbols against the live ``/info`` meta universe.

    ONE httpx POST per resolver lifetime; the parsed map is cached. The
    array index of each universe entry is the asset id, so entries are
    taken in order (delisted entries included — skipping any would shift
    every later index and corrupt the wire asset ids).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._cache: dict[str, AssetMeta] | None = None

    async def resolve(self, coin: str) -> AssetMeta:
        if self._cache is None:
            self._cache = await self._fetch()
        meta = self._cache.get(coin)
        if meta is None:
            raise AssetUnresolved(
                f"unknown coin {coin!r} — not in the {self.base_url} universe"
            )
        return meta

    async def _fetch(self) -> dict[str, AssetMeta]:
        try:
            async with httpx.AsyncClient(timeout=_META_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{self.base_url}/info", json={"type": "meta"}
                )
                resp.raise_for_status()
                payload: Any = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AssetUnresolved(
                f"meta fetch failed against {self.base_url}/info: {exc}"
            ) from exc
        return self._parse(payload)

    @staticmethod
    def _parse(payload: Any) -> dict[str, AssetMeta]:
        try:
            universe = payload["universe"]
            mapping: dict[str, AssetMeta] = {}
            for idx, entry in enumerate(universe):
                mapping[entry["name"]] = AssetMeta(
                    asset=idx,
                    max_leverage=int(entry["maxLeverage"]),
                    sz_decimals=int(entry["szDecimals"]),
                )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise AssetUnresolved(f"meta payload parse failed: {exc}") from exc
        return mapping


class MappingAssetResolver:
    """Network-free test double with the same ``resolve`` interface."""

    def __init__(self, mapping: dict[str, AssetMeta]) -> None:
        self._mapping = dict(mapping)

    async def resolve(self, coin: str) -> AssetMeta:
        meta = self._mapping.get(coin)
        if meta is None:
            raise AssetUnresolved(f"unknown coin {coin!r} — not in the mapping")
        return meta
