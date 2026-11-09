# Hyperliquid TESTNET fire-test — RECORDED EVIDENCE (2026-11-08)

Operator: repo maintainer · Wallet: a NEW Warden-dedicated testnet-only key
(`.env.testnet`, gitignored — disposable, faucet/testnet-transfer funded,
never a production key; the repo is PUBLIC — this record contains NO key
material).

## Run

```
uv run --extra hl python scripts/fire_test.py order --coin BTC --notional 15
```

```
BTC asset=3 mid=75474.5 szDecimals=5
ceil-reference size=0.0002 (round UP, never plain round)
wired size=0.0002 effective notional=$15.09 (HL floor $10)
ORDER_OK
ORDER_OID=64900271392
WARDEN_TESTNET_FIRE_TEST=pass
```

## Exchange-verified state (POST /info clearinghouseState)

- position: BTC LONG, positionValue **$15.095**, marginUsed 0.75475,
  maxLeverage 40 (the market clamp visible on the position)
- the pre-order probe: `probe --coin BTC` → PROBE_TRANSPORT_OK
  (signature verified upstream while the account was still unfunded)

## Verdict

The typed-action allowlist → two-key arm → market-physics clamp → SDK
signing → POST /exchange path is LIVE end-to-end on Hyperliquid TESTNET.
`WARDEN_TESTNET_FIRE_TEST=pass` is earned and recorded here; the executor's
arm gate references this evidence.
