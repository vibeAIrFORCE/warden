# Warden

**An open, policy-capped trading agent.** Warden follows a signal book into
perpetual futures, asks a decision model before every move, and refuses
itself before it ever overspends — by construction, not by prompt.

Built for the NEAR *Build Beyond Boundaries* hackathon (Nov 2026) as a
three-layer verifiable trader:

```
┌─────────────────────────────────────────────────────────────┐
│  BRAIN      any decision model. Default: our fine-tuned     │
│             4B classifier (open weights, trained on real    │
│             trade outcomes) · pluggable: any local model    │
│             or a heuristic — the interface is the contract  │
├─────────────────────────────────────────────────────────────┤
│  POLICY     the wallet that refuses. Position caps, size     │
│             caps, deployed-capital caps, a daily fuse —     │
│             checked BEFORE an order is ever built. The      │
│             executor accepts three typed actions and        │
│             nothing else; withdrawals are structurally      │
│             unsendable.                                     │
├─────────────────────────────────────────────────────────────┤
│  SETTLEMENT deposit from any chain via NEAR Intents         │
│             (1Click) straight into your Hyperliquid perps   │
│             balance. Your EVM address IS your HL account —  │
│             no signup, no API keys.                         │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
uv sync
uv run pytest -q                       # the teeth, network-free

# the follow loop (dry — prints decisions, sends nothing):
uv run python -m warden.agent --dry
```

### Arm the executor

The executor is inert until TWO keys are set, and every dispatch re-checks
both: `WARDEN_ENABLED=1` (the flag) plus `WARDEN_TESTNET_FIRE_TEST=pass`
(the marker an operator exports only after a recorded testnet fire-test).
Status today: **signing wired — testnet fire-test pending**. The typed-action
allowlist and the arm gate are live and tested — see *Do X, see Y* below for
the refusal probes you can run yourself, right now, network-free.

## The shape

```python
from warden.signals import VibeSignalProvider   # our data by default
from warden.brain   import HeuristicBrain       # swap in your model
from warden.policy  import PolicyWallet
from warden.executor import HyperliquidExecutor

agent = Agent(
    signals = VibeSignalProvider(),             # or FileSignalProvider(...), or yours
    brain   = HeuristicBrain(min_conviction=60),# or ModelBrain("./my-4b")
    policy  = PolicyWallet(max_trade_usd=25, max_open_positions=3,
                           daily_budget_usd=50),
    executor= HyperliquidExecutor.from_env(),   # inert until armed
)
agent.run()
```

Every layer is an interface. Bring your own signals, your own brain, your
own exchange — the policy layer and the action allowlist are the parts we
think everyone should copy.

## Status

| Layer | State |
|---|---|
| Policy wallet + action allowlist | working, unit-tested (the teeth) |
| Signal providers | local files working; public feed wired but returns zero rows today (probed through `VibeSignalProvider().latest()`, 2026-11-08) |
| Brain adapters | heuristic working · model slot stubbed (weights pending eval gate) |
| HL executor (sign + post) | signing wired — testnet fire-test pending (the typed-action allowlist and the two-key arm gate are live and tested) |
| 1Click settlement | quote/status client built; end-to-end pending live fire |
| The follow loop | dry mode working |

## Do X, see Y

Every expected output below was captured from a real run in this repo on
2026-11-08. Counts and timings drift as the teeth battery grows — the shape
is what matters.

### Install

```bash
uv sync
```

```text
Resolved 91 packages in 0.68ms
Audited 18 packages in 0.05ms
```

### Run the teeth (the refusal battery — network-free)

```bash
uv run pytest -q
```

```text
........................................................................ [ 60%]
................................................                         [100%]
120 passed in 0.22s
```

### Watch the follow loop (dry — decisions print, nothing sends)

```bash
timeout 15 uv run python -u -m warden.agent --dry
```

The loop polls the live feed every 60 seconds until Ctrl-C, so the `timeout 15`
IS part of the recipe — it stops the infinite loop for you (exit code 124 is
the expected stop, not an error). `-u` keeps the banner visible when the
timeout fires: without it the line sits in Python's stdout buffer and is lost
when SIGTERM lands under a pipe. First-cycle output today:

```text
[warden] DRY — decisions print, nothing sends
```

`FOLLOW` / `skip` lines print here when the feed has open signals — today the
public feed returns zero usable rows (see Status).

### Prove the executor refuses (network-free, unarmed)

A raw dict never reaches the signer — the allowlist is a type check, so
withdrawals and transfers are structurally unsendable:

```bash
uv run python -c "import sys; from warden.executor import HyperliquidExecutor; sys.excepthook = lambda t, v, b: print(f'{t.__name__} - {v}'); HyperliquidExecutor().sign_and_post({'type': 'withdraw'}).send(None)"
```

```text
ActionNotAllowed - dict is not on the allowlist — withdrawals and transfers are structurally unsendable
```

An allowlisted action still refuses to move while unarmed — the two-key arm
gate fires before any key is touched. Both probes exit 1 by design: the
refusal is the demo.

```bash
uv run python -c "import sys; from warden.executor import HyperliquidExecutor, OrderAction; sys.excepthook = lambda t, v, b: print(f'{t.__name__} - {v}'); HyperliquidExecutor().sign_and_post(OrderAction(coin='BTC', asset=3, is_buy=True, size=0.001, limit_px=1.0)).send(None)"
```

```text
NotArmed - WARDEN_ENABLED not set — the executor is inert
```

## Honest notes

- Real-money trading is **armed by hand** — the executor is inert until an
  explicit two-key arm (a flag + a recorded testnet fire-test), and every
  cycle re-checks both.
- The default signal feed is provided by
  [VIBE](https://vibe.airforce) (our smart-money aggregation pipeline);
  implement `SignalProvider` for anything else.
- This is hackathon software: small sizes, every loss visible.

## License

Apache-2.0
