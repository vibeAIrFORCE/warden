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
| Signal providers | working (public feed + local files) |
| Brain adapters | heuristic working · model slot stubbed (weights pending eval gate) |
| HL executor (sign + post) | built against hyperliquid-python-sdk; testnet first |
| 1Click settlement | quote/status client built; end-to-end pending live fire |
| The follow loop | dry mode working |

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
