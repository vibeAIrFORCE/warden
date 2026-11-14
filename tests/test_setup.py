"""--setup — the guided interactive flow, network-free.

The wire rides httpx.MockTransport: brain verify (200 / 401 / 500),
trial mint, feed verify. The terminal is simulated — `_ask` monkeypatched
for pastes/answers, `_interactive` for the TTY check — so pytest never
blocks and never touches a real key. .env bytes are asserted directly.
"""

import httpx
import pytest

from warden import agent as A
from warden.agent import (
    _mask_secret,
    _mask_token,
    _run_setup,
    _verify_brain,
    _write_env_var,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _mint_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": {"token": "trial_" + "ab" * 16}})


def _setup_client() -> httpx.AsyncClient:
    return _client(lambda request: (
        _mint_ok(request)
        if request.url.path.endswith("/trial-auth")
        else httpx.Response(200, json={
            "success": True, "data": {"deliveries": [
                {"id": "v1", "coin": "BTC", "side": "LONG"},
                {"id": "v2", "coin": "ETH", "side": "SHORT"},
            ]},
        })
    ))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The setup steps read NEAR_API_KEY / VIBE_* from the process env —
    every test starts from a clean slate."""
    for var in ("NEAR_API_KEY", "VIBE_API_KEY", "VIBE_TRIAL_TOKEN"):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------ helpers


def test_mask_token_never_prints_whole():
    tok = "trial_" + "a1b2c3d4" * 5
    m = _mask_token(tok)
    assert m.startswith("trial_a1b2c3")
    assert m.endswith("…")
    assert tok not in m and len(m) < len(tok)


def test_mask_secret_first4_last4():
    assert _mask_secret("sk-ant-verylongsecretkey1234") == "sk-a…1234"
    assert _mask_secret("short") == "…"          # short: degenerate honestly


def test_write_env_var_replaces_line_in_place(tmp_path):
    env = tmp_path / ".env"
    env.write_text("WARDEN_TICKERS=BTC\nNEAR_API_KEY=old\nVIBE_TRIAL_TOKEN=t1\n")
    fresh = _write_env_var(env, "NEAR_API_KEY", "newkey")
    assert fresh is False
    text = env.read_text()
    assert "NEAR_API_KEY=newkey\n" in text
    assert "old" not in text
    assert "WARDEN_TICKERS=BTC\n" in text and "VIBE_TRIAL_TOKEN=t1\n" in text


def test_write_env_var_appends_when_absent_and_creates_file(tmp_path):
    env = tmp_path / ".env"
    assert _write_env_var(env, "NEAR_API_KEY", "k1") is True    # created fresh
    text = env.read_text()
    assert text.startswith("# Warden configuration")
    assert "NEAR_API_KEY=k1\n" in text
    _write_env_var(env, "WARDEN_BRAIN_URL", "https://x/v1")
    assert "WARDEN_BRAIN_URL=https://x/v1\n" in env.read_text()


# ------------------------------------------------------------ brain verify


async def test_verify_brain_wire_shape():
    """ONE POST to {base}/chat/completions, Bearer key, tiny payload —
    the NearBrain wire, verified at the wire level."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["payload"] = request.read()
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "OK"}}]})

    async with _client(handler) as http:
        ok, kind = await _verify_brain(http, "sk-live-key-9999")
    assert ok is True and kind == "verified"
    assert seen["url"] == ("https://cloud-api.near.ai/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-live-key-9999"
    body = seen["payload"].decode()
    assert '"content": "Reply with OK"' in body or '"content":"Reply with OK"' in body


async def test_verify_brain_kinds():
    def ok(request):
        return httpx.Response(200, json={})

    def rejected(request):
        return httpx.Response(401, json={"error": "bad key"})

    def down(request):
        return httpx.Response(503, text="boom")

    async with _client(ok) as http:
        assert await _verify_brain(http, "k") == (True, "verified")
    async with _client(rejected) as http:
        assert await _verify_brain(http, "k") == (False, "rejected")
    async with _client(down) as http:
        assert await _verify_brain(http, "k") == (False, "unreachable")


# ------------------------------------------------------------ guided flow


async def test_guided_brain_success_writes_both_vars(tmp_path, monkeypatch,
                                                     capsys):
    """① paste → 200 → WARDEN_BRAIN_URL + NEAR_API_KEY written TOGETHER;
    the summary masks the key (full token absent from captured output)."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trial-auth"):
            return _mint_ok(request)
        if "/chat/completions" in str(request.url):
            seen["verify"] = str(request.url)
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"data": {"deliveries": []}})

    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask", lambda prompt, **kw: "sk-real-brain-key-4321")
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _client(handler) as http:
        rc = await _run_setup(http=http)

    assert rc == 0
    env = tmp_path / ".env"
    text = env.read_text()
    assert "verify" in seen and seen["verify"].endswith("/chat/completions")
    assert "WARDEN_BRAIN_URL=https://cloud-api.near.ai/v1\n" in text
    assert "NEAR_API_KEY=sk-real-brain-key-4321\n" in text
    assert "VIBE_TRIAL_TOKEN=trial_" in text
    out = capsys.readouterr().out
    assert "brain verified and written" in out
    assert "sk-real-brain-key-4321" not in out          # full key NEVER prints
    assert "NEAR_API_KEY=sk-r…4321" in out              # masked form does
    assert "_mask" not in out


async def test_guided_brain_401_not_written_then_skip(tmp_path, monkeypatch,
                                                      capsys):
    """401 → ✗ rejected line, NOTHING brain-shaped in .env, retry prompt
    answered with Enter → skip; ② still mints."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/trial-auth"):
            return _mint_ok(request)
        if "/chat/completions" in str(request.url):
            calls["n"] += 1
            return httpx.Response(401, json={"error": "nope"})
        return httpx.Response(200, json={"data": {"deliveries": []}})

    answers = iter(["sk-bad-key", ""])          # paste, then Enter = skip

    def ask(prompt, **kw):
        return next(answers)

    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask", ask)
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _client(handler) as http:
        rc = await _run_setup(http=http)

    assert rc == 0
    text = (tmp_path / ".env").read_text()
    assert "NEAR_API_KEY" not in text           # the rejected key: NOT written
    assert "WARDEN_BRAIN_URL" not in text
    assert "VIBE_TRIAL_TOKEN=trial_" in text    # ② ran anyway
    assert calls["n"] == 1                      # skip ended the verify loop
    out = capsys.readouterr().out
    assert "key rejected" in out
    assert "sk-bad-key" not in out


async def test_guided_brain_unreachable_then_retry_then_verified(
        tmp_path, monkeypatch, capsys):
    """timeout/5xx → ONE retry offer; second attempt verified → written."""
    flips = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/trial-auth"):
            return _mint_ok(request)
        if "/chat/completions" in str(request.url):
            flips["n"] += 1
            if flips["n"] == 1:
                return httpx.Response(500, text="upstream dead")
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"data": {"deliveries": []}})

    answers = iter(["sk-flip-key", "y"])        # paste, retry=yes

    def ask(prompt, **kw):
        return next(answers)

    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask", ask)
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _client(handler) as http:
        rc = await _run_setup(http=http)

    assert rc == 0
    text = (tmp_path / ".env").read_text()
    assert "NEAR_API_KEY=sk-flip-key\n" in text
    assert flips["n"] == 2
    assert "NEAR unreachable" in capsys.readouterr().out


async def test_guided_brain_enter_skips_verify_never_called(
        tmp_path, monkeypatch, capsys):
    """Enter at the paste prompt → no verify POST at all."""
    brain_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/trial-auth"):
            return _mint_ok(request)
        if "/chat/completions" in str(request.url):
            brain_calls["n"] += 1
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"data": {"deliveries": []}})

    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask", lambda prompt, **kw: "")
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _setup_client() as http:
        await _run_setup(http=http)

    assert brain_calls["n"] == 0
    assert "skipped" in capsys.readouterr().out


async def test_guided_prechecks_env_already_configured(tmp_path, monkeypatch,
                                                       capsys):
    """NEAR_API_KEY in env → ① skips paste+verify entirely; a set
    VIBE_TRIAL_TOKEN with Enter-default keeps it (no second mint)."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no network expected — everything preconfigured")

    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask",
                        lambda prompt, **kw: "n")    # refresh offer → keep
    monkeypatch.setenv("NEAR_API_KEY", "sk-from-env-8888")
    monkeypatch.setenv("VIBE_TRIAL_TOKEN", "trial_existing0000")
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _client(handler) as http:
        rc = await _run_setup(http=http)

    assert rc == 0
    out = capsys.readouterr().out
    assert "brain already configured in environment" in out
    assert "keeping the existing token" in out
    assert "sk-from-env-8888" not in out            # masked everywhere


async def test_guided_feed_verify_counts_open_signals(tmp_path, monkeypatch,
                                                      capsys):
    """② mint → feed 200 with 2 open rows → the count prints."""
    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask", lambda prompt, **kw: "")
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _setup_client() as http:
        await _run_setup(http=http)

    out = capsys.readouterr().out
    assert "feed reachable — 2 open signal(s)" in out
    assert "trial_ababab" in out                    # masked token in line+summary


# ------------------------------------------------------------ non-interactive


async def test_non_interactive_flag_mint_only(tmp_path, monkeypatch, capsys):
    """--non-interactive → the fire-and-forget mint even on a fake TTY;
    no prompt text anywhere, exit 0."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/trial-auth")
        return _mint_ok(request)

    monkeypatch.setattr(A, "_interactive", lambda: True)   # TTY, but flag wins
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _client(handler) as http:
        rc = await _run_setup(http=http, non_interactive=True)

    assert rc == 0
    text = (tmp_path / ".env").read_text()
    assert "VIBE_TRIAL_TOKEN=trial_ababab" in text
    assert "NEAR_API_KEY" not in text
    out = capsys.readouterr().out
    assert "next: uv run python -u -m warden.agent --dry" in out
    assert "①" not in out and "BRAIN" not in out


async def test_non_tty_stdin_degrades_to_mint_only(tmp_path, monkeypatch,
                                                   capsys):
    """stdin not a TTY (pipes/CI) → mint-only + the why-line, even though
    _interactive was overridden — this test drives the real isatty gate."""

    class FakeStdin:
        isatty = lambda self: False          # noqa: E731 — the gate itself

    monkeypatch.setattr(A.sys, "stdin", FakeStdin())
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _setup_client() as http:
        rc = await _run_setup(http=http)

    assert rc == 0
    text = (tmp_path / ".env").read_text()
    assert "VIBE_TRIAL_TOKEN=trial_ababab" in text
    out = capsys.readouterr().out
    assert "non-interactive shell" in out


async def test_setup_mint_failure_exit_1(tmp_path, monkeypatch, capsys):
    """Mint non-200 on the non-interactive path → exit 1, honest stderr,
    no token line written."""
    monkeypatch.setattr(A, "_interactive", lambda: False)

    def handler(request):
        return httpx.Response(503, text="down")

    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))
    async with _client(handler) as http:
        rc = await _run_setup(http=http, non_interactive=True)

    assert rc == 1
    err = capsys.readouterr().err
    assert "trial mint failed" in err
    assert not (tmp_path / ".env").exists()


# ------------------------------------------------------------ rerun idempotence


async def test_rerun_replaces_lines_not_duplicates(tmp_path, monkeypatch):
    """Second guided run: NEAR_API_KEY + VIBE_TRIAL_TOKEN lines replaced in
    place — one line each, other lines byte-identical."""
    ran = {"brain": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/trial-auth"):
            return _mint_ok(request)
        if "/chat/completions" in str(request.url):
            ran["brain"] += 1
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"data": {"deliveries": []}})

    keys = iter(["sk-first-key-1111", "sk-second-key-2222"])
    monkeypatch.setattr(A, "_interactive", lambda: True)
    monkeypatch.setattr(A, "_ask",
                        lambda prompt, **kw: "y" if "Retry" in prompt
                        else next(keys))
    monkeypatch.setenv("WARDEN_ENV_FILE", str(tmp_path / ".env"))

    async with _setup_client() as http:
        await _run_setup(http=http)
    text1 = (tmp_path / ".env").read_text()
    async with _setup_client() as http:
        await _run_setup(http=http)
    text2 = (tmp_path / ".env").read_text()

    assert "NEAR_API_KEY=sk-first-key-1111" in text1
    assert "NEAR_API_KEY=sk-second-key-2222" in text2
    assert "sk-first-key-1111" not in text2          # replaced, not appended
    assert text2.count("NEAR_API_KEY=") == 1
    assert text2.count("VIBE_TRIAL_TOKEN=") == 1
