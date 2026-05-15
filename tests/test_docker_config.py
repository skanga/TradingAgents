from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dockerignore_entries() -> set[str]:
    return {
        line.strip()
        for line in (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_dockerignore_excludes_local_secrets_and_generated_artifacts():
    entries = _dockerignore_entries()

    assert "auth.json" in entries
    assert ".env*" in entries
    assert "reports/" in entries
    assert "tradingagents/share/" in entries


def test_ollama_compose_profile_uses_tradingagents_env_and_service_url():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    lines = {line.strip() for line in compose.splitlines()}

    assert "- TRADINGAGENTS_LLM_PROVIDER=ollama" in lines
    assert "- TRADINGAGENTS_BACKEND_URL=http://ollama:11434/v1" in lines
    assert "- LLM_PROVIDER=ollama" not in lines


def test_compose_runtime_defaults_are_localhost_not_site_specific_nas():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "192.168.2.34" not in compose
    assert "- CORS_ORIGINS=${CORS_ORIGINS:-http://localhost:3000,http://127.0.0.1:3000}" in compose
    assert "- NEXT_PUBLIC_WS_BASE=${NEXT_PUBLIC_WS_BASE:-ws://localhost:8000}" in compose


def test_api_cors_defaults_are_localhost_only(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    from service.app import _allowed_origins

    origins = _allowed_origins()

    assert origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
