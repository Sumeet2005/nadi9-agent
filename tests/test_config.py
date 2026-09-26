from nadi9.config import Settings, get_settings


def test_default_settings():
    settings = get_settings()
    assert settings.env == "development"
    assert settings.llm_provider == "mock"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.max_model_calls == 25
    assert settings.retrieval_top_k == 5


def test_settings_environment_overrides(monkeypatch):
    monkeypatch.setenv("NADI9_ENV", "production")
    monkeypatch.setenv("NADI9_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")
    monkeypatch.setenv("NADI9_MAX_MODEL_CALLS", "50")
    monkeypatch.setenv("NADI9_DATABASE_URL", "sqlite:///./test_override.db")

    settings = Settings()

    assert settings.env == "production"
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "sk-env-test-key"
    assert settings.max_model_calls == 50
    assert settings.database_url == "sqlite:///./test_override.db"
