from config.settings import get_settings

def test_openai_settings():
    settings = get_settings()
    assert settings.openai.api_key is not None
    assert settings.openai.default_model == "gpt-4o"
    assert settings.openai.embedding_model == "text-embedding-3-small"
