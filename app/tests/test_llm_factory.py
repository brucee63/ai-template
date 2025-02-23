import services.llm_factory as llm_factory

def test_get_settings():
    settings = llm_factory.get_settings()
    assert settings.openai.api_key is not None
    assert settings.openai.default_model == "gpt-4o"
    assert settings.openai.embedding_model == "text-embedding-3-small"
    assert settings.database.service_url is not None
    assert settings.vector_store.table_name == "embeddings"
    assert settings.vector_store.embedding_dimensions == 1536
    assert settings.vector_store.time_partition_interval.days == 7
    assert settings.openai.temperature == 0.0
    assert settings.openai.max_retries == 3
    assert settings.openai.max_tokens is None

