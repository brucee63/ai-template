import services.llm_factory as llm_factory

def load_llm_factory():
    # Load the settings
    settings = llm_factory.get_settings()
    print(f"opeanai api_key set: {settings.openai.api_key is not None}")
    print(f"default model: {settings.openai.default_model}")

if __name__ == "__main__":
    load_llm_factory()    