from prompts.prompt_manager import PromptManager

def test_prompts():
    test_prompt = PromptManager.get_prompt("sample", test_value="World")
    assert test_prompt == "Hello World!"

    template_info = PromptManager.get_template_info("sample")
    assert template_info == {
        "name": "sample",
        "description": "This is a test prompt",
        "author": "Test Author",
        "variables": ["test_value"],
        "frontmatter": {"description": "This is a test prompt", "author": "Test Author"},
    }

def test_prompts_cond():
    test_prompt = PromptManager.get_prompt("sample_cond", test_value="World")
    assert test_prompt == "\nHello World!\n"

    test_prompt = PromptManager.get_prompt("sample_cond", test_value="Goodbye")
    assert test_prompt == "\nNot Hello World!\n"

    template_info = PromptManager.get_template_info("sample_cond")
    assert template_info == {
        "name": "sample_cond",
        "description": "This is a test prompt with condition",
        "author": "Test Author",
        "variables": ["test_value"],
        "frontmatter": {"description": "This is a test prompt with condition", "author": "Test Author"},
    }
