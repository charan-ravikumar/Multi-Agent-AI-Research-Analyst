from app.secrets_env import export_secrets_to_env


def test_top_level_and_nested_and_lowercase_secrets_are_exported():
    env: dict[str, str] = {}
    names = export_secrets_to_env({
        "GROQ_API_KEY": "g",
        "general": {"gemini_api_key": "m", "langfuse": {"LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com"}},
        "TOKEN_BUDGET_PER_RUN": 60000,
        "ARXIV_ENABLED": False,
        "UNRELATED_SECRET": "x",
    }, env)
    assert env["GROQ_API_KEY"] == "g" and env["GEMINI_API_KEY"] == "m"
    assert env["LANGFUSE_BASE_URL"] == "https://us.cloud.langfuse.com"
    assert env["TOKEN_BUDGET_PER_RUN"] == "60000" and env["ARXIV_ENABLED"] == "false"
    assert "UNRELATED_SECRET" not in env and "UNRELATED_SECRET" not in names
    assert "g" not in names and "m" not in names            # names only, never values


def test_existing_values_win_but_empty_ones_are_filled():
    env = {"GROQ_API_KEY": "from-env", "GEMINI_API_KEY": ""}
    export_secrets_to_env({"GROQ_API_KEY": "from-secrets", "GEMINI_API_KEY": "filled"}, env)
    assert env == {"GROQ_API_KEY": "from-env", "GEMINI_API_KEY": "filled"}


def test_non_scalar_values_are_ignored():
    env: dict[str, str] = {}
    export_secrets_to_env({"GROQ_API_KEY": ["a", "b"], "LLM_PRICE_PER_1M_TOKENS": '{"m":[1,2]}'}, env)
    assert "GROQ_API_KEY" not in env and env["LLM_PRICE_PER_1M_TOKENS"] == '{"m":[1,2]}'
