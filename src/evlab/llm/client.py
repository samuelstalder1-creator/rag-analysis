from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        ...


def build_llm_client(config: dict) -> LLMClient:
    provider = str(config.get("provider", "abstain"))
    if provider == "local_openai":
        from evlab.llm.local import LocalOpenAIClient

        return LocalOpenAIClient(
            base_url=str(config["base_url"]),
            model=str(config["model"]),
            api_key_env=str(config.get("api_key_env", "LOCAL_LLM_API_KEY")),
            timeout=float(config.get("timeout", 120)),
            max_retries=int(config.get("max_retries", 0)),
            retry_delay=float(config.get("retry_delay", 5)),
            max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
        )
    if provider == "ollama":
        from evlab.llm.local import OllamaClient

        return OllamaClient(
            base_url=str(config.get("base_url", "http://localhost:11434")),
            model=str(config["model"]),
            timeout=float(config.get("timeout", 120)),
            max_retries=int(config.get("max_retries", 0)),
            retry_delay=float(config.get("retry_delay", 5)),
            max_tokens=int(config["max_tokens"]) if config.get("max_tokens") else None,
            num_ctx=int(config["num_ctx"]) if config.get("num_ctx") else None,
            think=config["think"] if "think" in config else None,
        )
    if provider == "gemini":
        from evlab.llm.gemini import GeminiClient

        return GeminiClient(
            model=str(config["model"]),
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
        )
    if provider == "abstain":
        return AbstainClient()
    raise ValueError(f"Unsupported LLM provider: {provider}")


class AbstainClient:
    provider = "abstain"
    model = "abstain"

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        return "I don't know"
