from __future__ import annotations

import os


class GeminiClient:
    provider = "gemini"

    def __init__(self, *, model: str, api_key_env: str = "GEMINI_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key in ${self.api_key_env}")
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install Gemini support with: uv sync --extra gemini") from exc

        client = genai.Client(api_key=api_key)
        config_kwargs: dict[str, object] = {"temperature": temperature}
        if seed is not None:
            config_kwargs["seed"] = seed
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return str(response.text or "").strip()
