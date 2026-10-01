from __future__ import annotations

import json
import os
import urllib.request


class LocalOpenAIClient:
    provider = "local_openai"

    def __init__(self, *, base_url: str, model: str, api_key_env: str = "LOCAL_LLM_API_KEY", timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(self.api_key_env, 'local')}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()


class OllamaClient:
    provider = "ollama"

    def __init__(self, *, base_url: str, model: str, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        options: dict[str, object] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "options": options}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", "")).strip()
