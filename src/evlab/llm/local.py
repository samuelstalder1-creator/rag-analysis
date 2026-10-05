from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class LocalOpenAIClient:
    provider = "local_openai"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "LOCAL_LLM_API_KEY",
        timeout: float = 120,
        max_retries: int = 0,
        retry_delay: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
        data = self._open_json(request)
        return str(data["choices"][0]["message"]["content"]).strip()

    def _open_json(self, request: urllib.request.Request) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code < 500 or attempt >= self.max_retries:
                    raise
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("unreachable retry state")


class OllamaClient:
    provider = "ollama"

    def __init__(self, *, base_url: str, model: str, timeout: float = 120, max_retries: int = 0, retry_delay: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
        data = self._open_json(request)
        return str(data.get("response", "")).strip()

    def _open_json(self, request: urllib.request.Request) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code < 500 or attempt >= self.max_retries:
                    raise
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("unreachable retry state")
