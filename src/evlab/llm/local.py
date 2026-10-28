from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought from reasoning models (e.g. deepseek-r1).

    Complete ``<think>...</think>`` blocks are dropped. An unterminated block
    (output cut off by the token limit) yields an empty string so that the
    pipeline rejects the attempt instead of storing reasoning as a document.
    """
    text = _THINK_BLOCK.sub("", text)
    if re.search(r"<think>", text, re.IGNORECASE):
        return ""
    # R1-style chat templates add "<think>" to the prompt, so the completion
    # holds the reasoning followed by a lone "</think>": keep only what follows.
    if re.search(r"</think>", text, re.IGNORECASE):
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


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
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def request_config(self) -> dict[str, object]:
        """Request settings that change the response; part of the cache key."""
        return {"max_tokens": self.max_tokens}

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(self.api_key_env, 'local')}",
                # Hosted APIs behind Cloudflare (e.g. Together AI) reject urllib's default agent.
                "User-Agent": "evlab/1.0",
            },
            method="POST",
        )
        data = self._open_json(request)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            return ""  # Incomplete documents must go through rejection/retry.
        content = choice["message"].get("content") or ""
        if len(re.findall(r"[ĠĊ]", content)) >= 3:
            raise ValueError(
                "LLM output contains undecoded ByteLevel tokenizer markers. "
                "Restart DeepSeek with the tokenizer prepared by "
                "scripts/prepare_deepseek_tokenizer.py; use a fresh generation cache."
            )
        return strip_reasoning(content.strip())

    def _open_json(self, request: urllib.request.Request) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    # Rate limited (hosted APIs): wait as long as the server asks.
                    try:
                        wait = float(exc.headers.get("Retry-After", ""))
                    except (TypeError, ValueError):
                        wait = self.retry_delay * (attempt + 1)
                    time.sleep(min(max(wait, 1.0), 120.0))
                    continue
                if exc.code < 500 or attempt >= self.max_retries:
                    # Keep the server's explanation (e.g. context length) in the error.
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:500]
                    except Exception:
                        detail = ""
                    if not detail:
                        raise
                    raise urllib.error.HTTPError(
                        exc.url, exc.code, f"{exc.reason}: {detail}", exc.headers, None
                    ) from exc
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
            time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("unreachable retry state")


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120,
        max_retries: int = 0,
        retry_delay: float = 5.0,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        think: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens  # -> options.num_predict (includes reasoning tokens)
        self.num_ctx = num_ctx  # Ollama defaults to a small context window
        self.think = think  # None: do not send; True/False: Ollama "think" flag

    def request_config(self) -> dict[str, object]:
        """Request settings that change the response; part of the cache key."""
        return {"max_tokens": self.max_tokens, "num_ctx": self.num_ctx, "think": self.think}

    def generate(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        options: dict[str, object] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        payload: dict[str, object] = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        if self.think is not None:
            payload["think"] = self.think
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data = self._open_json(request)
        return strip_reasoning(str(data.get("response", "")).strip())

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
