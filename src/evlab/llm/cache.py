from __future__ import annotations

import sqlite3
from pathlib import Path

from evlab.io import stable_hash
from evlab.llm.client import LLMClient


class CachedLLMClient:
    def __init__(self, client: LLMClient, cache_path: str | Path) -> None:
        self.client = client
        self.provider = client.provider
        self.model = client.model
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def generate(
        self, prompt: str, *, temperature: float = 0.0, seed: int | None = None
    ) -> str:
        payload = {
            "provider": self.provider,
            "endpoint": getattr(self.client, "base_url", None),
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
        }
        # Settings such as max_tokens change the answer. They only enter the key when
        # set, so clients without such settings keep their existing cache entries.
        request = self._request_config()
        if request:
            payload["request"] = request
        key = stable_hash(payload)
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "select response from llm_cache where cache_key = ?", (key,)
            ).fetchone()
            if row:
                return str(row[0])
            response = self.client.generate(prompt, temperature=temperature, seed=seed)
            conn.execute(
                "insert or ignore into llm_cache(cache_key, provider, model, prompt, response) values (?, ?, ?, ?, ?)",
                (key, self.provider, self.model, prompt, response),
            )
            return response

    def _request_config(self) -> dict:
        describe = getattr(self.client, "request_config", None)
        config = describe() if callable(describe) else {}
        return {key: value for key, value in config.items() if value is not None}

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                create table if not exists llm_cache (
                    cache_key text primary key,
                    provider text not null,
                    model text not null,
                    prompt text not null,
                    response text not null,
                    created_at datetime default current_timestamp
                )
                """
            )
