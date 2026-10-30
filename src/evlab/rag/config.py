"""The reader (answer-generating LLM) section of an experiment config.

::

    reader:
      llm: configs/readers/qwen.yaml   # preset file ...
      # llm:                           # ... or inline
      #   provider: local_openai       # local_openai | ollama | gemini | abstain
      #   base_url: http://localhost:8000/v1
      #   model: Qwen/Qwen2.5-7B-Instruct
      #   max_tokens: 256
      temperature: 0.0
      seed: null
      top_k: 10                        # retrieved documents placed in the context
      context_order: retrieval         # retrieval | reverse, after selecting top_k
      prompt: prompts/reader_v1.txt    # template file with {question} and {context}
      retrievers: [bm25]               # which retrievers feed the reader (default: all)
      cache_path: cache/reader.sqlite

``reader`` itself may also be a path to a file with these keys. The reader runs
when ``reader.enabled`` is true, which by default means ``queries: rag``.
"""

from __future__ import annotations

from pathlib import Path

from evlab.config import load_config
from evlab.io import stable_hash
from evlab.rag.reader import READER_PROMPT

READER_KEYS = {"enabled", "llm", "temperature", "seed", "top_k", "prompt", "cache_path", "retrievers", "context_order"}
LLM_KEYS = {
    "provider", "base_url", "model", "api_key_env", "timeout",
    "max_retries", "retry_delay", "max_tokens",
}
PROVIDERS = {"local_openai", "ollama", "gemini", "abstain"}


def resolve_reader(config: dict, run_names: list[str]) -> dict | None:
    raw = config.get("reader") or {}
    if isinstance(raw, str):
        raw = load_config(raw)
    raw = dict(raw)
    unknown = set(raw) - READER_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in reader: {sorted(unknown)}; allowed: {sorted(READER_KEYS)}")
    if not raw.get("enabled", config.get("queries") == "rag"):
        return None
    llm = raw.get("llm")
    if llm is None:
        raise ValueError("RAG evaluation requires reader.llm in the experiment config")
    llm_source = "inline"
    if isinstance(llm, str):
        llm_source = llm
        loaded = load_config(llm)
        llm = loaded.get("llm", loaded)
    llm = dict(llm)
    unknown = set(llm) - LLM_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in reader.llm: {sorted(unknown)}; allowed: {sorted(LLM_KEYS)}")
    provider = llm.get("provider", "abstain")
    if provider not in PROVIDERS:
        raise ValueError(f"reader.llm.provider must be one of {sorted(PROVIDERS)}, got {provider!r}")
    if provider != "abstain" and not llm.get("model"):
        raise ValueError("reader.llm.model is required")
    if provider == "local_openai" and not llm.get("base_url"):
        raise ValueError("reader.llm.base_url is required for provider local_openai")

    prompt_source, template = _prompt(raw.get("prompt"))
    retrievers = raw.get("retrievers") or list(run_names)
    if isinstance(retrievers, str):
        retrievers = [retrievers]
    missing = [name for name in retrievers if name not in run_names]
    if missing:
        raise ValueError(f"reader.retrievers {missing} are not in retrievers {run_names}")
    top_k = int(raw.get("top_k", 10))
    if top_k < 1:
        raise ValueError("reader.top_k must be >= 1")
    context_order = str(raw.get("context_order", "retrieval"))
    if context_order not in {"retrieval", "reverse"}:
        raise ValueError("reader.context_order must be retrieval or reverse")
    return {
        **({"context_order": context_order} if "context_order" in raw else {}),
        "llm": llm,
        "llm_source": llm_source,
        "temperature": float(raw.get("temperature", 0.0)),
        "seed": None if raw.get("seed") is None else int(raw["seed"]),
        "top_k": top_k,
        "prompt_source": prompt_source,
        "prompt_template": template,
        "prompt_hash": stable_hash(template)[:12],
        "retrievers": list(retrievers),
        "cache_path": str(raw.get("cache_path", "cache/reader.sqlite")),
    }


def reader_summary(spec: dict | None) -> dict | None:
    if spec is None:
        return None
    return {
        **({"context_order": spec["context_order"]} if "context_order" in spec else {}),
        "provider": spec["llm"].get("provider", "abstain"),
        "model": spec["llm"].get("model"),
        "llm_source": spec["llm_source"],
        "temperature": spec["temperature"],
        "seed": spec["seed"],
        "top_k": spec["top_k"],
        "prompt": spec["prompt_source"],
        "prompt_hash": spec["prompt_hash"],
        "retrievers": spec["retrievers"],
    }


def _prompt(value: object) -> tuple[str, str]:
    if value is None:
        return "builtin:reader_v1", READER_PROMPT
    text = str(value)
    if "{question}" in text:
        source, template = "inline", text
    else:
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"reader.prompt file not found: {path}")
        source, template = text, path.read_text(encoding="utf-8")
    for field in ("{question}", "{context}"):
        if field not in template:
            raise ValueError(f"reader.prompt must contain {field}")
    try:
        template.format(question="q", context="c")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            "reader.prompt has unescaped braces; write literal { } as {{ }}"
        ) from exc
    return source, template
