import urllib.error

import json
import pytest

from evlab.llm.local import LocalOpenAIClient


def test_local_openai_retries_url_errors(monkeypatch):
    calls = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError(ConnectionRefusedError("refused"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    client = LocalOpenAIClient(base_url="http://localhost:8000/v1", model="m", max_retries=1)

    assert client.generate("prompt") == "ok"
    assert calls["count"] == 2


def _fake_response(payload: bytes):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return payload

    return FakeResponse()


def test_strip_reasoning_removes_think_blocks():
    from evlab.llm.local import strip_reasoning

    assert strip_reasoning("<think>hmm\nmore</think>\nAnswer text.") == "Answer text."
    assert strip_reasoning("Plain text.") == "Plain text."
    # Opening tag lives in the prompt (vLLM + R1 template): only "</think>" appears.
    assert strip_reasoning("reasoning without opening tag</think>\n\nFinal text.") == "Final text."
    # Cut off inside the reasoning block: nothing usable, must not leak the thoughts.
    assert strip_reasoning("<think>still thinking about it") == ""


def test_ollama_sends_options_and_strips_reasoning(monkeypatch):
    import json

    from evlab.llm.client import build_llm_client

    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_response(b'{"response":"<think>x</think>Result."}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = build_llm_client(
        {"provider": "ollama", "model": "deepseek-r1:8b", "max_tokens": 4096, "num_ctx": 8192, "think": True}
    )

    assert client.generate("p", temperature=0.3, seed=7) == "Result."
    body = seen["body"]
    assert body["think"] is True
    assert body["options"] == {"temperature": 0.3, "seed": 7, "num_predict": 4096, "num_ctx": 8192}


def test_ollama_omits_unset_options(monkeypatch):
    import json

    from evlab.llm.client import build_llm_client

    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _fake_response(b'{"response":"ok"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    build_llm_client({"provider": "ollama", "model": "m"}).generate("p")

    assert "think" not in seen["body"]
    assert seen["body"]["options"] == {"temperature": 0.0}


def test_local_openai_rejects_corrupt_tokenizer_output(monkeypatch):
    payload = json.dumps({"choices": [{"message": {"content": "ĊDonĠHenleyĠsangĠinĠ1994."}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _fake_response(payload))
    client = LocalOpenAIClient(base_url="http://localhost:8000/v1", model="m")
    with pytest.raises(ValueError, match="ByteLevel tokenizer markers"):
        client.generate("prompt")


@pytest.mark.parametrize("content", [None, "Don Henley sang"])
def test_local_openai_rejects_truncated_documents(monkeypatch, content):
    payload = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": content}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _fake_response(payload))
    client = LocalOpenAIClient(base_url="http://localhost:8000/v1", model="m")
    assert client.generate("prompt") == ""


def test_local_openai_null_content_is_empty(monkeypatch):
    payload = b'{"choices":[{"message":{"content":null}}]}'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _fake_response(payload))
    client = LocalOpenAIClient(base_url="http://localhost:8000/v1", model="m")
    assert client.generate("prompt") == ""
