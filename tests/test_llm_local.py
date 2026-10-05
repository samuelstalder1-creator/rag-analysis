import urllib.error

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
