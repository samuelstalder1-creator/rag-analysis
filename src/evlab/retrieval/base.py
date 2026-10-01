from __future__ import annotations

from typing import Protocol, Sequence

from evlab.schema import Doc, Hit, Query


class Retriever(Protocol):
    name: str

    def index(self, docs: Sequence[Doc]) -> None:
        ...

    def search(self, queries: Sequence[Query], top_k: int) -> dict[str, list[Hit]]:
        ...
