from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Origin = Literal["human", "cocktail_llm", "generated"]
Condition = Literal["correct", "false"]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    origin: Origin
    dataset: str = "nqplus"
    query_id: str | None = None
    condition: Condition | None = None
    gold_answer: str | None = None
    target_answer: str | None = None
    parent_id: str | None = None
    seed_id: str | None = None
    provenance_root_id: str | None = None
    generation_depth: int = 0
    generator_model: str | None = None
    prompt_version: str | None = None
    sample_idx: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        return f"{self.title} {self.text}".strip()

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title.strip() else self.text.strip()

    @property
    def root_id(self) -> str:
        return self.provenance_root_id or original_id(self.doc_id)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    answer: str | None = None
    aliases: tuple[str, ...] = ()
    relevant_doc_ids: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int = 1


@dataclass(frozen=True)
class Hit:
    query_id: str
    doc_id: str
    rank: int
    score: float
    retriever: str


@dataclass(frozen=True)
class ReaderAnswer:
    query_id: str
    answer: str
    citations: tuple[str, ...] = ()
    label: Literal["correct", "target", "abstain", "other"] = "other"


def namespaced_id(namespace: str, original: str) -> str:
    return f"{namespace}::{original}"


def original_id(doc_id: str) -> str:
    return doc_id.split("::", 1)[1] if "::" in doc_id else doc_id
