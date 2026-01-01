"""Small end-to-end API for command-line and embedded inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import build_evidence_graph
from .reasoning import CentralizedAgent
from .repository import KnowledgeRepository


class OptSage:
    """End-to-end facade used by the inference CLI and embedding applications."""

    def __init__(self, repository: KnowledgeRepository, **agent_options: Any) -> None:
        self.repository = repository
        self.agent = CentralizedAgent(repository, **agent_options)

    def diagnose(self, case: Mapping[str, Any]) -> dict[str, Any]:
        graph = build_evidence_graph(case)
        result = self.agent.diagnose(graph)
        result["evidence_graph"] = graph.to_dict()
        return result


def load_sample(path: str | Path) -> tuple[KnowledgeRepository, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "optsage-sample-v1":
        raise ValueError("unsupported sample schema")
    repository = KnowledgeRepository.from_cases(payload.get("history") or [])
    return repository, list(payload.get("queries") or [])
