"""Stable schemas and immutable value objects shared across OptSage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


ROOT_CAUSES = ("L1", "L2", "fiber")
GRAPH_SCHEMA = "optsage-evidence-graph-v1"
REPOSITORY_SCHEMA = "optsage-knowledge-v1"
SYSTEM_VERSION = "paper-mvp-v4"
CONSTRAINT_VERSION = "physical-constraints-v1"
PAPER_DEFAULT_TOP_K = 3
PAPER_DEFAULT_TAU = 0.70
PAPER_DEFAULT_EPSILON = 0.70
JUDGE_DIMENSIONS = (
    "evidence_completeness",
    "constraint_compliance",
    "reasoning_validity",
)


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceGraph:
    case_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    evidence_tokens: tuple[str, ...]
    semantic_edges: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    topology: Mapping[str, Any]
    context: Mapping[str, Any]
    coverage: float
    schema_version: str = GRAPH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "nodes": [asdict(item) for item in self.nodes],
            "edges": [asdict(item) for item in self.edges],
            "evidence_tokens": list(self.evidence_tokens),
            "semantic_edges": list(self.semantic_edges),
            "missing_tokens": list(self.missing_tokens),
            "topology": dict(self.topology),
            "context": dict(self.context),
            "coverage": self.coverage,
        }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class HistoricalCase:
    case_id: str
    root_cause: str
    graph: EvidenceGraph
    confirmed_by: str
    confirmation_time: str = ""
    reasoning_chain: Mapping[str, Any] = field(default_factory=dict)
    sops: tuple[str, ...] = ()
    source_version: str = "initial"
    applicable_pattern: str = "all"
    validation_result: str = "operator_confirmed"
    status: str = "active"
    merged_case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "root_cause": self.root_cause,
            "confirmed_by": self.confirmed_by,
            "confirmation_time": self.confirmation_time,
            "reasoning_chain": dict(self.reasoning_chain),
            "sops": list(self.sops),
            "source_version": self.source_version,
            "applicable_pattern": self.applicable_pattern,
            "validation_result": self.validation_result,
            "status": self.status,
            "merged_case_ids": list(self.merged_case_ids),
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True)
class Candidate:
    case_id: str
    root_cause: str
    evidence_similarity: float
    graph_similarity: float
    topology_compatible: bool
    shared_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    extra_evidence: tuple[str, ...]
    mutually_exclusive_evidence: tuple[str, ...]
    reusable_sops: tuple[str, ...]
    reusable_reasoning_chain: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
