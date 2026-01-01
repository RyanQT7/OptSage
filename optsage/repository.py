"""Versioned historical evidence repository and dual-view retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .evidence import build_evidence_graph
from .models import (
    REPOSITORY_SCHEMA,
    ROOT_CAUSES,
    Candidate,
    EvidenceGraph,
    HistoricalCase,
    _digest,
)


class KnowledgeRepository:
    """Immutable, versioned collection of operator-confirmed graph instances."""

    def __init__(
        self,
        cases: Sequence[HistoricalCase] = (),
        *,
        parent_version: str = "",
        heuristic_versions: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.cases = tuple(cases)
        self.parent_version = parent_version
        self.heuristic_versions = tuple(dict(item) for item in heuristic_versions)
        active_cases = tuple(item for item in self.cases if item.status == "active")
        self.idf = self._idf(item.graph.evidence_tokens for item in active_cases)
        self.graph_idf = self._idf(item.graph.semantic_edges for item in active_cases)
        self.version = _digest({
            "parent_version": parent_version,
            "cases": [item.to_dict() for item in self.cases],
            "heuristics": self.heuristic_versions,
        })

    @staticmethod
    def _idf(documents: Iterable[Sequence[str]]) -> dict[str, float]:
        docs = [set(item) for item in documents]
        frequency = Counter(token for doc in docs for token in doc)
        count = len(docs)
        return {
            token: math.log((count + 1) / (freq + 1)) + 1.0
            for token, freq in frequency.items()
        }

    @classmethod
    def from_cases(cls, cases: Sequence[Mapping[str, Any]]) -> "KnowledgeRepository":
        rows = []
        for case in cases:
            root_cause = str(case.get("confirmed_root_cause") or "")
            if root_cause not in ROOT_CAUSES:
                raise ValueError(f"invalid confirmed root cause: {root_cause!r}")
            inference_case = {key: value for key, value in case.items() if key != "confirmed_root_cause"}
            rows.append(HistoricalCase(
                case_id=str(case["case_id"]),
                root_cause=root_cause,
                graph=build_evidence_graph(inference_case),
                confirmed_by=str(case.get("confirmed_by") or "operator"),
                confirmation_time=str(case.get("confirmation_time") or ""),
                reasoning_chain=dict(case.get("reasoning_chain") or {}),
                sops=tuple(str(item) for item in case.get("sops") or ()),
                source_version=str(case.get("source_version") or "initial"),
                applicable_pattern=str(case.get("applicable_pattern") or "all"),
                validation_result=str(case.get("validation_result") or "operator_confirmed"),
            ))
        return cls(rows)

    def _weighted_jaccard(self, left: Sequence[str], right: Sequence[str], weights: Mapping[str, float]) -> float:
        a, b = set(left), set(right)
        union = a | b
        if not union:
            return 0.0
        active_count = sum(item.status == "active" for item in self.cases)
        unseen_weight = math.log(active_count + 1) + 1.0
        numerator = sum(weights.get(item, unseen_weight) for item in sorted(a & b))
        denominator = sum(weights.get(item, unseen_weight) for item in sorted(union))
        return round(numerator / denominator, 6) if denominator else 0.0

    def search(self, graph: EvidenceGraph, top_k: int = 3) -> tuple[Candidate, ...]:
        rows = []
        online_tokens = set(graph.evidence_tokens)
        for item in self.cases:
            if item.status != "active":
                continue
            historical_tokens = set(item.graph.evidence_tokens)
            online_by_slot = {token.rsplit(":", 1)[0]: token for token in online_tokens}
            exclusive: list[str] = []
            for historical in historical_tokens:
                slot = historical.rsplit(":", 1)[0]
                if slot in online_by_slot and online_by_slot[slot] != historical:
                    exclusive.extend((historical, online_by_slot[slot]))
            compatible = all(
                not graph.topology.get(key)
                or not item.graph.topology.get(key)
                or graph.topology[key] == item.graph.topology[key]
                for key in ("network_layer", "lane_count")
            )
            rows.append(Candidate(
                case_id=item.case_id,
                root_cause=item.root_cause,
                evidence_similarity=self._weighted_jaccard(
                    graph.evidence_tokens, item.graph.evidence_tokens, self.idf
                ),
                graph_similarity=self._weighted_jaccard(
                    graph.semantic_edges, item.graph.semantic_edges, self.graph_idf
                ),
                topology_compatible=compatible,
                shared_evidence=tuple(sorted(online_tokens & historical_tokens)),
                missing_evidence=tuple(sorted(historical_tokens - online_tokens)),
                extra_evidence=tuple(sorted(online_tokens - historical_tokens)),
                mutually_exclusive_evidence=tuple(sorted(set(exclusive))),
                reusable_sops=item.sops,
                reusable_reasoning_chain=dict(item.reasoning_chain),
            ))
        rows.sort(key=lambda item: (
            item.topology_compatible,
            min(item.evidence_similarity, item.graph_similarity),
            item.evidence_similarity + item.graph_similarity,
            item.case_id,
        ), reverse=True)
        compatible_rows = [item for item in rows if item.topology_compatible]
        pool = compatible_rows or rows
        return tuple(pool[: max(1, int(top_k))])

    def evolve(
        self,
        case: Mapping[str, Any],
        *,
        confirmed_by: str,
        applicable_pattern: str = "all",
        validation_result: str = "operator_confirmed",
    ) -> "KnowledgeRepository":
        """Return a new repository only for an independently confirmed case."""
        if not confirmed_by.strip():
            raise ValueError("operator identity is required for knowledge evolution")
        root_cause = str(case.get("confirmed_root_cause") or "")
        if root_cause not in ROOT_CAUSES:
            raise ValueError("confirmed_root_cause must be L1, L2, or fiber")
        if any(item.case_id == str(case.get("case_id")) for item in self.cases):
            raise ValueError("case_id already exists; existing versions are never overwritten")
        inference_case = {key: value for key, value in case.items() if key != "confirmed_root_cause"}
        row = HistoricalCase(
            case_id=str(case["case_id"]), root_cause=root_cause,
            graph=build_evidence_graph(inference_case), confirmed_by=confirmed_by,
            confirmation_time=datetime.now(timezone.utc).isoformat(),
            reasoning_chain=dict(case.get("reasoning_chain") or {}),
            sops=tuple(str(item) for item in case.get("sops") or ()),
            source_version=self.version,
            applicable_pattern=applicable_pattern,
            validation_result=validation_result,
        )
        return KnowledgeRepository(
            self.cases + (row,),
            parent_version=self.version,
            heuristic_versions=self.heuristic_versions,
        )

    @staticmethod
    def _pattern_key(case: HistoricalCase) -> str:
        return _digest({
            "topology": case.graph.topology,
            "evidence_tokens": case.graph.evidence_tokens,
            "semantic_edges": case.graph.semantic_edges,
        })

    def consolidate(self) -> "KnowledgeRepository":
        """Deduplicate equal confirmed patterns without hiding label conflicts."""
        retained: list[HistoricalCase] = []
        positions: dict[tuple[str, str], int] = {}
        for case in self.cases:
            identity = (self._pattern_key(case), case.root_cause)
            if identity not in positions:
                positions[identity] = len(retained)
                retained.append(case)
            else:
                index = positions[identity]
                original = retained[index]
                retained[index] = HistoricalCase(
                    **{
                        **original.__dict__,
                        "merged_case_ids": tuple(sorted(set(
                            original.merged_case_ids + (case.case_id,) + case.merged_case_ids
                        ))),
                    }
                )
        return KnowledgeRepository(
            retained,
            parent_version=self.version,
            heuristic_versions=self.heuristic_versions,
        )

    def refine_heuristics(
        self,
        reviewed_failures: Sequence[Mapping[str, Any]],
        *,
        effective_month: str,
    ) -> "KnowledgeRepository":
        """Create an auditable monthly heuristic revision from reviewed failures."""
        if len(effective_month) != 7 or effective_month[4] != "-":
            raise ValueError("effective_month must use YYYY-MM")
        causes = Counter(str(item.get("error_cause") or "unspecified") for item in reviewed_failures)
        revision = {
            "effective_month": effective_month,
            "source_version": self.version,
            "reviewed_count": len(reviewed_failures),
            "error_cause_counts": dict(sorted(causes.items())),
            "validation_result": "requires_offline_validation",
        }
        return KnowledgeRepository(
            self.cases,
            parent_version=self.version,
            heuristic_versions=self.heuristic_versions + (revision,),
        )

    def deprecate_case(self, case_id: str, *, validation_result: str) -> "KnowledgeRepository":
        """Retain an outdated instance for rollback while removing it from retrieval."""
        if not validation_result.strip():
            raise ValueError("validation_result is required")
        found = False
        rows = []
        for case in self.cases:
            if case.case_id == case_id:
                found = True
                case = HistoricalCase(**{
                    **case.__dict__,
                    "status": "deprecated",
                    "validation_result": validation_result,
                })
            rows.append(case)
        if not found:
            raise ValueError("unknown case_id")
        return KnowledgeRepository(
            rows,
            parent_version=self.version,
            heuristic_versions=self.heuristic_versions,
        )

    def pattern_audit(self) -> list[dict[str, Any]]:
        """Expose pure and mixed-label pattern buckets for operator review."""
        buckets: dict[str, list[HistoricalCase]] = defaultdict(list)
        for case in self.cases:
            buckets[self._pattern_key(case)].append(case)
        return [{
            "pattern_id": pattern_id,
            "case_ids": [item.case_id for item in rows],
            "root_causes": sorted({item.root_cause for item in rows}),
            "applicable_patterns": sorted({item.applicable_pattern for item in rows}),
            "statuses": sorted({item.status for item in rows}),
            "mixed_label": len({item.root_cause for item in rows}) > 1,
        } for pattern_id, rows in sorted(buckets.items())]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPOSITORY_SCHEMA,
            "version": self.version,
            "parent_version": self.parent_version,
            "cases": [item.to_dict() for item in self.cases],
            "pattern_audit": self.pattern_audit(),
            "heuristic_versions": list(self.heuristic_versions),
        }
