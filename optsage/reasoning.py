"""Three-way routing, constrained reasoning, judging, and abstention."""

from __future__ import annotations

import json
import math
import urllib.request
from collections import Counter
from typing import Any, Mapping, Protocol, Sequence

from .constraints import (
    _admissible_relations,
    _constraint_analysis,
    _permitted_constraints,
)
from .models import (
    CONSTRAINT_VERSION,
    JUDGE_DIMENSIONS,
    PAPER_DEFAULT_EPSILON,
    PAPER_DEFAULT_TAU,
    PAPER_DEFAULT_TOP_K,
    ROOT_CAUSES,
    SYSTEM_VERSION,
    Candidate,
    EvidenceGraph,
    _canonical,
)
from .repository import KnowledgeRepository


def _route(candidates: Sequence[Candidate], tau: float) -> tuple[str, str]:
    if not candidates:
        return "novel", "no historical case"
    compatible = [item for item in candidates if item.topology_compatible] or list(candidates)
    complete = [
        item for item in compatible
        if item.evidence_similarity == 1.0 and item.graph_similarity == 1.0
    ]
    if complete and len({item.root_cause for item in complete}) == 1:
        return "common", "complete dual-view match with a pure historical signature"
    partial = [
        item for item in compatible
        if item.evidence_similarity >= tau and item.graph_similarity >= tau
    ]
    if partial or complete:
        return "ambiguous", "partial dual-view match or mixed historical signature"
    return "novel", "historical similarity is below the routing threshold"


def _critical_missing_evidence(candidates: Sequence[Candidate]) -> tuple[str, ...]:
    """Missing history is critical only if it changes a support/exclusion relation."""
    if not candidates:
        return ()
    result = set()
    for token in candidates[0].missing_evidence:
        relations = _admissible_relations(token)
        if any(value in {"support", "contradict"} for value in relations.values()):
            result.add(token)
    return tuple(sorted(result))


def _novel_sop(graph: EvidenceGraph) -> tuple[str, ...]:
    """Versioned inspection order used when historical evidence is insufficient."""
    steps = [
        "SOP1 enumerate L1, L2, and fiber hypotheses",
        "SOP2 exclude hypotheses contradicted by deterministic physical relations",
        "SOP3 inspect same-lane directional evidence across both directions",
    ]
    if graph.missing_tokens:
        steps.append("SOP4 request unavailable observations that can change admissibility")
    steps.append("SOP5 emit only evidence- and constraint-grounded reasoning steps")
    return tuple(steps)


def _geometric_confidence(values: Mapping[str, float]) -> float:
    clipped = [max(0.0, min(1.0, float(item))) for item in values.values()]
    if not clipped or any(item == 0 for item in clipped):
        return 0.0
    return round(math.prod(clipped) ** (1.0 / len(clipped)), 6)


class StructuredReasoner(Protocol):
    """Interface for an optional LLM that returns grounded structured JSON."""

    def reason(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class StructuredJudger(Protocol):
    """Independent lightweight judge for the three dimensions in Eq. (2)."""

    def judge(self, payload: Mapping[str, Any]) -> Mapping[str, float]:
        ...


class OpenAICompatibleReasoner:
    """Small OpenAI-compatible adapter; no SDK dependency is required."""

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def reason(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        protocol = {
            "verdict": "L1|L2|fiber",
            "reasoning_steps": [{
                "claim": "concise diagnostic relation",
                "cited_evidence": ["exact evidence token"],
                "cited_constraint": "P1|P2|P3|P4",
            }],
        }
        prompt = (
            "Act as the centralized OptSage reasoner. Return exactly one JSON object. "
            "Every step must cite an exact supplied evidence token and one allowed physical "
            f"constraint. Protocol: {_canonical(protocol)} Input: {_canonical(payload)}"
        )
        body = _canonical({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            message = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        value = json.loads(message)
        if not isinstance(value, Mapping):
            raise ValueError("reasoner response must be a JSON object")
        return value


class OpenAICompatibleJudger:
    """LLM judger kept separate from the primary reasoner as required by §V-C.1."""

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def judge(self, payload: Mapping[str, Any]) -> Mapping[str, float]:
        protocol = {dimension: "number in [0,1]" for dimension in JUDGE_DIMENSIONS}
        prompt = (
            "Act as the independent OptSage confidence judger. Return exactly one JSON object. "
            "Score evidence completeness, constraint compliance, and reasoning validity without "
            "changing the root-cause verdict. "
            f"Protocol: {_canonical(protocol)} Input: {_canonical(payload)}"
        )
        body = _canonical({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            message = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        value = json.loads(message)
        return _validate_judger_output(value)


def _validate_judger_output(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(JUDGE_DIMENSIONS):
        raise ValueError("judger must return exactly the three Eq. (2) dimensions")
    result: dict[str, float] = {}
    for dimension in JUDGE_DIMENSIONS:
        score = _number(value[dimension])
        if score is None or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid judger score for {dimension}")
        result[dimension] = score
    return result


def _validate_reasoner_output(value: Mapping[str, Any], evidence: Sequence[str]) -> list[str]:
    errors: list[str] = []
    if value.get("verdict") not in ROOT_CAUSES:
        errors.append("verdict must be L1, L2, or fiber")
    steps = value.get("reasoning_steps")
    if not isinstance(steps, list) or not steps:
        return errors + ["reasoning_steps must be a non-empty list"]
    allowed_evidence = set(evidence)
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            errors.append(f"step {index} must be an object")
            continue
        cited = step.get("cited_evidence")
        if not isinstance(cited, list) or not cited or not set(cited) <= allowed_evidence:
            errors.append(f"step {index} cites unavailable evidence")
            cited = []
        constraint_id = step.get("cited_constraint")
        if constraint_id not in {"P1", "P2", "P3", "P4"}:
            errors.append(f"step {index} cites an unknown constraint")
        elif cited and not any(
            constraint_id in _permitted_constraints(token, evidence) for token in cited
        ):
            errors.append(f"step {index} cites a constraint that does not apply")
        if not isinstance(step.get("claim"), str) or not step["claim"].strip():
            errors.append(f"step {index} has no claim")
    return errors


class CentralizedAgent:
    """One auditable agent that coordinates deterministic analytical tools."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        top_k: int = PAPER_DEFAULT_TOP_K,
        tau: float = PAPER_DEFAULT_TAU,
        epsilon: float = PAPER_DEFAULT_EPSILON,
        reasoner: StructuredReasoner | None = None,
        judger: StructuredJudger | None = None,
        max_revisions: int = 3,
    ) -> None:
        if int(top_k) < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 <= float(tau) <= 1.0 or not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("tau and epsilon must be within [0, 1]")
        self.repository = repository
        self.top_k = int(top_k)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.reasoner = reasoner
        self.judger = judger
        self.max_revisions = max(1, int(max_revisions))

    def diagnose(self, graph: EvidenceGraph) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        candidates = self.repository.search(graph, self.top_k)
        trace.append({"tool": "search_evidence_graph", "result": [item.to_dict() for item in candidates]})
        pattern, route_reason = _route(candidates, self.tau)
        constraints = _constraint_analysis(graph)
        trace.append({"tool": "check_physical_constraints", "result": constraints})
        trace.append({"tool": "list_missing_evidence", "result": list(graph.missing_tokens)})
        compatible = [item for item in candidates if item.topology_compatible] or list(candidates)
        best = compatible[0] if compatible else None
        critical_missing = _critical_missing_evidence(compatible) if pattern == "ambiguous" else ()
        branch_context: dict[str, Any]
        if pattern == "common":
            branch_context = {
                "operation": "history_based_validation",
                "reused_case_id": best.case_id if best else None,
                "reused_reasoning_chain": dict(best.reusable_reasoning_chain) if best else {},
            }
        elif pattern == "ambiguous":
            branch_context = {
                "operation": "constraint_guided_arbitration",
                "critical_missing_evidence": list(critical_missing),
                "mutually_exclusive_evidence": list(best.mutually_exclusive_evidence) if best else [],
            }
        else:
            branch_context = {
                "operation": "hypothesis_driven_inference",
                "admissible_hypotheses": list(ROOT_CAUSES),
                "sop": list(_novel_sop(graph)),
            }
        trace.append({"tool": branch_context["operation"], "result": branch_context})

        model_result: Mapping[str, Any] | None = None
        unresolved_reasoner_violation = False
        if self.reasoner is not None:
            reasoning_input = {
                "case_id": graph.case_id,
                "pattern": pattern,
                "evidence_tokens": list(graph.evidence_tokens),
                "missing_tokens": list(graph.missing_tokens),
                "constraints": constraints,
                "constraint_version": CONSTRAINT_VERSION,
                "historical_candidates": [item.to_dict() for item in candidates],
                "branch_context": branch_context,
            }
            for attempt in range(self.max_revisions):
                try:
                    candidate_result = self.reasoner.reason(reasoning_input)
                    errors = _validate_reasoner_output(candidate_result, graph.evidence_tokens)
                except Exception as exc:  # Fail closed at the optional model boundary.
                    candidate_result, errors = {}, [f"reasoner failure: {type(exc).__name__}"]
                trace.append({
                    "tool": "structured_reasoner",
                    "attempt": attempt + 1,
                    "result": dict(candidate_result),
                    "validation_errors": errors,
                })
                if not errors:
                    model_result = candidate_result
                    break
                reasoning_input["revision_feedback"] = errors
            unresolved_reasoner_violation = model_result is None

        scores: Counter[str] = Counter()
        for root_cause, reasons in constraints["support"].items():
            scores[root_cause] += 2 * len(reasons)
        for root_cause, reasons in constraints["exclude"].items():
            scores[root_cause] -= 3 * len(reasons)
        if best:
            historical_weight = 4 if pattern == "common" else 2 if pattern == "ambiguous" else 0
            scores[best.root_cause] += historical_weight
            if best.mutually_exclusive_evidence:
                scores[best.root_cause] -= 2
        if model_result is not None:
            scores[str(model_result["verdict"])] += 2
        verdict = max(ROOT_CAUSES, key=lambda item: (scores[item], -ROOT_CAUSES.index(item)))
        max_score = scores[verdict]
        ties = [item for item in ROOT_CAUSES if scores[item] == max_score]
        reasoning_validity = 1.0 if max_score > 0 and len(ties) == 1 else 0.0
        if pattern == "common" and model_result is not None and best:
            if model_result["verdict"] != best.root_cause:
                reasoning_validity = 0.0
        missing_ratio = len(graph.missing_tokens) / max(1, len(graph.evidence_tokens))
        deterministic_dimensions = {
            "evidence_completeness": round(graph.coverage * (1.0 - missing_ratio), 6),
            "constraint_compliance": 0.0 if unresolved_reasoner_violation else float(constraints["compliance"]),
            "reasoning_validity": reasoning_validity,
        }
        confidence_dimensions = deterministic_dimensions
        if self.judger is not None:
            try:
                confidence_dimensions = _validate_judger_output(self.judger.judge({
                    "evidence_graph": graph.to_dict(),
                    "root_cause": verdict,
                    "reasoning": dict(model_result or {}),
                    "branch_context": branch_context,
                    "deterministic_checks": deterministic_dimensions,
                }))
                judge_error = ""
            except Exception as exc:
                confidence_dimensions = {dimension: 0.0 for dimension in JUDGE_DIMENSIONS}
                judge_error = f"judger failure: {type(exc).__name__}"
            trace.append({
                "tool": "independent_confidence_judger",
                "result": confidence_dimensions,
                "error": judge_error,
            })
        confidence = _geometric_confidence(confidence_dimensions)
        action = "final" if confidence >= self.epsilon else (
            "request_evidence" if graph.missing_tokens else "human_review"
        )
        reasoning_steps = []
        if pattern == "common" and best and best.reusable_reasoning_chain:
            reasoning_steps.extend(dict(item) for item in best.reusable_reasoning_chain.get("reasoning_steps", []))
        for reason in constraints["support"].get(verdict, []):
            reasoning_steps.append({"effect": "support", "target": verdict, "claim": reason})
        for reason in constraints["exclude"].get(verdict, []):
            reasoning_steps.append({"effect": "exclude", "target": verdict, "claim": reason})
        if best:
            reasoning_steps.append({
                "effect": "support" if pattern != "novel" else "neutral",
                "target": verdict,
                "claim": f"Best historical dual-view match is {best.case_id} "
                         f"({best.evidence_similarity:.1%}, {best.graph_similarity:.1%}).",
            })
        if model_result is not None:
            reasoning_steps.extend({
                "effect": "support",
                "target": model_result["verdict"],
                "claim": step["claim"],
                "cited_evidence": list(step["cited_evidence"]),
                "cited_constraint": step["cited_constraint"],
                "source": "structured_reasoner",
            } for step in model_result["reasoning_steps"])
        limiting_factors = []
        if critical_missing:
            limiting_factors.append("missing critical evidence")
        if best and best.mutually_exclusive_evidence:
            limiting_factors.append("conflicting observations")
        if confidence_dimensions["constraint_compliance"] < 1.0:
            limiting_factors.append("physical constraint non-compliance")
        if confidence_dimensions["reasoning_validity"] < 1.0:
            limiting_factors.append("invalid or inconclusive reasoning")
        failure_context = {
            "case_id": graph.case_id,
            "alarm": graph.context.get("alarm"),
            "network_layer": graph.topology.get("network_layer"),
            "lane_count": graph.topology.get("lane_count"),
            "components": list(ROOT_CAUSES),
        }
        reasoning_chain = {
            "failure_context": failure_context,
            "key_observations": list(graph.evidence_tokens),
            "reasoning_steps": reasoning_steps,
            "rcl_conclusion": {
                "root_cause": verdict if action == "final" else None,
                "proposed_root_cause": verdict,
                "action": action,
            },
            "result_reliability": {
                "confidence": confidence,
                "dimensions": confidence_dimensions,
                "threshold": self.epsilon,
                "limiting_factors": limiting_factors,
            },
        }
        return {
            "system_version": SYSTEM_VERSION,
            "case_id": graph.case_id,
            "pattern": pattern,
            "route_reason": route_reason,
            "action": action,
            "verdict": verdict if action == "final" else None,
            "proposed_verdict": verdict,
            "confidence": confidence,
            "confidence_dimensions": confidence_dimensions,
            "reasoning_steps": reasoning_steps,
            "reasoning_chain": reasoning_chain,
            "missing_information": list(critical_missing or graph.missing_tokens) if action == "request_evidence" else [],
            "branch_context": branch_context,
            "tool_trace": trace,
            "knowledge_version": self.repository.version,
        }
