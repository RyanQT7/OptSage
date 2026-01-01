"""Deterministic physical relations used to bound diagnostic reasoning."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .models import CONSTRAINT_VERSION, EvidenceGraph, ROOT_CAUSES


def _admissible_relations(token: str) -> dict[str, str]:
    """Return paper-style support/contradict/unavailable hypothesis relations."""
    relations = {root: "neutral" for root in ROOT_CAUSES}
    parts = token.split(":")
    if token.startswith("missing:"):
        return {root: "unavailable" for root in ROOT_CAUSES}
    if token.startswith("direction:") and token.endswith(":not_emitted"):
        sender = parts[1].split("_to_")[0]
        relations[sender] = "support"
        relations["fiber"] = "contradict"
    elif token.startswith("direction:") and token.endswith(":emitted_not_received"):
        receiver = parts[1].split("_to_")[1]
        relations[receiver] = "support"
        relations["fiber"] = "support"
    elif token.startswith(("state:", "log:")) and token.endswith(":asserted"):
        relations[parts[1]] = "support"
    return relations


def _observation_directions(endpoint: str, name: str) -> tuple[str, ...]:
    """Associate endpoint state/log observations with their signal direction."""
    if endpoint not in {"L1", "L2"}:
        return ()
    peer = "L2" if endpoint == "L1" else "L1"
    normalized = name.lower().replace("_", "")
    if normalized.startswith("tx"):
        return (f"{endpoint}_to_{peer}",)
    if normalized.startswith("rx"):
        return (f"{peer}_to_{endpoint}",)
    return (f"{endpoint}_to_{peer}", f"{peer}_to_{endpoint}")


def _constraint_analysis(graph: EvidenceGraph) -> dict[str, Any]:
    support: dict[str, list[str]] = {item: [] for item in ROOT_CAUSES}
    exclude: dict[str, list[str]] = {item: [] for item in ROOT_CAUSES}
    for token in graph.evidence_tokens:
        parts = token.split(":")
        if token.startswith("direction:") and token.endswith(":not_emitted"):
            sender = parts[1].split("_to_")[0]
            support[sender].append(f"P1 transmitter-off: {token}")
            exclude["fiber"].append(f"P1 medium alone cannot stop emission: {token}")
        elif token.startswith("direction:") and token.endswith(":emitted_not_received"):
            receiver = parts[1].split("_to_")[1]
            support[receiver].append(f"P2 receive-chain candidate: {token}")
            support["fiber"].append(f"P2 propagation-path candidate: {token}")
        elif token.startswith(("state:", "log:")) and token.endswith(":asserted"):
            endpoint = parts[1]
            support[endpoint].append(f"P3 asserted endpoint state: {token}")
    # Same lane degraded in both directions strengthens the shared medium.
    emitted = [item for item in graph.evidence_tokens if item.endswith(":emitted_not_received")]
    lanes = Counter(item.split(":")[-2] for item in emitted)
    for lane, count in lanes.items():
        if count >= 2:
            support["fiber"].append(f"P4 bidirectional same-lane degradation: {lane}")
    return {
        "version": CONSTRAINT_VERSION,
        "support": support,
        "exclude": exclude,
        "compliance": 1.0,
    }


def _permitted_constraints(token: str, evidence: Sequence[str]) -> set[str]:
    allowed: set[str] = set()
    if token.startswith("direction:") and token.endswith(":not_emitted"):
        allowed.add("P1")
    if token.startswith("direction:") and token.endswith(":emitted_not_received"):
        allowed.add("P2")
        lane = token.split(":")[-2]
        same_lane = [
            item for item in evidence
            if item.startswith("direction:")
            and item.endswith(":emitted_not_received")
            and item.split(":")[-2] == lane
        ]
        if len(same_lane) >= 2:
            allowed.add("P4")
    if token.startswith(("state:", "log:")) and token.endswith(":asserted"):
        allowed.add("P3")
    return allowed
