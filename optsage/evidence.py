"""Label-free evidence normalization and graph construction (paper §IV)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .constraints import _admissible_relations, _observation_directions
from .models import Edge, EvidenceGraph, Node, ROOT_CAUSES


def _lane_sort(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_state(name: str, value: float) -> str:
    normalized = name.lower().replace("_", "")
    if normalized in {"txpower", "rxpower"}:
        return "down" if value <= -30.0 else "normal"
    if normalized in {"mediasnr", "hostsnr", "serdessnr"}:
        return "degraded" if value <= 1.0 else "normal"
    if normalized == "temperature":
        return "out_of_range" if not -10.0 <= value <= 85.0 else "normal"
    if normalized in {"voltage", "current"}:
        return "out_of_range" if value < 0 else "normal"
    return "observed"


def _performance_state(row: Mapping[str, Any], value: float) -> str:
    """Apply hardware range, historical baseline, then the public fallback."""
    valid_range = row.get("valid_range")
    if isinstance(valid_range, Sequence) and not isinstance(valid_range, (str, bytes)) and len(valid_range) == 2:
        lower, upper = _number(valid_range[0]), _number(valid_range[1])
        if lower is not None and upper is not None and not lower <= value <= upper:
            return "out_of_range"
    baseline = row.get("historical_baseline")
    if isinstance(baseline, Mapping):
        center = _number(baseline.get("mean"))
        deviation = _number(baseline.get("std"))
        multiplier = _number(baseline.get("z_threshold")) or 3.0
        if center is not None and deviation is not None and deviation > 0:
            if abs(value - center) > multiplier * deviation:
                return "baseline_deviation"
    return _metric_state(str(row.get("name") or ""), value)


def _flag_state(value: Any) -> str:
    text = str(value).strip().lower()
    return "asserted" if value is True or text in {
        "1", "true", "yes", "asserted", "abnormal", "down", "fail", "failed", "los", "lol"
    } else "normal"


def _log_state(name: str, value: Any) -> str:
    event = f"{name} {value}".lower().replace("_", "")
    if any(item in event for item in ("linkdown", "portfail", "subhealthy", "failure")):
        return "asserted"
    if "linkup" in event or "stable" in event or "recover" in event:
        return "normal"
    return _flag_state(value)


def _token_edges(token: str) -> set[str]:
    """Project a token into semantic-prefix edges for graph similarity."""
    parts = token.split(":")
    edges: set[str] = set()
    parent = f"family:{parts[0]}"
    for depth in range(1, len(parts)):
        child = "path:" + ":".join(parts[: depth + 1])
        edges.add(f"{parent}|segment_{depth}|{child}")
        parent = child
    return edges


def build_evidence_graph(case: Mapping[str, Any]) -> EvidenceGraph:
    """Build a label-free, time-ordered and direction-aware evidence graph."""
    forbidden = {"label", "root_cause", "confirmed_root_cause"}
    if forbidden.intersection(case):
        raise ValueError("online evidence construction rejects label-bearing cases")
    case_id = str(case.get("case_id") or "unknown")
    topology = dict(case.get("topology") or {})
    context = dict(case.get("context") or {})
    nodes: list[Node] = [
        Node("component:L1", "component", {"role": "local_transceiver"}),
        Node("component:L2", "component", {"role": "peer_transceiver"}),
        Node("component:fiber", "component", {"role": "link_medium"}),
        *(Node(f"hypothesis:{root}", "hypothesis", {"root_cause": root}) for root in ROOT_CAUSES),
    ]
    edges: list[Edge] = [
        Edge("component:L1", "component:fiber", "physical"),
        Edge("component:fiber", "component:L2", "physical"),
    ]
    tokens: set[str] = set()
    missing: set[str] = set()
    observed_slots = 0
    total_slots = 0
    directional: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)

    telemetry = case.get("telemetry") or []
    if not isinstance(telemetry, list):
        raise TypeError("telemetry must be a list")
    ordered = sorted(
        enumerate(telemetry),
        key=lambda item: (str(item[1].get("timestamp", "")), item[0]),
    )
    previous_node = ""
    for order, (_, row) in enumerate(ordered):
        endpoint = str(row.get("endpoint") or "")
        if endpoint not in {"L1", "L2", "fiber"}:
            raise ValueError(f"unsupported telemetry endpoint: {endpoint!r}")
        category = str(row.get("category") or "unknown").lower()
        name = str(row.get("name") or "unknown")
        timestamp = str(row.get("timestamp") or "")
        telemetry_id = f"telemetry:{order}"
        nodes.append(Node(telemetry_id, "observation", {
            "endpoint": endpoint, "category": category, "name": name,
            "timestamp": timestamp,
        }))
        edges.append(Edge(f"component:{endpoint}", telemetry_id, "attachment"))
        if previous_node:
            edges.append(Edge(previous_node, telemetry_id, "temporal"))
        previous_node = telemetry_id

        lanes = row.get("lanes")
        if isinstance(lanes, Mapping):
            for lane in sorted((str(item) for item in lanes), key=_lane_sort):
                total_slots += 1
                value = _number(lanes.get(lane))
                if value is None:
                    token = f"missing:{endpoint}:{name}:lane{lane}"
                    missing.add(token)
                    state, relation = "missing", "unavailable"
                else:
                    observed_slots += 1
                    state = _performance_state(row, value)
                    relation = "normal" if state in {"normal", "observed"} else "anomalous"
                    token = f"metric:{endpoint}:{name}:lane{lane}:{state}"
                    tokens.add(token)
                    normalized = name.lower().replace("_", "")
                    if normalized in {"txpower", "rxpower"}:
                        directional[(endpoint, normalized)][lane] = value
                evidence_id = f"evidence:{len(nodes)}"
                nodes.append(Node(evidence_id, "evidence", {
                    "token": token, "state": state, "value": value,
                }))
                edges.append(Edge(telemetry_id, evidence_id, "attachment"))
                for root_cause, admissible in _admissible_relations(token).items():
                    if admissible != "neutral":
                        edges.append(Edge(evidence_id, f"hypothesis:{root_cause}", "semantic", {"relation": admissible}))
        else:
            total_slots += 1
            value = row.get("value")
            if value is None:
                state = "missing"
                token = f"missing:{endpoint}:{name}"
                missing.add(token)
                relation = "unavailable"
            else:
                observed_slots += 1
                if category == "log":
                    state = _log_state(name, value)
                elif category == "state":
                    state = _flag_state(value)
                else:
                    state = "context"
                token = f"{category}:{endpoint}:{name}:{state}"
                tokens.add(token)
                relation = "anomalous" if state == "asserted" else "normal"
            evidence_id = f"evidence:{len(nodes)}"
            nodes.append(Node(evidence_id, "evidence", {"token": token, "state": state}))
            edges.append(Edge(telemetry_id, evidence_id, "attachment"))
            if category in {"state", "log"}:
                for direction in _observation_directions(endpoint, name):
                    edges.append(Edge(
                        telemetry_id,
                        evidence_id,
                        "directional",
                        {"direction": direction},
                    ))
            for root_cause, admissible in _admissible_relations(token).items():
                if admissible != "neutral":
                    edges.append(Edge(evidence_id, f"hypothesis:{root_cause}", "semantic", {"relation": admissible}))

    # Direction-aware binding is categorical. The active data contract does not
    # permit subtracting uncalibrated endpoint readings as absolute link loss.
    lane_count = int(topology.get("lane_count") or 0)
    for sender, receiver in (("L1", "L2"), ("L2", "L1")):
        tx = directional.get((sender, "txpower"), {})
        rx = directional.get((receiver, "rxpower"), {})
        if not tx or not rx or set(tx) != set(rx) or (lane_count and len(tx) != lane_count):
            missing.add(f"missing:direction:{sender}_to_{receiver}:compatible_lane_binding")
            continue
        for lane in sorted(tx, key=_lane_sort):
            tx_value, rx_value = tx[lane], rx[lane]
            if tx_value is None or rx_value is None:
                continue
            tx_on, rx_on = tx_value > -30.0, rx_value > -30.0
            if tx_on and not rx_on:
                state = "emitted_not_received"
            elif not tx_on:
                state = "not_emitted"
            else:
                state = "paired_normal"
            token = f"direction:{sender}_to_{receiver}:lane{lane}:{state}"
            tokens.add(token)
            node_id = f"evidence:{len(nodes)}"
            nodes.append(Node(node_id, "evidence", {"token": token, "state": state}))
            edges.append(Edge(f"component:{sender}", node_id, "directional"))
            edges.append(Edge(node_id, f"component:{receiver}", "directional"))
            for root_cause, admissible in _admissible_relations(token).items():
                if admissible != "neutral":
                    edges.append(Edge(node_id, f"hypothesis:{root_cause}", "semantic", {"relation": admissible}))

    semantic = set()
    for token in tokens | missing:
        semantic.update(_token_edges(token))
        semantic.update(
            f"{token}|{relation}|hypothesis:{root_cause}"
            for root_cause, relation in _admissible_relations(token).items()
            if relation != "neutral"
        )
    coverage = round(observed_slots / total_slots, 6) if total_slots else 0.0
    return EvidenceGraph(
        case_id=case_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        evidence_tokens=tuple(sorted(tokens | missing)),
        semantic_edges=tuple(sorted(semantic)),
        missing_tokens=tuple(sorted(missing)),
        topology=topology,
        context=context,
        coverage=coverage,
    )
