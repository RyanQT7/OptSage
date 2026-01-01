"""Compatibility surface for integrations written against ``optsage.system``.

The implementation lives in focused paper-aligned modules. Re-exporting the
established symbols here keeps existing scripts reproducible across the split.
"""

from .api import OptSage, load_sample
from .constraints import _constraint_analysis
from .evidence import build_evidence_graph
from .models import (
    CONSTRAINT_VERSION,
    GRAPH_SCHEMA,
    JUDGE_DIMENSIONS,
    PAPER_DEFAULT_EPSILON,
    PAPER_DEFAULT_TAU,
    PAPER_DEFAULT_TOP_K,
    REPOSITORY_SCHEMA,
    ROOT_CAUSES,
    SYSTEM_VERSION,
    Candidate,
    Edge,
    EvidenceGraph,
    HistoricalCase,
    Node,
)
from .reasoning import (
    CentralizedAgent,
    OpenAICompatibleJudger,
    OpenAICompatibleReasoner,
    StructuredJudger,
    StructuredReasoner,
    _geometric_confidence,
    _route,
)
from .repository import KnowledgeRepository

__all__ = (
    "CONSTRAINT_VERSION",
    "GRAPH_SCHEMA",
    "JUDGE_DIMENSIONS",
    "PAPER_DEFAULT_EPSILON",
    "PAPER_DEFAULT_TAU",
    "PAPER_DEFAULT_TOP_K",
    "REPOSITORY_SCHEMA",
    "ROOT_CAUSES",
    "SYSTEM_VERSION",
    "Candidate",
    "CentralizedAgent",
    "Edge",
    "EvidenceGraph",
    "HistoricalCase",
    "KnowledgeRepository",
    "Node",
    "OpenAICompatibleJudger",
    "OpenAICompatibleReasoner",
    "OptSage",
    "StructuredJudger",
    "StructuredReasoner",
    "build_evidence_graph",
    "load_sample",
)
