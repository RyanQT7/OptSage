# OptSage v4 paper-to-code crosswalk

This document maps the attached submission's design claims to the review artifact.
It is a code navigation aid, not an evaluation-results release.

| Paper section | Mechanism | Implementation |
| --- | --- | --- |
| V-A.1 | Time-aware preprocessing | `optsage.evidence.build_evidence_graph` stable-sorts telemetry by timestamp and emits `temporal` edges. |
| V-A.1 | Direction-aware binding | `build_evidence_graph` pairs equal-index Tx/Rx lanes only when lane sets and topology width agree. It emits categorical directional states and binds endpoint state/log observations to the corresponding signal direction. |
| V-A.2 | Anomalous, normal, missing evidence | `_performance_state`, `_flag_state`, `_log_state`, and explicit `missing_tokens`. Per-observation hardware ranges and historical baselines are supported; no missing value is imputed. |
| V-A.2 | Four telemetry categories | Hardware/context, performance metrics, state flags, and log events share one input/event model. |
| V-A.3 | Unified evidence graph | `EvidenceGraph` contains component, observation, evidence, and hypothesis nodes. It emits physical, attachment, temporal, semantic, and directional edges; semantic relations are `support`, `contradict`, or `unavailable`. |
| V-B.1 | Physical constraint toolset | `optsage.constraints._constraint_analysis` returns versioned support/exclude relations and is invoked in the agent trace. |
| V-B.2, Eq. 1 | Evidence similarity | `optsage.repository.KnowledgeRepository._weighted_jaccard` computes IDF-weighted Jaccard over evidence tokens. |
| V-B.2 | Graph similarity | `_token_edges` projects evidence to semantic graph edges; the same weighted score is computed independently. |
| V-B.2 | Top-k/topology matching | `KnowledgeRepository.search` prefers compatible network-layer/lane-width cases, defaults to Top-3, and returns shared, historical-only, online-only, mutually exclusive evidence, reusable chains, and SOPs. |
| V-B.3 | Common/ambiguous/novel routing | `optsage.reasoning._route` requires both scores to be exactly 100% for common, both scores to be at least `tau=70%` for ambiguous, and otherwise selects novel. A mixed-label complete bucket is downgraded to arbitration. |
| IV, V-B.3 | Centralized agent | `optsage.reasoning.CentralizedAgent` coordinates graph search, constraint checking, missing-evidence inspection, and an optional structured LLM under one global context. |
| V-B.3 | Three autonomous branches | Common reuses the verified historical chain and independently validates it; ambiguous exposes candidate conflicts and only marks relation-changing omissions critical; novel enumerates all three hypotheses, applies exclusions, inspects same-lane directionality, and executes a versioned SOP. |
| V-B.3 | Bounded reasoning repair | `_validate_reasoner_output` rejects unavailable evidence, unknown constraints, and constraints that do not apply to the cited evidence; the agent returns targeted validation feedback for up to `max_revisions` attempts. |
| V-C.1, Eq. 2 | Decoupled judging | `StructuredJudger` and `OpenAICompatibleJudger` are independent of the primary reasoner and must return exactly evidence completeness, constraint compliance, and reasoning validity. |
| V-C.1, Eq. 2 | Confidence and abstention | `_geometric_confidence` computes the exact geometric mean. `epsilon=70%` controls final output versus evidence collection/operator review. Invalid model output fails closed. |
| V-C.2 | Traceable reasoning chain | `diagnose` returns the paper's five fields: failure context, key observations, reasoning steps, RCL conclusion, and result reliability, plus the full tool trace and repository version. |
| V-C.3 | Evolvable knowledge | `evolve` accepts only independently confirmed cases; `consolidate` merges duplicate instances without hiding conflicts; `refine_heuristics` creates monthly validation-pending revisions; `deprecate_case` retains superseded knowledge for traceability while removing it from retrieval. Every version records its parent. |
| VII-D | Paper defaults | `PAPER_DEFAULT_TOP_K=3`, `PAPER_DEFAULT_TAU=0.70`, and `PAPER_DEFAULT_EPSILON=0.70`. |

## Deliberate artifact boundaries

- The branch contains one sanitized synthetic file, `data/sample.json`.
- Production telemetry, model credentials, experiment caches, generated reports,
  and the full private evaluation corpus are excluded.
- The synthetic fixture validates artifact mechanisms; it does not reproduce
  the paper's production performance figures.
- Online inputs containing `label`, `root_cause`, or `confirmed_root_cause` are
  rejected before evidence construction.
- Knowledge evolution is an explicit operator-confirmed API. A diagnostic result
  cannot write itself back into history.

## Directional measurement contract

The active sample and repository contract establishes same-index optical-lane
pairing but does not establish cross-device calibration for absolute power loss.
Accordingly, the MVP exposes `not_emitted`, `emitted_not_received`, and
`paired_normal` directional evidence. The paper describes subtracting endpoint
Tx/Rx readings as transmission loss; this repository's validated data contract
does not establish cross-device calibration for that absolute quantity, so the
public artifact intentionally uses the categorical equivalent. This is the only
declared implementation boundary from the v4 design, and prevents an unsupported
numeric subtraction from being presented as a physical fact.

## Runtime boundary

The paper deployment uses a primary LLM plus a separate lightweight LLM judger.
The artifact implements both adapters and their validation contracts. Offline
inference intentionally leaves both adapters unset and exercises the same
branches with deterministic physical tools, so it needs no credentials or
network access. The synthetic fixture validates mechanisms, not the paper's
production accuracy, latency, or deployment-scale claims.
