# Artifact validation gate

| Gate | Status | Evidence |
| --- | --- | --- |
| Online label isolation | PASS | `test_online_label_is_rejected` |
| Modular boundaries and import compatibility | PASS | `test_legacy_system_surface_reexports_modular_implementation` |
| Unified graph node/edge model | PASS | `test_graph_contains_all_paper_edge_types` |
| Directional measurement contract | PASS | `test_direction_is_categorical_not_absolute_loss` |
| State/log directional association | PASS | `test_state_and_log_observations_are_direction_bound` |
| Common/ambiguous/novel routing | PASS | `test_sample_covers_three_routes` |
| Traceable tool/reasoning output | PASS | `test_output_is_traceable` |
| Structured LLM validation/revision | PASS | `test_structured_reasoner_is_checked_and_revised` |
| Exact dual-view routing boundaries | PASS | `test_paper_routing_requires_both_views_and_pure_complete_bucket` |
| Topology-first retrieval/fallback | PASS | `test_topology_compatible_history_precedes_fallback` |
| History-chain reuse and novel SOP | PASS | `test_common_reuses_verified_chain_and_novel_runs_sop` |
| Eq. (2) and independent judger | PASS | `test_eq2_geometric_mean_and_independent_judger` |
| Five-part reasoning chain | PASS | `test_output_is_traceable` |
| Range/baseline/log extraction | PASS | `test_performance_range_baseline_and_log_templates` |
| Versioned consolidation/refinement/deprecation | PASS | `EvolutionTests` |
| Low-confidence degradation | PASS | `test_insufficient_case_does_not_force_a_class` |
| Confirmed, immutable evolution | PASS | `test_evolution_requires_confirmation_and_preserves_versions` |
| Safe consolidation/conflict audit | PASS | `test_consolidation_deduplicates_but_preserves_label_conflicts` |
| Single public data file | PASS | Git tree contains only `data/sample.json` under `data/` |

Validation command:

```bash
python3 -m unittest discover -s tests -v
```

The sample validates framework behavior only. It is not a substitute for the
paper's production evaluation and must not be used to recompute those metrics.
