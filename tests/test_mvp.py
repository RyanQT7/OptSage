import copy
import json
import unittest
from pathlib import Path

from optsage import OptSage, build_evidence_graph, load_sample
from optsage.models import Candidate
from optsage.reasoning import _geometric_confidence, _route


SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample.json"


class ArchitectureTests(unittest.TestCase):
    def test_legacy_system_surface_reexports_modular_implementation(self):
        from optsage.api import OptSage as ApiOptSage
        from optsage.evidence import build_evidence_graph as graph_builder
        from optsage.system import OptSage as LegacyOptSage
        from optsage.system import build_evidence_graph as legacy_graph_builder

        self.assertIs(ApiOptSage, LegacyOptSage)
        self.assertIs(graph_builder, legacy_graph_builder)


class EvidenceConstructionTests(unittest.TestCase):
    def setUp(self):
        self.repository, self.queries = load_sample(SAMPLE)

    def test_online_label_is_rejected(self):
        case = copy.deepcopy(self.queries[0])
        case["root_cause"] = "fiber"
        with self.assertRaisesRegex(ValueError, "label-bearing"):
            build_evidence_graph(case)

    def test_graph_contains_all_paper_edge_types(self):
        graph = build_evidence_graph(self.queries[0])
        kinds = {item.kind for item in graph.edges}
        self.assertTrue({"physical", "attachment", "temporal", "semantic", "directional"} <= kinds)
        self.assertEqual(
            {"component", "observation", "evidence", "hypothesis"},
            {item.kind for item in graph.nodes},
        )

    def test_performance_range_baseline_and_log_templates(self):
        case = {
            "case_id": "extraction",
            "topology": {"network_layer": "server-leaf", "lane_count": 1},
            "telemetry": [
                {"timestamp": "1", "endpoint": "L1", "category": "performance", "name": "Voltage", "lanes": {"0": 9}, "valid_range": [0, 5]},
                {"timestamp": "2", "endpoint": "L1", "category": "performance", "name": "Current", "lanes": {"0": 9}, "historical_baseline": {"mean": 1, "std": 1, "z_threshold": 3}},
                {"timestamp": "3", "endpoint": "L1", "category": "log", "name": "PortFail", "value": "subhealthy"},
            ],
        }
        graph = build_evidence_graph(case)
        joined = "\n".join(graph.evidence_tokens)
        self.assertIn("out_of_range", joined)
        self.assertIn("baseline_deviation", joined)
        self.assertIn("log:L1:PortFail:asserted", joined)

    def test_state_and_log_observations_are_direction_bound(self):
        graph = build_evidence_graph({
            "case_id": "direction-bound",
            "topology": {"network_layer": "server-leaf", "lane_count": 1},
            "telemetry": [
                {"timestamp": "1", "endpoint": "L1", "category": "state", "name": "RxLOS", "value": True},
                {"timestamp": "2", "endpoint": "L2", "category": "log", "name": "LinkDown", "value": True},
            ],
        })
        directions = {
            edge.attributes.get("direction")
            for edge in graph.edges
            if edge.kind == "directional" and edge.attributes.get("direction")
        }
        self.assertEqual({"L2_to_L1", "L1_to_L2"}, directions)

    def test_direction_is_categorical_not_absolute_loss(self):
        graph = build_evidence_graph(self.queries[0])
        direction_tokens = [item for item in graph.evidence_tokens if item.startswith("direction:")]
        self.assertTrue(any(item.endswith(":emitted_not_received") for item in direction_tokens))
        self.assertNotIn("loss_db", json.dumps(graph.to_dict()))


class ReasoningTests(unittest.TestCase):
    def setUp(self):
        self.repository, self.queries = load_sample(SAMPLE)
        self.system = OptSage(self.repository, top_k=3, tau=0.70, epsilon=0.70)

    def test_sample_covers_three_routes(self):
        results = {item["case_id"]: item for item in map(self.system.diagnose, self.queries)}
        self.assertEqual("common", results["query-common-001"]["pattern"])
        self.assertEqual("ambiguous", results["query-ambiguous-001"]["pattern"])
        self.assertEqual("novel", results["query-novel-001"]["pattern"])
        self.assertEqual("fiber", results["query-common-001"]["verdict"])
        self.assertEqual("L1", results["query-ambiguous-001"]["verdict"])
        self.assertEqual("L2", results["query-novel-001"]["verdict"])

    def test_output_is_traceable(self):
        result = self.system.diagnose(self.queries[0])
        self.assertEqual(
            ["search_evidence_graph", "check_physical_constraints", "list_missing_evidence"],
            [item["tool"] for item in result["tool_trace"]][:3],
        )
        self.assertTrue(result["reasoning_steps"])
        self.assertEqual(
            {"failure_context", "key_observations", "reasoning_steps", "rcl_conclusion", "result_reliability"},
            set(result["reasoning_chain"]),
        )
        self.assertEqual(self.repository.version, result["knowledge_version"])

    def test_common_reuses_verified_chain_and_novel_runs_sop(self):
        common = self.system.diagnose(self.queries[0])
        novel = self.system.diagnose(self.queries[2])
        self.assertIn("Verified same-lane", common["reasoning_steps"][0]["claim"])
        self.assertEqual("history_based_validation", common["branch_context"]["operation"])
        self.assertEqual("hypothesis_driven_inference", novel["branch_context"]["operation"])
        self.assertEqual(["L1", "L2", "fiber"], novel["branch_context"]["admissible_hypotheses"])
        self.assertTrue(novel["branch_context"]["sop"])

    def test_eq2_geometric_mean_and_independent_judger(self):
        self.assertAlmostEqual((0.8 * 0.9 * 1.0) ** (1 / 3), _geometric_confidence({
            "evidence_completeness": 0.8,
            "constraint_compliance": 0.9,
            "reasoning_validity": 1.0,
        }), places=6)

        class Judger:
            def judge(self, payload):
                return {
                    "evidence_completeness": 0.8,
                    "constraint_compliance": 0.9,
                    "reasoning_validity": 1.0,
                }

        result = OptSage(self.repository, judger=Judger()).diagnose(self.queries[0])
        self.assertTrue(any(item["tool"] == "independent_confidence_judger" for item in result["tool_trace"]))
        self.assertEqual(_geometric_confidence(result["confidence_dimensions"]), result["confidence"])

    def test_paper_routing_requires_both_views_and_pure_complete_bucket(self):
        def candidate(root, evidence, graph):
            return Candidate("x-" + root, root, evidence, graph, True, (), (), (), (), (), {})

        self.assertEqual("novel", _route([candidate("L1", 0.99, 0.69)], 0.70)[0])
        self.assertEqual("ambiguous", _route([candidate("L1", 0.70, 0.70)], 0.70)[0])
        self.assertEqual("common", _route([candidate("L1", 1.0, 1.0)], 0.70)[0])
        self.assertEqual(
            "ambiguous",
            _route([candidate("L1", 1.0, 1.0), candidate("L2", 1.0, 1.0)], 0.70)[0],
        )

    def test_topology_compatible_history_precedes_fallback(self):
        graph = build_evidence_graph(self.queries[0])
        candidates = self.repository.search(graph, top_k=3)
        self.assertTrue(candidates)
        self.assertTrue(all(item.topology_compatible for item in candidates))

    def test_insufficient_case_does_not_force_a_class(self):
        case = {
            "case_id": "insufficient",
            "topology": {"network_layer": "server-leaf", "lane_count": 2},
            "telemetry": [
                {"timestamp": "2026-03-04T00:00:00Z", "endpoint": "L1", "category": "state", "name": "RxLOS", "value": None}
            ],
        }
        result = self.system.diagnose(case)
        self.assertIn(result["action"], {"request_evidence", "human_review"})
        self.assertIsNone(result["verdict"])

    def test_structured_reasoner_is_checked_and_revised(self):
        class ScriptedReasoner:
            def __init__(self):
                self.calls = 0

            def reason(self, payload):
                self.calls += 1
                if self.calls == 1:
                    return {"verdict": "fiber", "reasoning_steps": [{
                        "claim": "unsupported", "cited_evidence": ["invented"],
                        "cited_constraint": "P9",
                    }]}
                return {"verdict": "fiber", "reasoning_steps": [{
                    "claim": "Both directions lose the same lane.",
                    "cited_evidence": ["direction:L1_to_L2:lane0:emitted_not_received"],
                    "cited_constraint": "P4",
                }]}

        reasoner = ScriptedReasoner()
        system = OptSage(self.repository, reasoner=reasoner, max_revisions=3)
        result = system.diagnose(self.queries[0])
        attempts = [item for item in result["tool_trace"] if item["tool"] == "structured_reasoner"]
        self.assertEqual(2, reasoner.calls)
        self.assertTrue(attempts[0]["validation_errors"])
        self.assertFalse(attempts[1]["validation_errors"])
        self.assertEqual("final", result["action"])


class EvolutionTests(unittest.TestCase):
    def test_evolution_requires_confirmation_and_preserves_versions(self):
        repository, queries = load_sample(SAMPLE)
        confirmed = copy.deepcopy(queries[2])
        confirmed["case_id"] = "confirmed-new-001"
        confirmed["confirmed_root_cause"] = "L2"
        evolved = repository.evolve(confirmed, confirmed_by="operator-7")
        self.assertNotEqual(repository.version, evolved.version)
        self.assertEqual(len(repository.cases) + 1, len(evolved.cases))
        self.assertEqual(3, len(repository.cases))
        with self.assertRaisesRegex(ValueError, "operator identity"):
            repository.evolve(confirmed, confirmed_by="")

    def test_consolidation_deduplicates_but_preserves_label_conflicts(self):
        repository, queries = load_sample(SAMPLE)
        duplicate = copy.deepcopy(queries[0])
        duplicate["case_id"] = "confirmed-fiber-duplicate"
        duplicate["confirmed_root_cause"] = "fiber"
        conflicting = copy.deepcopy(queries[0])
        conflicting["case_id"] = "confirmed-fiber-conflict"
        conflicting["confirmed_root_cause"] = "L1"
        evolved = repository.evolve(duplicate, confirmed_by="operator-7")
        evolved = evolved.evolve(conflicting, confirmed_by="operator-8")
        consolidated = evolved.consolidate()
        self.assertEqual(len(evolved.cases) - 1, len(consolidated.cases))
        self.assertTrue(any(item.merged_case_ids for item in consolidated.cases))
        self.assertTrue(any(item["mixed_label"] for item in consolidated.pattern_audit()))

    def test_monthly_heuristic_revision_is_versioned_not_activated_silently(self):
        repository, _ = load_sample(SAMPLE)
        revised = repository.refine_heuristics(
            [{"error_cause": "missing_evidence"}, {"error_cause": "insufficient_constraint"}],
            effective_month="2026-09",
        )
        self.assertEqual(repository.version, revised.parent_version)
        self.assertEqual("requires_offline_validation", revised.heuristic_versions[-1]["validation_result"])

    def test_deprecated_knowledge_is_retained_but_not_retrieved(self):
        repository, queries = load_sample(SAMPLE)
        deprecated = repository.deprecate_case(
            "history-fiber-001", validation_result="superseded_after_review"
        )
        serialized = deprecated.to_dict()
        self.assertTrue(any(item["status"] == "deprecated" for item in serialized["cases"]))
        candidates = deprecated.search(build_evidence_graph(queries[0]))
        self.assertNotIn("history-fiber-001", {item.case_id for item in candidates})


if __name__ == "__main__":
    unittest.main()
