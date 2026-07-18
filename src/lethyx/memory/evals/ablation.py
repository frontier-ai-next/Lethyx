"""No-LLM static memory ablation evaluation."""

from lethyx.memory.evals.fixtures import static_scenarios
from lethyx.memory.evals.metrics import (
    contamination_rate,
    directive_precision,
    directive_recall,
    retrieval_leakage_rate,
    token_roi,
)

CONDITIONS = (
    "A0_NO_MEMORY",
    "A1_RETRIEVAL_ONLY",
    "A2_PROACTIVE_DIRECTIVE_ONLY",
    "A3_PROACTIVE_PLUS_SUPPLEMENTAL",
    "A4_PROACTIVE_PLUS_HINDSIGHT_CANDIDATES_GATED",
    "A5_PROACTIVE_PLUS_TRIGGERS_PLUS_UTILITY",
    "A6_CANONICAL_MEMPALACE_ONLY",
    "A7_MNEMOSYNE_PASSIVE_RECALL_ONLY",
    "A8_CANONICAL_PLUS_MNEMOSYNE",
    "A9_CANONICAL_PLUS_HINDSIGHT_PLUS_MNEMOSYNE",
)


def run_static_ablation() -> dict[str, object]:
    rows = []
    for condition in CONDITIONS:
        items = []
        expected = []
        for scenario in static_scenarios():
            scenario_items = list(scenario["items"])
            if condition == "A0_NO_MEMORY":
                scenario_items = []
            elif "RETRIEVAL_ONLY" in condition or "MNEMOSYNE" in condition:
                scenario_items = [i for i in scenario_items if not i.get("directive")]
            items.extend(scenario_items)
            expected.extend(scenario.get("expected", []))
        rows.append(
            {
                "condition": condition,
                "directive_precision": directive_precision(items),
                "directive_recall": directive_recall(expected, items),
                "contamination_rate": contamination_rate(items),
                "retrieval_leakage_rate": retrieval_leakage_rate(items),
                "token_roi": token_roi(1.0, sum(int(i.get("token_cost", 10)) for i in items)),
            }
        )
    return {"schema_version": "1", "conditions": rows}
