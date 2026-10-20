from lethyx.memory.query_analysis import EvidenceRequirement
from lethyx.memory.recall_plan import RecallPlan


def test_semantic_synthesis_query_does_not_compile_subjects_facets_or_roles() -> None:
    plan = RecallPlan.compile(
        "Final synthesis: compare Fireworks, Groq, and MistralAI by package and dependency."
    )

    assert plan.query_type == "broad_recall"
    assert plan.paths == ()
    assert plan.symbols == ()
    assert plan.entities == ()
    assert plan.evidence_roles == ()
    assert plan.requirements == ()
    assert "requested_facets" not in plan.metadata


def test_prompt_text_does_not_create_automatic_code_terms() -> None:
    plan = RecallPlan.compile(
        "In src/pkg/parser.py, inspect Parser.parse_config without inferring other files."
    )

    assert plan.query_type == "broad_recall"
    assert plan.paths == ()
    assert plan.symbols == ()
    assert plan.entities == ()
    assert plan.requirements == ()


def test_typed_exact_terms_are_caller_owned() -> None:
    plan = RecallPlan.compile(
        "For case-alpha-lifecycle-decision-01 compare Chroma and Hindsight.",
        entities=("case-alpha-lifecycle-decision-01",),
    )

    assert plan.query_type == "exact_lookup"
    assert plan.entities == ("case-alpha-lifecycle-decision-01",)


def test_explicit_requirements_remain_typed_caller_owned_contract() -> None:
    requirement = EvidenceRequirement(
        subject="parser decision",
        role="decision",
        accepted_literals=("OV-EXACT",),
    )
    plan = RecallPlan.compile(
        "explain the selected record",
        evidence_roles=("decision",),
        requirements=(requirement,),
    )

    assert plan.evidence_roles == ("decision",)
    assert plan.requirements == (requirement,)
