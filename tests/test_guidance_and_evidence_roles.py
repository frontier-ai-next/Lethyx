from um_mem.memory.blocks.catalog import BlockDescriptor, compile_semantic_catalog
from um_mem.memory.blocks.models import MemoryBlock
from um_mem.memory.query_analysis import EvidenceRequirement
from um_mem.memory.recall_plan import RecallPlan


def _catalog_block() -> MemoryBlock:
    return MemoryBlock(
        block_id="code_resources",
        block_type="repository_code",
        temperature="warm",
        scope="workspace",
        authority="advisory",
        delivery_policy="on_demand",
        retention="durable",
        tags=("parser", "code"),
        semantic_memberships=("source", "repository"),
        guidance={
            "synopsis": "Repository code and exact symbol evidence.",
            "use_when": ["debugging code", "finding source"],
            "avoid_when": ["policy-only questions"],
        },
        token_budget=128,
    )


def test_active_catalog_exposes_one_descriptor_contract_without_routing_projections() -> None:
    catalog = compile_semantic_catalog([_catalog_block()])
    descriptor = catalog.descriptor("code_resources")
    payload = descriptor.to_mapping()

    assert isinstance(descriptor, BlockDescriptor)
    assert payload["semantic_memberships"] == ("source", "repository")
    assert payload["guidance"]["synopsis"]
    assert "lexical_terms" not in payload
    assert not hasattr(catalog, "router_cards")
    assert not hasattr(catalog, "reader_guidance_cards")


def test_recall_plan_accepts_only_caller_typed_structural_hints() -> None:
    plan = RecallPlan.compile(
        "Inspect src/um_mem/api.py and RecallPlan.compile for Acme Corp",
        workspace_id="alpha",
        logical_block_ids=("code_resources",),
        paths=("src/um_mem/api.py",),
        symbols=("RecallPlan.compile",),
    )

    assert plan.logical_block_ids == ("code_resources",)
    assert plan.paths == ("src/um_mem/api.py",)
    assert plan.symbols == ("RecallPlan.compile",)
    assert plan.entities == ()
    assert plan.evidence_roles == ()
    assert plan.requirements == ()
    assert "router_cards" not in plan.metadata
    assert "reader_guidance_cards" not in plan.metadata


def test_semantic_evidence_requirements_exist_only_when_the_caller_supplies_them() -> None:
    requirement = EvidenceRequirement(
        subject="release",
        role="test",
        accepted_literals=("release-check-17",),
    )
    implicit = RecallPlan.compile(
        "Compare release source and verification evidence",
        logical_block_ids=("workspace_episodes",),
    )
    explicit = RecallPlan.compile(
        "Compare release source and verification evidence",
        logical_block_ids=("workspace_episodes",),
        evidence_roles=("test",),
        requirements=(requirement,),
    )

    assert implicit.query_type == "broad_recall"
    assert implicit.evidence_roles == ()
    assert implicit.requirements == ()
    assert explicit.evidence_roles == ("test",)
    assert explicit.requirements == (requirement,)


def test_search_all_is_an_explicit_scope_choice_not_a_routing_outcome() -> None:
    plan = RecallPlan.compile("unknown topic", workspace_id="alpha", search_all=True)

    assert plan.search_all is True
    assert plan.logical_block_ids == ()
    assert plan.metadata == {}
