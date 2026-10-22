from pathlib import Path

from um_mem.contracts import RecallFrame
from um_mem.memory.backends.contracts import MemoryIndexNode
from um_mem.memory.operational_slo import RecallBudget
from um_mem.memory.recall_plan import RecallPlan
from um_mem.memory.recall_strategy import compile_reconstructive_strategy
from um_mem.memory.reconstruction_qualification import (
    ReconstructionQualificationCase,
    qualify_reconstructive_product,
)
from um_mem.memory.reconstructive_recall import (
    ReconstructionObservation,
    ReconstructionTraceStore,
    decide_reconstruction_escalation,
    pin_recall_frame,
    run_reconstructive_lite,
)


def _frame(
    query: str,
    *,
    coverage: float = 0.2,
    widening_allowed: bool = True,
) -> RecallFrame:
    return RecallFrame(
        query=query,
        workspace_id="alpha",
        run_id="run-1",
        task_id="task-1",
        subgoal_id="subgoal-1",
        episode_id="episode-1",
        scope_chain=("workspace:alpha",),
        relevant_entities=("Planner",),
        relevant_paths=("src/planner.py",),
        desired_temporal_mode="historical",
        query_time="2026-07-15T10:00:00Z",
        expected_source_types=("procedure",),
        coverage=coverage,
        widening_allowed=widening_allowed,
    )


def _pin(frame: RecallFrame):
    return pin_recall_frame(
        frame,
        canonical_snapshot_revision="sha256:" + "a" * 64,
        catalog_revision="catalog:1",
        policy_revision="policy:1",
    )


def _plan(query: str, *, query_type: str | None = None) -> RecallPlan:
    return RecallPlan(
        query=query,
        query_type=query_type,
        workspace_id="alpha",
        run_id="run-1",
        scope_chain=("workspace:alpha",),
        limit=8,
    )


def _nodes() -> tuple[MemoryIndexNode, ...]:
    return tuple(
        MemoryIndexNode(
            node_id=f"node-{index}",
            title=f"Planner procedure src/planner.py decision {index}",
            node_type=kind,
            workspace_id="alpha",
            metadata={"record_id": f"record-{index}", "kind": kind},
        )
        for index, kind in enumerate(
            ("observation", "entity", "path", "temporal", "decision", "procedure"),
            start=1,
        )
    )


def _qualification():
    case = ReconstructionQualificationCase(
        case_id="multi-hop",
        declared_multi_hop=True,
        one_hop_coverage=0.5,
        reconstruction_coverage=0.9,
        one_hop_latency_ms=5,
        reconstruction_latency_ms=20,
        latency_budget_ms=50,
        one_hop_tokens=20,
        reconstruction_tokens=80,
        token_budget=100,
        reconstruction_steps=2,
    )
    return qualify_reconstructive_product(
        (case,),
        evidence_revision="sha256:" + "b" * 64,
    )


def test_exact_lookup_never_pays_reconstruction_and_persists_abstention(tmp_path: Path) -> None:
    query = "exact src/planner.py"
    pinned = _pin(_frame(query, coverage=0.0))
    base = _plan(query, query_type="exact_lookup")
    compiled = compile_reconstructive_strategy(base, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        base,
        frame=pinned,
        conflict_count=2,
        temporal_ambiguity=True,
    )
    calls = 0

    def executor(_step, _budget):
        nonlocal calls
        calls += 1
        return ReconstructionObservation((), 0, 0, 0)

    store = ReconstructionTraceStore(tmp_path, workspace_id="alpha")
    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=executor,
        trace_store=store,
    )

    assert calls == 0
    assert result.evidence_ids == ()
    assert result.trace.disposition == "abstained"
    assert result.trace.reason == "exact_lookup_one_hop"
    assert store.read()[0]["prompt_visible"] is False


def test_only_declared_synthesis_temporal_conflict_or_low_coverage_triggers_escalation() -> None:
    sufficient = _pin(_frame("ordinary recall", coverage=0.9))
    ordinary = _plan("ordinary recall", query_type="broad_recall")
    assert decide_reconstruction_escalation(ordinary, frame=sufficient).escalate is False

    synthesis = _plan("compare plans", query_type="synthesis")
    synthesis_frame = _pin(_frame("compare plans", coverage=0.9))
    assert decide_reconstruction_escalation(
        synthesis,
        frame=synthesis_frame,
        product_qualification=_qualification(),
    ).reasons == ("synthesis",)

    temporal = _plan("when did it change", query_type="temporal")
    temporal_frame = _pin(_frame("when did it change", coverage=0.9))
    assert decide_reconstruction_escalation(
        temporal,
        frame=temporal_frame,
        temporal_ambiguity=True,
        product_qualification=_qualification(),
    ).reasons == ("temporal_ambiguity",)
    conflict = decide_reconstruction_escalation(
        ordinary,
        frame=sufficient,
        conflict_count=1,
        product_qualification=_qualification(),
    )
    assert conflict.reasons == ("conflict",)

    low = _pin(_frame("ordinary recall", coverage=0.1))
    assert decide_reconstruction_escalation(
        ordinary,
        frame=low,
        product_qualification=_qualification(),
    ).reasons == ("low_coverage",)


def test_widening_policy_can_block_every_reconstruction_trigger() -> None:
    frame = _pin(_frame("compare plans", coverage=0.0, widening_allowed=False))
    plan = _plan("compare plans", query_type="synthesis")

    decision = decide_reconstruction_escalation(
        plan,
        frame=frame,
        conflict_count=3,
        temporal_ambiguity=True,
    )

    assert decision.escalate is False
    assert decision.reasons == ("widening_not_allowed",)


def test_runner_enforces_step_candidate_token_and_latency_limits(tmp_path: Path) -> None:
    query = "How did Planner procedure change src/planner.py decision?"
    pinned = _pin(_frame(query))
    base = _plan(query, query_type="synthesis")
    compiled = compile_reconstructive_strategy(base, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(),
    )
    budgets: list[RecallBudget] = []

    def executor(step, budget: RecallBudget) -> ReconstructionObservation:
        budgets.append(budget)
        return ReconstructionObservation(
            evidence_ids=(f"evidence-{step.ordinal}",),
            tokens_used=min(5, budget.max_tokens),
            latency_ms=min(6, budget.max_latency_ms),
            coverage_after=min(0.8, 0.2 + step.ordinal * 0.1),
        )

    store = ReconstructionTraceStore(tmp_path, workspace_id="alpha")
    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=executor,
        budget=RecallBudget(
            max_latency_ms=20,
            max_steps=2,
            max_tokens=12,
            max_candidates=8,
        ),
        trace_store=store,
    )

    assert len(result.trace.steps) <= 2
    assert result.trace.total_candidates <= 8
    assert result.trace.total_tokens <= 12
    assert result.trace.total_latency_ms <= 20
    assert result.trace.disposition == "partial"
    assert result.trace.reason == "hard_budget_exhausted"
    assert all(budget.max_steps <= 2 for budget in budgets)


def test_trace_records_step_benefit_and_stays_out_of_prompt(tmp_path: Path) -> None:
    query = "compare Planner procedure decisions"
    pinned = _pin(_frame(query, coverage=0.2))
    base = _plan(query, query_type="synthesis")
    compiled = compile_reconstructive_strategy(base, frame=pinned, index_nodes=_nodes())
    decision = decide_reconstruction_escalation(
        base,
        frame=pinned,
        product_qualification=_qualification(),
    )

    def executor(step, _budget) -> ReconstructionObservation:
        return ReconstructionObservation(
            evidence_ids=(f"record-evidence-{step.ordinal}",),
            tokens_used=10,
            latency_ms=5,
            coverage_after=0.95,
        )

    store = ReconstructionTraceStore(tmp_path, workspace_id="alpha")
    result = run_reconstructive_lite(
        compiled,
        escalation=decision,
        executor=executor,
        trace_store=store,
    )
    persisted = store.read()[0]

    assert result.trace.reason == "target_coverage_reached"
    assert result.trace.steps[0].useful is True
    assert result.trace.steps[0].coverage_after > result.trace.steps[0].coverage_before
    assert result.trace.prompt_visible is False
    assert result.trace.host_actions_allowed is False
    assert persisted["trace_id"] == result.trace.trace_id
    assert query not in store.path.read_text(encoding="utf-8")
