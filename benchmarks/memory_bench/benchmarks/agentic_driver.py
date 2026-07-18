"""Common driver for agentic benchmarks (MemoryArena, EvoMemBench).

Agentic benchmarks differ from recall benchmarks in that the agent
interacts with an environment over multiple steps, rather than answering
a single question.  Memory is used in isolation mode:

  loop:
    obs = env.reset() or env.step(action)
    context = store.retrieve(obs_text, k=k)
    action = policy_llm(obs, context)
    store.write(MemoryRecord(obs + action))   # retain experience

The fixed AgentPolicy (LLM) and Environment are supplied by each benchmark's
loader.  The runner calls run_agentic_episode() for each probe.
"""

import logging
import time
from dataclasses import replace
from typing import Any

from memory_bench.core.types import CaseResult, Episode, Probe, RunSpec

log = logging.getLogger(__name__)

# Maximum number of steps per episode (safety guard)
_MAX_STEPS = 50


def run_agentic_episode(
    *,
    provider: Any,
    provider_id: str | None = None,
    bench: Any,
    episode: Episode,
    probe: Probe,
    store: Any,
    spec: RunSpec,
) -> CaseResult:
    """Run one agentic probe and return a CaseResult.

    The environment and agent policy are taken from probe.metadata (set by
    the benchmark loader):
      probe.metadata["env"]    -> Environment instance
      probe.metadata["policy"] -> AgentPolicy instance (optional; defaults to NullPolicy)
    """
    t0 = time.monotonic()
    provider_label = provider_id or getattr(provider, "name", str(provider))

    env = probe.metadata.get("env")
    policy = probe.metadata.get("policy")

    if env is None:
        return CaseResult(
            provider=provider_label,
            benchmark=bench.name,
            episode_id=episode.id,
            probe_id=probe.id,
            category=probe.category,
            status="skipped",
            error="no environment provided for agentic probe",
            latency_ms=(time.monotonic() - t0) * 1000,
            diagnostics={
                "question": probe.question,
                "failure_mode": "agentic_skipped_no_environment",
            },
        )

    if policy is None:
        from memory_bench.benchmarks._null_policy import NullPolicy

        policy = NullPolicy()

    try:
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < _MAX_STEPS:
            obs_text = _obs_to_text(obs)
            store.retrieve(
                obs_text,
                k=spec.limits.recall_k,
                max_tokens=spec.limits.max_context_tokens,
            )
            action = policy.act(obs, store)
            # Retain observation + action in memory
            from memory_bench.core.types import MemoryRecord

            store.write(
                MemoryRecord(
                    id=f"{probe.id}_step{steps}",
                    content=f"Observation: {obs_text}\nAction: {action}",
                    kind="observation",
                    tags=("agentic", probe.id),
                )
            )
            obs, reward, done, info = env.step(action)
            steps += 1

        env_score = env.score()
        # Inject env_score into probe metadata for the task_success judge
        augmented_probe = replace(probe, metadata={**probe.metadata, "env_score": env_score})

        from memory_bench.adapters.judges import TaskSuccessJudge
        from memory_bench.core.types import Answer

        judge = TaskSuccessJudge()
        answer = Answer(text=str(env_score))
        judgment = judge.judge(augmented_probe, answer)

        return CaseResult(
            provider=provider_label,
            benchmark=bench.name,
            episode_id=episode.id,
            probe_id=probe.id,
            category=probe.category,
            status="ok",
            judgment=judgment,
            retrieval={"n_steps": steps},
            latency_ms=(time.monotonic() - t0) * 1000,
            diagnostics={
                "question": probe.question,
                "failure_mode": "ok" if judgment.correct else "agentic_task_score_below_threshold",
                "env_score": env_score,
                "success_threshold": float(probe.metadata.get("success_threshold", 0.5)),
                "n_steps": steps,
            },
        )
    except Exception as exc:
        log.error("agentic episode failed (probe=%s): %s", probe.id, exc, exc_info=True)
        return CaseResult(
            provider=provider_label,
            benchmark=bench.name,
            episode_id=episode.id,
            probe_id=probe.id,
            category=probe.category,
            status="error",
            error=str(exc),
            latency_ms=(time.monotonic() - t0) * 1000,
            diagnostics={
                "question": probe.question,
                "failure_mode": "agentic_exception",
                "exception_type": type(exc).__name__,
            },
        )


def _obs_to_text(obs: Any) -> str:
    """Convert an observation dict or string to text for retrieve()."""
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(obs)
