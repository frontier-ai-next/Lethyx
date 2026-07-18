"""Responder implementations for memory_bench.

LLMResponder — the standard reader LLM used in isolation-mode evaluation.
EchoResponder — deterministic test double that returns a preset answer.
"""

import logging

from memory_bench.core.llm import LLMConfig, call_llm
from memory_bench.core.text import format_context, is_abstain_response
from memory_bench.core.types import Answer, RetrievedItem, Usage

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a factual question-answering assistant.
Answer the question using ONLY the information provided in the memory context below.
If the answer cannot be found in the context, respond with exactly: I don't know
Do not hallucinate or use knowledge outside the provided context.
Keep your answer concise (1-3 sentences).
"""


class LLMResponder:
    """Fixed reader LLM for isolation-mode benchmarking.

    The same LLMResponder instance is shared across all providers within a
    benchmark run so the LLM is the constant and memory quality is the variable.

    The system prompt enforces strict context-only answering and "I don't know"
    for abstention, required by LongMemEval and BEAM abstention categories.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()

    def answer(
        self,
        question: str,
        context: list[RetrievedItem],
        *,
        instructions: str = "",
    ) -> Answer:
        context_text = format_context(
            context,
            max_tokens=self._config.max_tokens * 3,
            include_metadata=True,
        )
        system = _SYSTEM_PROMPT + (f"\n{instructions}" if instructions else "")
        user = f"## Memory context\n\n{context_text}\n\n## Question\n\n{question}"

        if not context:
            user = (
                f"## Memory context\n\n(no relevant memory retrieved)\n\n## Question\n\n{question}"
            )

        try:
            text, usage = call_llm(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                config=self._config,
            )
            abstained = is_abstain_response(text)
            return Answer(text=text, abstained=abstained, usage=usage, raw=text)
        except Exception as exc:
            log.error("LLMResponder.answer failed: %s", exc)
            return Answer(
                text="",
                abstained=True,
                usage=Usage(),
                raw=str(exc),
            )


class EchoResponder:
    """Test double that returns a preset answer without calling any LLM.

    Useful for unit tests of the benchmark pipeline without network access.
    ``answers`` maps question strings to response strings; unmatched questions
    return ``default``.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        *,
        default: str = "I don't know",
    ) -> None:
        self._answers = dict(answers or {})
        self._default = default

    def answer(
        self,
        question: str,
        context: list[RetrievedItem],
        *,
        instructions: str = "",
    ) -> Answer:
        text = self._answers.get(question, self._default)
        abstained = is_abstain_response(text)
        return Answer(text=text, abstained=abstained, usage=Usage(calls=0), raw=text)
