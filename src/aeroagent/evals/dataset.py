"""Eval case definitions and a small starter dataset for the agent.

Each EvalCase pairs a prompt with a rubric the judge model uses to score the agent's response. Keep
criteria short and checkable - the judge is a small local model too, so vague or compound criteria
produce noisy scores.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single test case for the eval harness."""

    case_id: str
    prompt: str
    criteria: str = Field(
        description="What a correct/good answer must contain or do. "
        "Used verbatim in the judge prompt."
    )
    category: str = Field(
        default="general",
        description="Rough grouping for reporting, e.g. 'factual', 'memory', 'ambiguous'.",
    )


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="factual_001",
        prompt="What is the capital of Japan?",
        criteria="The answer must state that Tokyo is the capital of Japan.",
        category="factual",
    ),
    EvalCase(
        case_id="factual_002",
        prompt="Who wrote the novel 'Pride and Prejudice'?",
        criteria="The answer must name Jane Austen as the author.",
        category="factual",
    ),
    EvalCase(
        case_id="current_events_001",
        prompt="What is the latest stable version of Python?",
        criteria="The answer must give a specific Python version number (e.g '3.13'), not a vague or refused response.",
        category="current_events",
    ),
    EvalCase(
        case_id="ambiguous_001",
        prompt="Tell me about Mercury.",
        criteria="The answer should address that 'Mercury' is ambiguous (planet vs. element) or clearly pick one "
        "interpretation and answer it correctly, without simply refusing to answer.",
        category="ambiguous",
    ),
    EvalCase(
        case_id="memory_001",
        prompt="What did we discuss earlier about local LLM quantization?",
        criteria="If prior memory exists on this topic, the answer should reference it. If no memory exists, the "
        "answer should say so clearly rather than fabricating a prior conversation.",
        category="memory",
    ),
]
