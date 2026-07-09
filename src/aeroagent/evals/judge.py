"""LLM-as-a-judge scoring for agent eval runs.

Uses the same LLMClient as the agent itself (JSON mode) to score a completed AgentState against an
EvalCase's criteria. Scoring is 1-5:
    1 = completely wrong or unusable
    3 = partially correct or on-topic but incomplete/imprecise
    5 = fully satisfies the criteria

Keep the judge prompt small and structered - small local models are inconsistent judges for anything
beyond a simple rubric check.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..state import AgentState
from .dataset import EvalCase


class JudgeResult(BaseModel):
    """Structured output from the judge model for a single eval case."""

    case_id: str
    score: int = Field(ge=1, le=5)
    rationale: str
    raw_answer: str


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grading assistant. You will be give a QUESTION, "
    "CRITERIA for a good answer, and an ANSWER produced by an AI agent. "
    "Score the Answer from 1 to 5 based only on the CRITERIA:\n"
    "1 = completely wrong or unusable\n"
    "2 = mostly wrong, minor relevant content\n"
    "3 = partially correct or on-topic but incomplete or imprecise\n"
    "4 = mostly correct with minor issues\n"
    "5 = fully satisfies the criteria\n\n"
    "Respond with ONLY vaild JSON, no markdown, no explanation outside the JSON:\n"
    '{"score": <1-5 integer>, "rationale": "<one or two sentence justification>"}\n'
)


async def judge_case(
    llm_client: LLMClient,
    eval_case: EvalCase,
    agent_state: AgentState,
) -> JudgeResult:
    """Score a completed agent run against its eval case criteria.

    Falls back to a score of 1 with an explanatory rationale if the judge model's output can't be parsed,
    rather than raising - a single bad judge call shouldn't crash a full eval run.
    """
    raw_answer = agent_state.final_answer or ""

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {eval_case.prompt}\n\n"
                f"CRITERIA: {eval_case.criteria}\n\n"
                f"ANSWER: {raw_answer}\n\n"
                "Respond with ONLY the JSON object."
            ),
        },
    ]

    try:
        response = await llm_client.chat_completion(messages, temperature=0.0)
        content = response["choices"][0]["message"]["content"]
        parsed = llm_client.extract_json(content)

        if not parsed or "score" not in parsed:
            return JudgeResult(
                case_id=eval_case.case_id,
                score=1,
                rationale="Judge output could not be parsed as valid JSON.",
                raw_answer=raw_answer,
            )

        score = int(parsed["score"])
        score = max(1, min(5, score))  # clamp defensively

        return JudgeResult(
            case_id=eval_case.case_id,
            score=score,
            rationale=str(parsed.get("rationale", "")),
            raw_answer=raw_answer,
        )
    except Exception as e:
        return JudgeResult(
            case_id=eval_case.case_id,
            score=1,
            rationale=f"Judge call failed: {e}",
            raw_answer=raw_answer,
        )
