"""Runs the real Agent over the eval dataset and produces a scored report.

This module drives actual LLM calls (both the agent and the judge), so it needs Ollama running
locally - unlike the unit tests, which mock LLMClient entirely. Run it directly:

    python -m src.aeroagent.evals.runner
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agent import Agent
from ..llm import LLMClient
from ..prompts.registry import DEFAULT_PROMPT_VERSION
from ..tools.memory import search_memory
from ..tools.search import web_search
from .dataset import EVAL_CASES, EvalCase
from .judge import judge_case

REPORT_DIR = Path("traces/evals")


def _build_agent_tools() -> dict[str, Any]:
    """Tools available to the agent under evaluation.

    save_to_memory is deliberately excluded - it's auto-invoked by the agent loop itself, never
    directy by the LLM.
    """
    return {
        "search_memory": search_memory,
        "web_search": web_search,
    }


async def run_eval_case(
    agent_llm: LLMClient,
    judge_llm: LLMClient,
    eval_case: EvalCase,
    prompt_version: str,
) -> dict[str, Any]:
    """Run one eval case end-to-end: agent run -> judge score -> combined record."""
    agent = Agent(
        llm_client=agent_llm,
        tools=_build_agent_tools(),
        prompt_version=prompt_version,
    )
    state = await agent.run(eval_case.prompt)
    result = await judge_case(judge_llm, eval_case, state)

    return {
        "case_id": eval_case.case_id,
        "category": eval_case.category,
        "prompt": eval_case.prompt,
        "criteria": eval_case.criteria,
        "prompt_version": state.prompt_version,
        "final_answer": state.final_answer,
        "status": state.status,
        "score": result.score,
        "rationale": result.rationale,
        "metrics": state.metrics,
    }


async def run_eval_suite(
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    cases: list[EvalCase] | None = None,
) -> dict[str, Any]:
    """Run the full eval dataset sequentially and build a summary report.

    Sequential (not concurrent) on purpose: local Ollama typically serves one request at a time efficiently,
    and sequential runs keep judge scoring easy to read in order while debugging.
    """
    cases = cases if cases is not None else EVAL_CASES
    agent_llm = LLMClient()
    judge_llm = LLMClient()

    results = []
    try:
        for case in cases:
            record = await run_eval_case(agent_llm, judge_llm, case, prompt_version)
            results.append(record)
            print(
                f"[{record['case_id']}] score={record['score']} status={record['status']}"
            )
    finally:
        await agent_llm.close()
        await judge_llm.close()

    scores = [r["score"] for r in results]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "case_count": len(results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "results": results,
    }
    return summary


def save_report(summary: dict[str, Any]) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filepath = REPORT_DIR / f"eval_report_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return str(filepath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AeroAgent eval suite.")
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt registry version to evaluate (default: %(default)s)",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_eval_suite(prompt_version=args.prompt_version))
    filepath = save_report(summary)

    print(
        f"\nAverage score: {summary['average_score']}/5 across {summary['case_count']} cases"
    )
    print(f"Report saved to: {filepath}")


if __name__ == "__main__":
    main()
