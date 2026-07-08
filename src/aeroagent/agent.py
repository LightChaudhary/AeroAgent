"""Core async state machine and agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .llm import LLMClient
from .state import AgentState
from .tools.memory import save_to_memory as _save_to_memory_backend


class Agent:
    """Async AI agent that plans, routes to tools, and tracks state."""

    def __init__(
            self,
            llm_client: LLMClient,
            tools: dict[str, Callable[..., Any]] | None = None,
            max_steps: int = 8,
            max_tool_calls: int = 2,
    ):
        self.llm = llm_client
        self.tools = tools or {}
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    async def _auto_save_to_memory(self, text: str, state: AgentState) -> None:
        """
        Automatically save web search results to memory.
        Called deterministically after every web_search — no LLM involvement.
        """
        try:
            result = _save_to_memory_backend(text=text)
            if asyncio.iscoroutine(result):
                result = await result
            state.add_step(
                "call_tool",
                tool_name="save_to_memory",
                tool_input={"text": text},
                output=str(result),
            )
        except Exception as e:
            state.add_step("error", tool_name="save_to_memory", output=f"Auto-save failed: {e}")

    async def _synthesize_answer(self, prompt: str, state: AgentState) -> str:
        """Use the LLM to synthesize a final answer from collected tool results."""
        content_steps = [
            s for s in state.steps
            if s.action == "call_tool" and s.tool_name != "save_to_memory"
        ]

        if not content_steps:
            return "Agent reached max steps without gathering any information."

        def _extract_relevant(output: str, query: str, max_chars: int = 3000) -> str:
            """Return the slice of output most likely to contain query-relevant content."""
            lower = output.lower()
            idx = lower.find(query.lower())
            if idx != -1:
                # Centre a window around the first mention of the query
                start = max(0, idx - 300)
                return output[start:start + max_chars]
            return output[:max_chars]

        results_text = "\n\n".join(
            f"--- {s.tool_name} ---\n{_extract_relevant(s.output, prompt)}"
            for s in content_steps
        )

        synthesis_messages = [
            {
                "role": "user",
                "content": (
                    f"User Question: {prompt}\n\n"
                    f"The following information was gathered:\n\n{results_text}\n\n"
                    "Based on this information, provide a clear and concise answer "
                    "to the user's question. Reply in plain text only — no JSON, "
                    "no markdown fences, no preamble."
                ),
            }
        ]

        try:
            # Use plain-text mode so the model answers naturally without
            # being forced into a JSON wrapper by response_format:json_object.
            resp = await self.llm.chat_completion_text(synthesis_messages)
            answer = resp["choices"][0]["message"]["content"].strip()
            metrics = resp.get("_metrics") or {}
            state.add_step(
                "think",
                output="Synthesis LLM call completed.",
                latency_ms = metrics.get("latency_ms"),
            )
            return answer if answer else "Agent gathered results but could not synthesize an answer."
        except Exception as e:
            state.add_step("error", output=f"Synthesis LLM call failed: {e}")
            last = content_steps[-1]
            return last.output

    async def run(self, prompt: str) -> AgentState:
        """Execute the agent loop for a given prompt."""
        state = AgentState(prompt=prompt, status="running")
        state.add_step("think", output="Initializing agent loop...")

        llm_visible_tools = {k: v for k, v in self.tools.items() if k != "save_to_memory"}

        system_prompt = (
            "You are a helpful AI assistant with memory. Follow this EXACT workflow:\n\n"
            "STEP 1: ALWAYS call search_memory first to check existing knowledge.\n"
            "STEP 2: If memory returned 'No relevant memory found', call web_search ONCE.\n"
            "        If memory returned useful results, skip to STEP 3.\n"
            "STEP 3: Once you have results (from memory OR web_search), you MUST finalize.\n\n"
            "CRITICAL RULES:\n"
            "- NEVER skip search_memory.\n"
            "- NEVER call web_search more than once.\n"
            "- NEVER call save_to_memory — it is handled automatically.\n"
            "- After gathering results, ALWAYS output a finalize action with your answer.\n"
            "- The 'final_answer' field is REQUIRED in the finalize action.\n\n"
            f"Available tools: {', '.join(llm_visible_tools.keys())}\n\n"
            "Respond with ONLY valid JSON — no markdown, no explanation, no extra text.\n\n"
            "JSON schemas:\n"
            '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "your query"}}\n'
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "your query"}}\n'
            '{"action": "finalize", "final_answer": "your complete answer here — this field is required"}\n'
        )

        for step_num in range(1, self.max_steps + 1):
            tool_steps = [
                s for s in state.steps
                if s.action == "call_tool" and s.tool_name != "save_to_memory"
            ]

            # Once we've hit max tool calls, force synthesis and exit
            if len(tool_steps) >= self.max_tool_calls:
                state.add_step("think", output="Max tool calls reached. Synthesizing answer.")
                break

            # Build LLM context from non-internal steps
            tool_results = "\n".join(
                f"Tool '{s.tool_name}' returned: {s.output[:2000]}"
                for s in tool_steps
            )

            # Annotate memory result so LLM knows whether to web_search
            memory_step = next(
                (s for s in tool_steps if s.tool_name == "search_memory"), None
            )
            memory_hint = ""
            if memory_step:
                has_hit = "No relevant memory found" not in memory_step.output
                memory_hint = (
                    "\nMEMORY STATUS: Results found — do NOT call web_search. Go straight to finalize.\n"
                    if has_hit else
                    "\nMEMORY STATUS: No results — call web_search once, then finalize.\n"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User Prompt: {prompt}\n\n"
                        f"Tools already called: {[s.tool_name for s in tool_steps]}\n\n"
                        f"Tool Results:\n{tool_results or 'None yet'}\n"
                        f"{memory_hint}\n"
                        "Now decide your next action. Respond with ONLY valid JSON."
                    ),
                },
            ]

            try:
                # 1. Get LLM decision
                response = await self.llm.chat_completion(messages)
                content = response["choices"][0]["message"]["content"]
                decision_latency_ms = (response.get("_metrics") or {}).get("latency_ms")
                print("\n===== RAW LLM OUTPUT =====")
                print(content)
                print("==========================\n")
                decision = self.llm.extract_json(content)

                # Retry once if JSON extraction failed
                if not decision:
                    state.add_step("think", output="LLM returned invalid JSON. Retrying.")
                    retry_messages = messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Respond with ONLY valid JSON matching one of the schemas "
                                "in the system prompt. No markdown, no explanation."
                            ),
                        },
                    ]
                    retry_response = await self.llm.chat_completion(retry_messages)
                    retry_content = retry_response["choices"][0]["message"]["content"]
                    decision_latency_ms = (retry_response.get("_metrics") or {}).get("latency_ms")
                    decision = self.llm.extract_json(retry_content)

                    if not decision:
                        # Model still isn't producing JSON - it likely just answered conversationally
                        # Treat its raw text as the final answer rather than discarding it via generic synthesis.
                        raw_answer = (retry_content or content or "").strip()
                        state.add_step(
                            "think", 
                            output="Retry also returned invalid JSON. Falling back to synthesis."
                        )
                        state.final_answer = raw_answer or "Agent could not produce a valid response."
                        state.status = "completed"
                        state.add_step("finalize", output=state.final_answer, latency_ms=decision_latency_ms)
                        break

                # Defensive parsing: infer action if small model omitted the 'action' key
                if "action" not in decision:
                    if "tool_name" in decision or "tool_input" in decision:
                        decision["action"] = "call_tool"
                    elif "final_answer" in decision:
                        decision["action"] = "finalize"
                    elif tool_steps:
                        # We have results — treat as implicit finalize
                        decision["action"] = "finalize"
                        decision["final_answer"] = decision.get("final_answer", "")
                    else:
                        decision["action"] = "think"

                action = decision.get("action", "think")

                # 2. Execute action
                if action == "call_tool":
                    tool_name = decision.get("tool_name")
                    tool_input = decision.get("tool_input", {})

                    if isinstance(tool_input, str):
                        tool_input = {"query": tool_input}

                    # Block web_search if memory already returned useful results.
                    # llama3.2:3b ignores the prompt instruction reliably, so enforce it in code.
                    if tool_name == "web_search":
                        memory_step = next(
                            (s for s in state.steps if s.action == "call_tool" and s.tool_name == "search_memory"),
                            None,
                        )
                        if memory_step and "No relevant memory found" not in memory_step.output:
                            state.add_step("think", output="Blocked web_search: memory hit found. Moving to synthesis.")
                            break

                    # Block duplicate queries per tool
                    already_called_queries = {
                        s.tool_input.get("query", "") or s.tool_input.get("text", "")
                        for s in state.steps
                        if s.action == "call_tool" and s.tool_name == tool_name
                    }
                    new_query = tool_input.get("query", "") or tool_input.get("text", "")
                    if new_query and new_query in already_called_queries:
                        state.add_step("think", output=f"Blocked duplicate {tool_name} call. Moving to synthesis.")
                        break

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**tool_input)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step(
                                "call_tool",
                                tool_name=tool_name,
                                tool_input=tool_input,
                                output=str(result),
                            )

                            # Auto-save web search results to memory
                            if tool_name == "web_search":
                                await self._auto_save_to_memory(str(result), state)

                        except Exception as e:
                            state.add_step("error", tool_name=tool_name, output=f"Tool execution failed: {e}")
                    else:
                        state.add_step("error", output=f"Tool '{tool_name}' not found.")

                elif action == "finalize":
                    called_tools = [s.tool_name for s in state.steps if s.action == "call_tool"]

                    # Guard: force search_memory if it was skipped
                    if "search_memory" not in called_tools and self.tools:
                        state.add_step("think", output="Blocked premature finalize. Forcing search_memory.")
                        try:
                            result = self.tools["search_memory"](query=prompt)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step(
                                "call_tool",
                                tool_name="search_memory",
                                tool_input={"query": prompt},
                                output=str(result),
                            )
                        except Exception as e:
                            state.add_step("error", tool_name="search_memory", output=f"Tool execution failed: {e}")
                        continue  # Re-enter loop to let LLM decide next step

                    final_answer = decision.get("final_answer", "").strip()

                    if not final_answer:
                        # LLM forgot to include the answer — synthesize it
                        state.add_step("think", output="Finalize had no answer. Synthesizing from tool results.")
                        final_answer = await self._synthesize_answer(prompt, state)

                    state.final_answer = final_answer
                    state.status = "completed"
                    state.add_step("finalize", output=final_answer)
                    break

                elif action == "think":
                    think_count = sum(1 for s in state.steps if s.action == "think")
                    if think_count >= 3:
                        # Stuck in think loop — break and synthesize
                        state.add_step("think", output="Too many think steps. Moving to synthesis.")
                        break

            except Exception as e:
                state.add_step("error", output=f"Agent loop crashed: {e}")
                state.status = "error"
                break

        # Fallback: if loop exhausted or broke early without a final answer
        if state.status == "running":
            state.add_step("think", output="Loop exited without finalize. Running synthesis fallback.")
            state.final_answer = await self._synthesize_answer(prompt, state)
            state.status = "completed"
            state.add_step("finalize", output=state.final_answer)

        state.aggregate_metrics()
        return state