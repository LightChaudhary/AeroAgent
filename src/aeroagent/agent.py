"""Core async state machine and agent loop."""

from __future__ import annotations
import asyncio
from typing import Any, Callable

from .state import AgentState
from .llm import LLMClient

class Agent:
    """Async AI agent that plans, routes to tools, and tracks state."""
    
    def __init__(self, llm_client: LLMClient, tools: dict[str, Callable[..., Any]] | None = None, max_steps: int = 5):
        self.llm = llm_client
        self.tools = tools or {}
        self.max_steps = max_steps

    async def run(self, prompt: str) -> AgentState:
        """Execute the agent loop for a given prompt."""
        state = AgentState(prompt=prompt, status="running")
        state.add_step("think", output="Initializing agent loop...")

        system_prompt = (
            "You are a helpful AI assistant. You MUST follow this exact workflow:\n\n"
            "STEP 1: Use 'call_tool' action with web_search for ANY factual/current question.\n"
            "STEP 2: After receiving tool output, use 'finalize' action with the final_answer.\n\n"
            "CRITICAL RULES:\n"
            "- NEVER use 'think' action more than once.\n"
            "- Never answer from memory. ALWAYS use web_search first.\n"
            "- 'finalize' action REQUIRES a 'final_answer' field.\n"
            "- 'call_tool' action REQUIRES 'tool_name' and 'tool_input' fields.\n"
            "- In 'finalize', your final_answer MUST be based ONLY on the tool output in the current state.\n"
            "- If the tool output does not contain a clear answer, say: 'The search did not return a definitive answer.'\n"
            "- NEVER use your own memory for facts, dates, or versions.\n\n"
            f"Available tools: {', '.join(self.tools.keys())}\n"
            "JSON Schema examples:\n"
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "Python latest stable release version date 2026"}}\n'
            '{"action": "finalize", "final_answer": "Based on search results..."}\n'
        )

        for step_num in range(1, self.max_steps + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Current State: {state.model_dump_json()}\nUser Prompt: {prompt}"
                },
            ]

            try: 
                # 1. Get LLM decision
                response = await self.llm.chat_completion(messages)
                content = response["choices"][0]["message"]["content"]
                print("\n===== RAW LLM OUTPUT =====")
                print(content)
                print("==========================\n")
                decision = self.llm.extract_json(content)

                if not decision:
                    state.add_step("think", output="LLM returned invalid JSON. Requesting correction.")

                    retry_messages = messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user", 
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Respond with ONLY valid JSON matching the schema "
                                "from the system prompt. Do not include markdown, "
                                "explanations, or extra text."
                            ),
                        },
                    ]

                    retry_response = await self.llm.chat_completion(retry_messages)
                    retry_content = retry_response["choices"][0]["message"]["content"]
                    decision = self.llm.extract_json(retry_content)

                    if not decision:
                        last_tool_step = next(
                            (s for s in reversed(state.steps) if s.action == "call_tool"), None
                        )
                        if last_tool_step:
                            state.final_answer = last_tool_step.output
                            state.status = "completed"
                        else:
                            state.add_step("error", output=f"Failed to parse LLM JSON after retry.\nOriginal: {content}\nRetry: {retry_content}")
                            state.status = "error"
                        break

                # DEFENSIVE PARSING: Infer action if the small model forgot the 'action' key
                if "action" not in decision:
                    if "tool_name" in decision or "tool_input" in decision:
                        decision["action"] = "call_tool"
                    elif "final_answer" in decision:
                        decision["action"] = "finalize"
                    elif any(s.action == "call_tool" for s in state.steps):
                        decision["action"] = "finalize"
                        decision["final_answer"] = ", ".join(f"{k}: {v}" for k, v in decision.items())
                    else:
                        decision["action"] = "think"

                action = decision.get("action", "think")

                # 2. Execute Action
                if action == "call_tool":
                    tool_name = decision.get("tool_name")
                    tool_input = decision.get("tool_input", {})

                    if isinstance(tool_input, str):
                        tool_input = {"query": tool_input}

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**tool_input)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step("call_tool", tool_name=tool_name, tool_input=tool_input, output=str(result))
                        except Exception as e:
                            state.add_step("error", tool_name=tool_name, output=f"Tool execution failed: {e}")
                    else:
                        state.add_step("error", output=f"Tool '{tool_name}' not found.")

                elif action == "finalize":
                    # Safety gate: block finalize if no tool has been called yet
                    if not any(s.action == "call_tool" for s in state.steps):
                        state.add_step("think", output="Blocked premature finalize. Forcing web_search.")
                        tool_input = {"query": prompt}
                        try:
                            result = self.tools["web_search"](**tool_input)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step("call_tool", tool_name="web_search", tool_input=tool_input, output=str(result))
                        except Exception as e:
                            state.add_step("error", tool_name="web_search", output=f"Tool execution failed: {e}")
                    else:
                        state.final_answer = decision.get("final_answer", "Task completed.")
                        state.status = "completed"
                        break

                elif action == "think":
                    think_count = sum(1 for s in state.steps if s.action == "think")
                    if think_count >= 2 and not any(s.action == "call_tool" for s in state.steps):
                        state.add_step("think", output="Reminder: You MUST use the web_search tool for factual questions.")

            except Exception as e:
                state.add_step("error", output=f"Agent loop crashed: {e}")
                state.status = "error"
                break
        
        if state.status == "running":
            state.status = "completed"
            state.final_answer = state.final_answer or "Unable to complete the task within the maximum number of steps."

        return state