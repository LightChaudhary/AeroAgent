"""Core async state machine and agent loop."""

from __future__ import annotations
import asyncio
import json
from typing import Any, Callable

from .state import AgentState, AgentStep
from .llm import LLMClient

class Agent:
    """Async AI agent that plans, routes to tools, and tracks state."""
    
    def __init__(self, llm_client: LLMClient, tools: dict[str, Callable[..., Any]] | None = None, max_steps: int = 3,):
        self.llm = llm_client
        self.tools = tools or {}
        self.max_steps = max_steps

    async def run(self, prompt: str) -> AgentState:
        """Execute the agent loop for a given prompt."""
        state = AgentState(prompt=prompt, status="running")
        state.add_step("think", output="Initializing agent loop...")

        system_prompt = (
            "You are a helpful AI assistant. You must respond ONLY in valid JSON format.\n"
            "Choose ONE action per step:\n"
            "1. 'think': to reason about the next step.\n"
            "2. 'call_tool': to use an available tool.\n"
            f"Available tools: {', '.join(self.tools.keys()) or 'none'}\n"
            "JSON Schema: {\"action\": \"think\"|\"call_tool\"|\"finalize\", \"tool_name\": \"...\", \"tool_input\": {...}, \"thought\": \"...\", \"final_answer\": \"...\"}"
        )

        for step_num in range(1, self.max_steps + 1):
            messages = [
                {
                    "role": "system", "content": system_prompt
                },
                {
                    "role": "user", "content": f"Current State: {state.model_dump_json()}\nUser Prompt: {prompt}"
                }
            ]
            
            try: 
                # 1. Get LLM decision
                response = await self.llm.chat_completion(messages)
                content = response["choices"][0]["message"]["content"]
                decision = self.llm.extract_json(content)

                if not decision or "action" not in decision:
                    state.add_step("error", output=f"Failed to parse LLM JSON: {content}")
                    state.status = "error"
                    break

                action = decision.pop("action")
                state.add_step(action, **decision)

                # 2. Execute Action
                if action == "call_tool":
                    tool_name = decision.get("tool_name")
                    tool_input = decision.get("tool_input", {})

                    if tool_name in self.tools:
                        state.add_step("think", output=f"Executing tool: {tool_name}")
                        try:
                            # Execute tool (sync or async)
                            result = self.tools[tool_name](**tool_input)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step("call_tool", tool_name=tool_name, tool_input=tool_input, output=str(result))
                        except Exception as e:
                            state.add_step("error", tool_name=tool_name, output=f"Tool execution failed: {e}")
                    else:
                        state.add_step("error", output=f"Tool '{tool_name}' not found.")

                elif action == "finalize":
                    state.final_answer = decision.get("final_answer", "Task completed.")
                    state.status = "completed"
                    break

                elif action == "think":
                    continue # Loop continues

            except Exception as e:
                state.add_step("error", output=f"Agent loop crashed: {e}")
                state.status = "error"
                break
        
        if state.status == "running":
            state.status = "completed"
            state.final_answer = state.final_answer or "Max steps reached. Here is the current state: " + state.model_dump_json()

        return state 