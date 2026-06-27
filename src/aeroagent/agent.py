"""Core async state machine and agent loop."""

from __future__ import annotations
import asyncio
from typing import Any, Callable

from .state import AgentState
from .llm import LLMClient

class Agent:
    """Async AI agent that plans, routes to tools, and tracks state."""
    
    def __init__(
            self,
            llm_client: LLMClient,
            tools: dict[str, Callable[..., Any]] | None = None,
            max_steps: int = 6,
            max_tool_calls: int = 3,
    ): 
        self.llm = llm_client
        self.tools = tools or {}
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    async def run(self, prompt: str) -> AgentState:
        """Execute the agent loop for a given prompt."""
        state = AgentState(prompt=prompt, status="running")
        state.add_step("think", output="Initializing agent loop...")

        system_prompt = (
            "You are a helpful AI assistant with memory. Follow this EXACT three-step workflow:\n\n"
            "STEP 1: ALWAYS call search_memory first to check if you already know the answer.\n"
            "STEP 2: If memory says 'No relevant memory found', call web_search ONCE.\n"
            "       If memory returned results, skip web_search and go directly to STEP 3.\n"
            "STEP 3: Finalize your answer, then call save_to_memory to persist what you learned.\n\n"
            "CRITICAL RULES:\n"
            "- Never skip search_memory - always check memory first.\n"
            "- Nver call web_seach if memory already has relevant results.\n"
            "- Never call web_search more than once.\n"
            "- Always call save_to_memory after finalizing.\n"
            "- Once you have results (from memory OR web), your next action MUST be finalize.\n\n"
            f"Available tools: {', '.join(self.tools.keys())}\n\n"
            "JSON schemas:\n"
            '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "your query"}}\n'
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "your query"}}\n'
            '{"action": "call_tool", "tool_name": "save_to_memory", "tool_input": {"text": "key facts to remember"}}\n'
            '{"action": "finalize", "final_answer": "your answer based on results"}\n'
        )

        for step_num in range(1, self.max_steps + 1):
            # Hard cap: force finalize if too many tool calls have been made
            tool_steps = [s for s in state.steps if s.action == "call_tool"]
            if len(tool_steps) >= self.max_tool_calls:
                # last = tool_steps[-1]
                # state.final_answer = last.output
                # state.status = "completed"
                break

            tool_results = "\n".join(
                f"Tool '{s.tool_name}' returned: {s.output[:2000]}"
                for s in state.steps 
                if s.action == "call_tool"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User Prompt: {prompt}\n\n"
                        f"Tools already called: {[s.tool_name for s in state.steps if s.action == 'call_tool']}\n\n"
                        f"Tool Results: \n{tool_results or 'None yet'}\n\n"
                        "Now decide your next action."
                    )
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
                            # Treat raw plain text as final answer instead of error
                            state.final_answer = content
                            state.status = "completed"
                            state.add_step("finalize", output=content)
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

                    # Block duplicate queries 
                    already_called={
                        s.tool_input.get("query", "") or s.tool_input.get("text", "")
                        for s in state.steps
                        if s.action == "call_tool" and s.tool_name == tool_name
                    }
                    new_input = tool_input.get("query", "") or tool_input.get("text", "")
                    if new_input in already_called and new_input:
                        # Force finalize instead
                        #last = next((s for s in reversed(state.steps) if s.action == "call_tool"), None)
                        #state.final_answer = last.output if last else "No results found."
                        #state.status = "completed"
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

                            # Auto finalize after first successful search for small models.
                            # Remove or increase max_tool_calls in __init__ to allow chaining
                            #tool_calls = [s for s in state.steps if s.action == "call_tool"]
                            #if len(tool_calls) >= self.max_too_calls:
                                #state.final_answer = str(result)
                                #state.status = "completed"
                                #break
                        except Exception as e:
                            state.add_step("error", tool_name=tool_name, output=f"Tool execution failed: {e}")
                    else:
                        state.add_step("error", output=f"Tool '{tool_name}' not found.")

                elif action == "finalize":
                    # Safety gate: must have called search_memory at minimum
                    called_tools = [s.tool_name for s in state.steps if s.action == "call_tool"]
                    has_memory_search = "search_memory" in called_tools
                    has_web_search = "web_search" in called_tools

                    if not has_memory_search and self.tools:
                        # Force search_memory first
                        state.add_step("think", output="Blocked premature finalize. Forcing search_memory.")
                        try:
                            result = self.tools["search_memory"](query=prompt)
                            if asyncio.iscoroutine(result):
                                result = await result
                            state.add_step(
                                "call_tool",
                                tool_name="search_memory",
                                tool_input={"query": prompt},
                                output= str(result),
                            )
                        except Exception as e:
                            state.add_step("error", tool_name="search_memory", output=f"Tool execution failed: {e}")
                    
                    elif has_memory_search and not has_memory_search:
                        # Check if memory had a hit - if not, force web_search
                        memory_result = next(
                            (s.output for s in state.steps if s.tool_name == "search_memory"), ""
                        )
                        if "No relevant memory found" in memory_result:
                            state.add_step("think", output="Memory miss. Forcing web_search.")
                            try:
                                result = self.tools["web_search"](query=prompt)
                                if asyncio.iscoroutine(result):
                                    result = await result
                                state.add_step(
                                    "call_tool",
                                    tool_name="web_search",
                                    tool_input={"query": prompt},
                                    output=str(result),
                                )
                            except Exception as e:
                                state.add_step("error", tool_name="web_search", output=f"Tool execution failed: {e}")
                        else:
                            # Memory hit - finalize directly
                            state.final_answer = decision.get("final_answer", "Task completed.")
                            state.status = "completed"
                            break
                    else:
                        # Has both memory search + web search (or memory hit) - finalize                    
                        state.final_answer = decision.get("final_answer", "Task completed.")
                        state.status = "completed"
                        break

                elif action == "think":
                    think_count = sum(1 for s in state.steps if s.action == "think")
                    if think_count >= 2 and not any(s.action == "call_tool" for s in state.steps):
                        state.add_step("think", output="Reminder: Call search_memory first, then web_search if needed.")

            except Exception as e:
                state.add_step("error", output=f"Agent loop crashed: {e}")
                state.status = "error"
                break
        
        if state.status == "running":
            # Loop exhausted without completing - salvage last tool result.
            last = next((s for s in reversed(state.steps) if s.action == "call_tool"), None)
            state.final_answer = last.output if last else "Agent reached max steps without an answer."
            state.status = "completed"

        return state