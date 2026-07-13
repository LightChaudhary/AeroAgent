"""CLI entry point for the AeroAgent framework."""

import asyncio
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.traceback import install

from src.aeroagent.agent import Agent
from src.aeroagent.llm import DEFAULT_MODEL, LLMClient
from src.aeroagent.tools.memory import search_memory

# Import tools to trigger registration and expose the callable
from src.aeroagent.tools.search import web_search
from src.aeroagent.tracer import tracer

# Enable rich, readable tracebacks for debugging
install()


async def run_agent_cli(prompt: str) -> None:
    """Execute the agent loop and display results via CLI."""
    console = Console()
    console.print(
        Panel(
            f"[cyan] Starting Agent with prompt:[/cyan]\n{prompt}", border_style="blue"
        )
    )

    # 1. Initialize components
    # Note: Ensure you have pulled this model via 'ollama pull llama3.2:3b'
    llm = LLMClient(model=DEFAULT_MODEL)
    tools = {
        "web_search": web_search,
        "search_memory": search_memory,
    }
    agent = Agent(llm_client=llm, tools=tools, max_steps=6)

    try:
        # 2. Run the agent
        console.print("[yellow] Agent is thinking and executing tools...[/yellow]")
        state = await agent.run(prompt)

        # 3. Trace the execution for debugging/observability
        trace_path = await tracer.save_trace(
            state, metadata={"model": DEFAULT_MODEL, "interface": "cli"}
        )
        console.print(f"[dim] Trace saved to: {trace_path}[/dim]")

        # 4. Display results
        if state.status == "completed" and state.final_answer:
            console.print(
                Panel(
                    Markdown(state.final_answer),
                    title="Final Answer",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    state.error_message or "Agent finished without a final answer.",
                    title="Status",
                    border_style="yellow",
                )
            )

            # Show execution steps for debugging
            console.print("\n[bold]Execution Steps:[/bold]")
            for step in state.steps:
                detail = step.output or step.tool_name or "No detail"
                console.print(f"    - [bold]{step.action.upper()}[/bold]: {detail}")

    except Exception as e:
        console.print(f"[red] Fatal Error: {e}[/red]")
        raise
    finally:
        # Clean up HTTP connections
        await llm.close()


def main() -> None:
    if len(sys.argv) < 2:
        console = Console()
        console.print(
            Panel(
                '[bold]Usage:[/bold] python main.py "Your prompt here"\n'
                '[dim]Example: python main.py "What are the new features in Python 3.13?"[/dim]',
                title="AeroAgent CLI",
                border_style="red",
            )
        )
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    asyncio.run(run_agent_cli(prompt))


if __name__ == "__main__":
    main()
