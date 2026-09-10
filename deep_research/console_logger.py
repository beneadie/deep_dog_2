"""
Console Logger for Deep Research Agent (runtime-aware).

Shows real-time console output for agent activities. Console output is gated
by the active runtime context's ``console_enabled`` flag:

  - CLI / standalone runs (console_enabled=True) print as before.
  - API runs (console_enabled=False) suppress console chatter; the host
    receives structured events via the observer instead.

Sub-agent id allocation and start/end timing live on the per-run Observer so
concurrent runs never share counters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from deep_research.runtime import get_runtime

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

# Emojis for different actions
EMOJI_SUPERVISOR = "🔬"
EMOJI_THINK = "💭"
EMOJI_SEARCH = "🔍"
EMOJI_TOOLS = "📋"
EMOJI_RESEARCH = "🔎"
EMOJI_DISCOVERY = "🧭"
EMOJI_REFINE = "✏️"
EMOJI_COMPLETE = "✅"
EMOJI_ERROR = "❌"


def _console() -> bool:
    """True when the active runtime wants console output."""
    try:
        return bool(get_runtime().console_enabled)
    except Exception:
        return True


def _timestamp() -> str:
    """Get current timestamp for logging."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ===== SUPERVISOR LOGGING =====

def log_supervisor_start(iteration: int, elapsed_minutes: float) -> None:
    """Log the start of a supervisor iteration."""
    if not _console():
        return
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{EMOJI_SUPERVISOR} SUPERVISOR | Iteration {iteration} | {elapsed_minutes:.1f} min elapsed{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 70}{Colors.RESET}")


def log_supervisor_thinking(purpose: str, reflection: str) -> None:
    """Log supervisor's thinking/reflection (purpose + denoise verdict when present)."""
    if not _console():
        return
    verdict = ""
    for v in ("CONTINUE_RESEARCH", "READY_TO_CONCLUDE", "TIME_LIMIT"):
        if f"VERDICT: {v}" in reflection:
            verdict = f" [{v}]"
    print(f"   {EMOJI_THINK} {Colors.GRAY}THINK({purpose}){verdict}:{Colors.RESET}")
    for line in reflection.splitlines() or [""]:
        print(f"      {Colors.GRAY}{line}{Colors.RESET}")


def log_supervisor_tool_calls(tool_calls: List[Dict[str, Any]]) -> None:
    """Log the tools the supervisor is calling."""
    if not _console():
        return
    if not tool_calls:
        print(f"   {Colors.YELLOW}(No tool calls){Colors.RESET}")
        return

    print(f"   {EMOJI_TOOLS} {Colors.BOLD}TOOLS TO CALL:{Colors.RESET}")

    research_tools = {
        "ResearchWeb", "ResearchGeneral", "ResearchReddit", "ResearchSubstack",
        "ResearchPubMed", "ResearchArxiv", "ResearchSEC",
    }

    for i, tc in enumerate(tool_calls):
        name = tc.get("name", "unknown")
        args = tc.get("args", {})

        # Choose connector based on position
        connector = "└─" if i == len(tool_calls) - 1 else "├─"

        # Format based on tool type
        if name in research_tools:
            topic = _truncate(args.get("research_topic", ""), 60)
            if args.get("discovery"):
                print(f"      {connector} {Colors.YELLOW}{Colors.BOLD}{name} [DISCOVERY MODE]:{Colors.RESET} \"{topic}\"")
            else:
                print(f"      {connector} {Colors.CYAN}{name}:{Colors.RESET} \"{topic}\"")
        elif name == "refine_draft_report":
            print(f"      {connector} {Colors.GREEN}refine_draft_report{Colors.RESET}")
        elif name == "think_tool":
            reflection = args.get("reflection", "")
            print(f"      {connector} {Colors.GRAY}think_tool:{Colors.RESET} \"{reflection}\"")
        elif name == "ResearchComplete":
            print(f"      {connector} {Colors.GREEN}ResearchComplete{Colors.RESET}")
        else:
            print(f"      {connector} {name}")


def log_supervisor_end() -> None:
    """Log the end of supervisor processing (before going to next iteration or finishing)."""
    pass  # Currently no-op, but can be used for summary


# ===== SUB-AGENT LOGGING =====

def log_sub_agent_start(topic: str) -> int:
    """Log the start of a sub-agent research task. Returns agent ID for tracking."""
    observer = get_runtime().observer
    agent_id = observer.next_subagent_id()
    observer.mark_subagent_start(agent_id)
    if not _console():
        return agent_id

    print()
    print(f"   ┌{'─' * 65}")
    truncated_topic = _truncate(topic, 55)
    print(f"   │ {EMOJI_RESEARCH} {Colors.BOLD}SUB-AGENT #{agent_id}:{Colors.RESET} \"{truncated_topic}\"")

    return agent_id


def log_sub_agent_tool_call(agent_id: int, tool_name: str, args: Dict[str, Any]) -> None:
    """Log a tool call within a sub-agent."""
    if not _console():
        return
    if tool_name in {"tavily_search", "exa_deep_search"}:
        query = _truncate(args.get("query", ""), 50)
        print(f"   │    {EMOJI_SEARCH} {Colors.CYAN}{tool_name}:{Colors.RESET} \"{query}\"")
    elif tool_name == "think_tool":
        reflection = args.get("reflection", "")
        print(f"   │    {EMOJI_THINK} {Colors.GRAY}think_tool:{Colors.RESET} \"{reflection}\"")
    else:
        print(f"   │    {tool_name}: {args}")


def log_sub_agent_complete(agent_id: int, search_count: int = 0) -> None:
    """Log completion of a sub-agent research task."""
    elapsed = get_runtime().observer.mark_subagent_end(agent_id)
    if not _console():
        return
    print(f"   │    {EMOJI_COMPLETE} {Colors.GREEN}Research complete ({search_count} searches, {elapsed:.1f}s){Colors.RESET}")
    print(f"   └{'─' * 65}")


# ===== DISCOVERY AGENT LOGGING =====

def log_discovery_start(brief: str) -> int:
    """Log the start of a discovery-mode task. Returns agent ID."""
    observer = get_runtime().observer
    agent_id = observer.next_subagent_id()
    observer.mark_subagent_start(agent_id)
    if not _console():
        return agent_id

    print()
    print(f"   ┌{'─' * 65}")
    truncated = _truncate(brief, 55)
    print(f"   │ {EMOJI_DISCOVERY} {Colors.YELLOW}{Colors.BOLD}DISCOVERY MODE #{agent_id}:{Colors.RESET} \"{truncated}\"")

    return agent_id


def log_discovery_complete(agent_id: int, search_count: int = 0) -> None:
    """Log completion of a discovery-mode task."""
    elapsed = get_runtime().observer.mark_subagent_end(agent_id)
    if not _console():
        return
    print(f"   │    {EMOJI_COMPLETE} {Colors.YELLOW}Discovery complete ({search_count} searches, {elapsed:.1f}s){Colors.RESET}")
    print(f"   └{'─' * 65}")


# ===== UTILITY LOGGING =====

def log_refine_start() -> None:
    """Log the start of draft report refinement."""
    if _console():
        print(f"   {EMOJI_REFINE} {Colors.GREEN}Refining draft report with new findings...{Colors.RESET}")


def log_refine_complete() -> None:
    """Log completion of draft report refinement."""
    if _console():
        print(f"   {EMOJI_COMPLETE} {Colors.GREEN}Draft report refined{Colors.RESET}")


def log_research_complete() -> None:
    """Log that all research is complete."""
    if not _console():
        return
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{EMOJI_COMPLETE} RESEARCH COMPLETE - Generating final report...{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'═' * 70}{Colors.RESET}")
    print()


def log_error(message: str) -> None:
    """Log an error."""
    if _console():
        print(f"   {EMOJI_ERROR} {Colors.RED}ERROR: {message}{Colors.RESET}")


# ===== RESET STATE =====

def reset() -> None:
    """Reset per-run observer state (sub-agent timers/counters)."""
    get_runtime().observer.reset_console()
