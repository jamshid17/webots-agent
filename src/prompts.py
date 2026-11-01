"""Prompts for the Webots LangGraph agent.

Kept in a separate file so the prompt wording can be tuned without touching
the graph wiring in agent4.py.
"""


def build_system_prompt(
    robot_names: list[str],
    default_robot: str | None,
    motions: list[str],
) -> str:
    """The LLM's system prompt, including its "memory" of the environment.

    Robots and motions are resolved once at graph build; embedding them here
    means the LLM answers list questions from memory instead of re-calling
    list_robots / list_motions on every turn.
    """
    robot_line = (
        f"- Available robots: {', '.join(robot_names)}"
        if robot_names
        else "- Available robots: none detected"
    )
    default_line = (
        f" (default/most recently active: {default_robot})" if default_robot else ""
    )
    motion_line = (
        f"- Available motions: {', '.join(motions)}"
        if motions
        else "- Available motions: none detected"
    )
    return (
        "You are a helpful assistant capable of controlling and "
        "interacting with the Webots simulation via MCP tools.\n\n"
        "Known environment (remembered — fetched once when this session started):\n"
        f"{robot_line}{default_line}\n"
        f"{motion_line}\n\n"
        "Rules:\n"
        "- Answer questions about available robots or motions from the lists above. "
        "Do not call list_robots or list_motions unless the user explicitly asks "
        "to refresh or re-scan the environment."
    )