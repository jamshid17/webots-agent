"""LangGraph agent with a Jev (TypeSafe) front-door intent router.

Pipeline:

    START -> jev_gate --high-confidence fast route--> fast_exec -> END
                      \--everything else-----------> llm_call <--> tools -> END

- jev_gate: one TypeSafe call per user message. All questions are evaluated in
  parallel (speculative fan-out), so asking "everything" costs a single round trip.
- fast_exec: calls the chosen MCP tool directly. No LLM on this path (~150-250ms
  end-to-end instead of 1.5-3s for the LLM loop).
- llm_call/tools: the original agent loop, unchanged from agent3.
- Memory: robots and motions are resolved once at graph build and injected into
  the LLM's system prompt. list_robots / list_motions questions are answered
  from that cache (kind="remembered"), so the agent remembers instead of going
  back to the tools after the first message.

Design rules (from the TypeSafe docs):
- Jev only ever sees closed sets (tool names, LED colors, motion names). Whatever
  fast_exec receives is a value the tool actually accepts.
- No numeric questions: Jev is not a calculator, so yaw/pitch floats and HEX
  colors fall through to the LLM path.
- Confidence gates scale with risk: emergency stop 0.9, simple/status 0.6.
- Any Jev error, low confidence, or missing piece falls through to the LLM.
  The fast path must never block or break the user.

Note on "STOP!" while the agent is running: this graph handles *new* messages.
Preempting an in-flight LLM run needs the client-side cancel (useStream) plus a
stop flag in the Webots MCP server — see the repo notes / agent4 test file.
"""

import json
import logging
import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typesafe_sdk import Choice, Noul, TypeSafeClient

try:
    # LangGraph imports this file as the src.agent4 module (root on sys.path).
    from src.prompts import build_system_prompt
except ImportError:
    # Tests import a bare "agent4" with src/ itself on sys.path.
    from prompts import build_system_prompt

load_dotenv()

logger = logging.getLogger("agent4")

# ---------------------------------------------------------------------------
# Jev configuration
# ---------------------------------------------------------------------------

# Pin a specific model version. The "jev-latest" alias moves forward when a new
# release ships and would silently re-tune your confidence thresholds.
JEV_MODEL = "jev-1.13.0"

# Confidence gates: the answer tells you WHAT; confidence tells you WHETHER TO ACT.
CONF_EMERGENCY = 0.9  # high-stakes action: stopping the robot
CONF_FAST = 0.6  # low-stakes actions: status queries, LED color
CONF_ARG = 0.6  # per-argument gate: "did the user state a color / motion?"

# Closed sets — must match what the Webots MCP server accepts.
LED_COLORS = ["red", "green", "blue", "white", "off"]

# Tools fast_exec may run without an LLM. Tools with float args (set_head_position,
# set_arm_position) are intentionally excluded: Jev is bad at numbers.
READ_ONLY_TOOLS = {
    "get_robot_status",
    "get_robot_position",
    "get_robot_capabilities",
    "check_webots_connection",
    "get_visual_perception",
}

# The Webots MCP server: single source of truth for the path. Its parent
# directory also holds data/ with per-robot status files, used to pick the
# default robot when several are connected.
MCP_SERVER_PATH = os.environ.get("MCP_SERVER_PATH", None)



def jev_client() -> TypeSafeClient | None:
    """Lazy singleton. Without a key the fast path degrades to the LLM path."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        logger.warning("TYPESAFE_API_KEY not set — Jev fast path disabled.")
        return None
    client = TypeSafeClient(model=JEV_MODEL)
    return client


def _most_recently_active(robot_names: list[str]) -> str | None:
    """Default robot: the one whose status file was updated most recently."""
    if not robot_names:
        return None
    if len(robot_names) == 1:
        return robot_names[0]
    data_dir = Path(MCP_SERVER_PATH).parent / "data"
    best, best_ts = None, -1.0
    for name in robot_names:
        status_file = data_dir / name / "status.json"
        ts = status_file.stat().st_mtime if status_file.exists() else 0.0
        if ts > best_ts:
            best, best_ts = name, ts
    return best or robot_names[0]


# ---------------------------------------------------------------------------
# Question battery + decision logic (pure, unit-testable)
# ---------------------------------------------------------------------------


def build_questions(
    tool_descriptions: dict[str, str],
    motions: list[str],
    robots: list[str] | None = None,
) -> dict:
    """Build one Jev request. All questions are asked together and run in parallel."""
    tool_criteria = dict(tool_descriptions)
    tool_criteria["other"] = (
        "No single tool fits; the request needs planning or a conversation"
    )

    questions = {
        "route": Choice(
            instructions="What kind of request is `msg`?",
            criteria={
                "emergency_stop": "The user wants the robot to stop right now (e.g. 'stop', 'halt', 'emergency', 'kill it')",
                "simple_command": "A single robot action needing no planning (e.g. 'turn on the lights', 'wave')",
                "status_query": "The user is asking about the robot's current state (e.g. 'what is the battery level?', 'where are you?', 'what do you see?')",
                "complex_planning": "A multi-step, ambiguous, or conversational request that needs reasoning",
            },
        ),
        "tool": Choice(
            instructions="Which single MCP tool would fulfill `msg`?",
            criteria=tool_criteria,
        ),
        "led_color": Choice(
            instructions="Which LED color does the user want?",
            criteria={
                "red": "A red light",
                "green": "A green light",
                "blue": "A blue light",
                "white": "A white light — also: simply turning the lights on with no color named",
                "off": "No light — turning the lights off",
            },
        ),
        "led_stated": Noul(
            instructions="Does `msg` name a specific LED color, or ask to turn the lights on or off?"
        ),
        "stop_requested": Noul(
            instructions="Does `msg` ask the robot to stop, halt, or abort?"
        ),
    }
    if motions:
        questions["motion"] = Choice(
            instructions="Which motion would the robot perform?",
            criteria={m: f"The motion called '{m}'" for m in motions},
        )
        questions["motion_stated"] = Noul(
            instructions="Does `msg` ask the robot to perform a motion, dance, or animation?"
        )
    if robots and len(robots) > 1:
        questions["robot"] = Choice(
            instructions="Which robot is the user talking about?",
            criteria={r: f"The robot called '{r}'" for r in robots},
        )
        questions["robot_stated"] = Noul(
            instructions="Does `msg` name a specific robot?"
        )
    return questions


def decide_fast(
    answers,
    *,
    tool_names: set[str],
    robots: list[str],
    default_robot: str | None,
    motions: list[str] | None = None,
) -> dict | None:
    """Turn Jev's answers into a fast action, or None (= use the LLM path).

    `answers` is a mapping of question id -> answer object (the real SDK response
    answers or a test double with the same fields).
    """
    route = answers["route"]
    tool = answers["tool"]

    # Which robot? One robot: no question needed. Several: use Jev's answer when
    # the user actually named one, else the default (most recently active).
    if not robots:
        return None
    if len(robots) == 1:
        robot_name = robots[0]
    else:
        robot_name = default_robot
        if (
            "robot" in answers
            and answers["robot_stated"].noul >= CONF_ARG
            and answers["robot"].confidence >= CONF_ARG
            and answers["robot"].choice in robots
        ):
            robot_name = answers["robot"].choice

    # --- Emergency stop: highest priority, highest bar ------------------------
    if (
        route.choice == "emergency_stop"
        and route.confidence >= CONF_EMERGENCY
        and answers["stop_requested"].noul >= 0.7
    ):
        if "stop_motion" in tool_names and robot_name:
            return {
                "tool": "stop_motion",
                "args": {"robot_name": robot_name},
                "kind": "emergency",
                "reply": f"⏹️ {robot_name}: stop flag set — no motion will run until you say resume.",
            }
        logger.warning(
            "Emergency request but the MCP server has no stop_motion tool — using the LLM path."
        )
        return None

    # --- Simple commands / status queries --------------------------------------
    if (
        route.choice not in ("simple_command", "status_query")
        or route.confidence < CONF_FAST
    ):
        return None
    if robot_name is None:
        return None
    name = tool.choice
    if name == "other" or tool.confidence < CONF_FAST:
        return None

    if name in READ_ONLY_TOOLS:
        return {"tool": name, "args": {"robot_name": robot_name}, "kind": "status"}

    if name in ("list_motions", "list_robots"):
        # Static knowledge, resolved once at graph build — answer from memory
        # instead of re-calling the MCP tool (kind="remembered" skips the
        # round trip entirely in fast_exec).
        items = motions or [] if name == "list_motions" else robots
        label = "motions" if name == "list_motions" else "robots"
        reply = (
            f"Available {label}: " + ", ".join(items)
            if items
            else f"No {label} are known right now."
        )
        return {"tool": name, "args": {}, "kind": "remembered", "reply": reply}

    if name == "set_led_color" and answers["led_stated"].noul >= CONF_ARG:
        color = answers["led_color"].choice
        return {
            "tool": "set_led_color",
            "args": {"robot_name": robot_name, "color": color},
            "kind": "action",
            "reply": f"✅ {robot_name}: LEDs set to {color}.",
        }

    if (
        name == "play_motion"
        and "motion" in answers
        and answers["motion_stated"].noul >= CONF_ARG
        and answers["motion"].confidence >= CONF_ARG
    ):
        motion = answers["motion"].choice
        return {
            "tool": "play_motion",
            "args": {"robot_name": robot_name, "motion_name": motion},
            "kind": "action",
            "reply": f"▶️ {robot_name} is playing '{motion}'.",
        }

    # Everything else (floats, open text, low confidence) goes to the LLM.
    return None


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def _try_json(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _names_from(raw) -> list[str]:
    """Extract robot/motion names from an MCP tool result.

    Depending on the adapter version the result arrives as a list of MCP text
    content blocks, a plain list of dicts, or a JSON string.
    """
    if isinstance(raw, str):
        raw = _try_json(raw)
    items = raw if isinstance(raw, list) else [raw] if raw is not None else []
    names = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "text":
            item = _try_json(item.get("text", ""))
        if isinstance(item, dict) and item.get("name"):
            names.append(item["name"])
    return names


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    fast: dict | None


# Test-only MCP client injection (never passed as a factory parameter: the
# platform treats any single factory parameter as the graph "config" dict).
_test_mcp_client: MultiServerMCPClient | None = None


async def make_graph(config: dict | None = None):
    """Build the agent. The LangGraph platform calls make_graph(config)."""
    client = _test_mcp_client or MultiServerMCPClient(
        {
            "webots_mcp": {
                "command": "/Users/jamshid/.pyenv/shims/python",
                "args": [MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    # 1. MCP tools
    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}

    # 2. Deterministic facts, resolved in code — not by Jev.
    # MCP tools are async-only, so use ainvoke (make_graph is async).
    try:
        robots_raw = await tools_by_name["list_robots"].ainvoke({}) or []
    except Exception:
        robots_raw = []
    robot_names = _names_from(robots_raw)
    default_robot = _most_recently_active(robot_names)
    logger.info("Jev gate ready: robots=%s default=%s", robot_names, default_robot)

    try:
        raw_motions = await tools_by_name["list_motions"].ainvoke({}) or []
    except Exception:
        raw_motions = []
    motions = _names_from(raw_motions)

    questions = build_questions(
        {t.name: (t.description or t.name).strip().split(".")[0] + "." for t in tools},
        motions,
        robot_names,
    )

    # --- System prompt: the LLM's "memory" of the environment -------------------
    # Wording lives in src/prompts.py so it can be tuned in one place.
    system_prompt = build_system_prompt(robot_names, default_robot, motions)

    # --- Node: the Jev gate -----------------------------------------------------

    def jev_gate(state: MessagesState) -> dict:

        logger.info("here jev gate")
        last = state["messages"][-1]
        logger.info(state["messages"])
        logger.info(last, type(last))
        # Messages arrive as dicts (platform input) or BaseMessage objects
        # (tests, later turns) — accept both shapes.
        last_type = last["type"] if isinstance(last, dict) else getattr(last, "type", None)
        if last_type != "human":
            logger.info("jev last message is not human message")
            return {"fast": None}
        msg_text = last["content"] if isinstance(last, dict) else last.content
        # if not isinstance(last, HumanMessage):
        #     logger.info("jev last message is not human message")
        #     return {"fast": None}
        jc = jev_client()
        logger.info("jev client%s", jc)
        if jc is None:
            logger.info("jev is not here")
            return {"fast": None}
        fast = None
        try:
            answers = jc.system_one(state={"msg": msg_text}, questions=questions).answers
            logger.info("answers%s", answers)
            fast = decide_fast(
                answers,
                tool_names=set(tools_by_name),
                robots=robot_names,
                default_robot=default_robot,
                motions=motions,
            )
            if fast:
                logger.info("Jev fast path -> %s %s", fast["tool"], fast["args"])
        except Exception as e:  # the fast path must never break the agent
            logger.warning("Jev gate failed (%s) — falling back to the LLM path.", e)
            fast = None
        logger.info("jev gate info here tretun fast")
        return {"fast": fast}

    # --- Node: fast executor — straight to the MCP tool, no LLM -----------------

    async def fast_exec(state: MessagesState) -> dict:
        # Async node on purpose: MCP tools are async-only, and ainvoke must run
        # on the same event loop the MCP session lives on.
        f = state["fast"]
        # Remembered lists (robots/motions) are answered from memory — no LLM,
        # no MCP round trip. The agent remembers instead of re-fetching.
        if f.get("kind") == "remembered":
            return {"messages": [AIMessage(content=f["reply"])]}
        call_id = f"fast-{uuid4().hex[:8]}"
        result = await tools_by_name[f["tool"]].ainvoke(f["args"])
        # MCP results may be a list of text content blocks; flatten to a string.
        if isinstance(result, list):
            texts = [
                b["text"]
                for b in result
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            result = (
                "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
            )
        elif not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        reply = result if f["kind"] == "status" else f["reply"]
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": f["tool"], "args": f["args"], "id": call_id}],
                ),
                ToolMessage(content=result, tool_call_id=call_id),
                AIMessage(content=reply),
            ]
        }

    # --- Original LLM loop (unchanged from agent3) ------------------------------
    # with one deliberate difference: the robot/motion lists resolved above are
    # baked into the system prompt, so the LLM remembers them instead of
    # calling list_robots / list_motions again on later messages.

    model = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature=1.0,  # Gemini 3.0+ defaults to 1.0
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    # model = ChatOpenAI(
    #     model="Qwen/Qwen3.8-27B-NVFP4",
    #     api_key=os.environ["QWEN_API_KEY"],
    #     base_url=os.environ["QWEN_BASE_URL"],
    #     temperature=0.7,
    # )
    model_with_tools = model.bind_tools(tools)

    def llm_call(state: MessagesState) -> dict:
        return {
            "messages": [
                model_with_tools.invoke(
                    [SystemMessage(content=system_prompt)] + state["messages"]
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    def route_after_gate(state: MessagesState) -> Literal["fast_exec", "llm_call"]:
        return "fast_exec" if state.get("fast") else "llm_call"

    # --- Wire it up -------------------------------------------------------------

    builder = StateGraph(MessagesState)
    builder.add_node("jev_gate", jev_gate)
    builder.add_node("fast_exec", fast_exec)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "jev_gate")
    builder.add_conditional_edges(
        "jev_gate", route_after_gate, ["fast_exec", "llm_call"]
    )
    builder.add_edge("fast_exec", END)
    builder.add_conditional_edges("llm_call", should_continue, ["tools", END])
    builder.add_edge("tools", "llm_call")

    return builder.compile()
