"""Unit + integration tests for the Jev intent router in src/agent4.py.

The TypeSafe client is stubbed, so no TYPESAFE_API_KEY is needed.
Run:  env/bin/python test_agent4.py
"""

import asyncio
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import agent4
from langchain.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def choice(value, confidence):
    return types.SimpleNamespace(choice=value, confidence=confidence, probabilities={})


def noul(value):
    return types.SimpleNamespace(noul=value)


def base_answers(**over):
    """Default is a 'complex planning' answer (=> no fast path). Override per scenario."""
    a = {
        "route": choice("complex_planning", 0.9),
        "tool": choice("other", 0.9),
        "led_color": choice("white", 1.0),
        "led_stated": noul(0.1),
        "stop_requested": noul(0.05),
        "motion": choice("TurnLeft40", 1.0),
        "motion_stated": noul(0.0),
    }
    a.update(over)
    return a


class FakeJev:
    """Maps a message substring to a canned set of TypeSafe answers."""

    def __init__(self, cases):
        self.cases = cases

    def system_one(self, state, questions):
        msg = state["msg"]
        for key, ans in self.cases.items():
            if key in msg:
                return types.SimpleNamespace(answers=ans)
        raise AssertionError(f"FakeJev has no canned answer for {msg!r}")


# ---------------------------------------------------------------------------
# 1. Unit tests: decide_fast (pure logic, no MCP, no network)
# ---------------------------------------------------------------------------

TOOL_NAMES = {
    "get_robot_status",
    "get_robot_position",
    "get_robot_capabilities",
    "check_webots_connection",
    "get_visual_perception",
    "set_led_color",
    "play_motion",
    "set_head_position",
    "set_arm_position",
    "reset_robot_pose",
    "list_motions",
    "list_robots",
}
ROBOT = "Jame"


def _single(a):
    return agent4.decide_fast(a, tool_names=TOOL_NAMES, robots=[ROBOT], default_robot=ROBOT)


def test_emergency_without_stop_tool_falls_back():
    a = base_answers(route=choice("emergency_stop", 0.97), stop_requested=noul(0.99))
    assert _single(a) is None


def test_emergency_with_stop_tool():
    a = base_answers(route=choice("emergency_stop", 0.97), stop_requested=noul(0.99))
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES | {"stop_motion"}, robots=[ROBOT], default_robot=ROBOT
    )
    assert f["tool"] == "stop_motion"
    assert f["args"] == {"robot_name": ROBOT}
    assert f["kind"] == "emergency"


def test_emergency_low_confidence_falls_back():
    a = base_answers(route=choice("emergency_stop", 0.8), stop_requested=noul(0.99))
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES | {"stop_motion"}, robots=[ROBOT], default_robot=ROBOT
    )
    assert f is None


def test_led_on_defaults_white():
    a = base_answers(
        route=choice("simple_command", 0.95),
        tool=choice("set_led_color", 1.0),
        led_stated=noul(0.9),
        led_color=choice("white", 0.9),
    )
    f = _single(a)
    assert f["tool"] == "set_led_color"
    assert f["args"]["color"] == "white"
    assert f["kind"] == "action"


def test_led_red():
    a = base_answers(
        route=choice("simple_command", 0.95),
        tool=choice("set_led_color", 0.99),
        led_stated=noul(0.95),
        led_color=choice("red", 0.98),
    )
    assert _single(a)["args"]["color"] == "red"


def test_led_low_stated_falls_back():
    a = base_answers(
        route=choice("simple_command", 0.95),
        tool=choice("set_led_color", 1.0),
        led_stated=noul(0.2),
    )
    assert _single(a) is None


def test_status_query():
    a = base_answers(route=choice("status_query", 0.9), tool=choice("get_robot_status", 0.95))
    f = _single(a)
    assert f["tool"] == "get_robot_status"
    assert f["kind"] == "status"


def test_play_motion():
    a = base_answers(
        route=choice("simple_command", 0.9),
        tool=choice("play_motion", 0.9),
        motion_stated=noul(0.9),
        motion=choice("TaiChi", 0.85),
    )
    f = _single(a)
    assert f["tool"] == "play_motion"
    assert f["args"]["motion_name"] == "TaiChi"


def test_float_tool_goes_to_llm():
    a = base_answers(route=choice("simple_command", 0.9), tool=choice("set_head_position", 0.8))
    assert _single(a) is None


def test_low_tool_confidence_goes_to_llm():
    a = base_answers(route=choice("simple_command", 0.9), tool=choice("get_robot_status", 0.4))
    assert _single(a) is None


def test_complex_planning_goes_to_llm():
    assert _single(base_answers()) is None


def test_no_robots_disables_fast_path():
    a = base_answers(route=choice("status_query", 0.9), tool=choice("get_robot_status", 0.95))
    assert agent4.decide_fast(a, tool_names=TOOL_NAMES, robots=[], default_robot=None) is None


def test_multi_robot_named_robot_selected():
    a = base_answers(
        route=choice("simple_command", 0.95),
        tool=choice("set_led_color", 1.0),
        led_stated=noul(0.9),
        led_color=choice("red", 0.95),
        robot=choice("Awn", 0.95),
        robot_stated=noul(0.9),
    )
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES, robots=["Awn", "Jame"], default_robot="Jame"
    )
    assert f["args"]["robot_name"] == "Awn"


def test_multi_robot_default_when_unnamed():
    a = base_answers(
        route=choice("simple_command", 0.95),
        tool=choice("set_led_color", 1.0),
        led_stated=noul(0.9),
        led_color=choice("red", 0.95),
        robot=choice("Jame", 0.5),
        robot_stated=noul(0.2),
    )
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES, robots=["Awn", "Jame"], default_robot="Jame"
    )
    assert f["args"]["robot_name"] == "Jame"


def test_list_motions_is_remembered():
    """Ask 'what motions can you do?' -> answered from the cached list, not re-fetched."""
    a = base_answers(
        route=choice("simple_command", 0.95), tool=choice("list_motions", 0.99)
    )
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES, robots=[ROBOT], default_robot=ROBOT,
        motions=["TaiChi", "Wave"],
    )
    assert f is not None
    assert f["kind"] == "remembered"
    assert f["args"] == {}
    assert "TaiChi" in f["reply"] and "Wave" in f["reply"]


def test_list_robots_is_remembered():
    a = base_answers(
        route=choice("status_query", 0.9), tool=choice("list_robots", 0.99)
    )
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES, robots=["Awn", ROBOT], default_robot=ROBOT,
        motions=["TaiChi"],
    )
    assert f is not None
    assert f["kind"] == "remembered"
    assert "Awn" in f["reply"] and ROBOT in f["reply"]


def test_list_motions_unknown_falls_back_to_memory_reply():
    """No motions detected at build time -> honest reply, still no tool round trip."""
    a = base_answers(
        route=choice("simple_command", 0.95), tool=choice("list_motions", 0.99)
    )
    f = agent4.decide_fast(
        a, tool_names=TOOL_NAMES, robots=[ROBOT], default_robot=ROBOT, motions=[]
    )
    assert f["kind"] == "remembered"
    assert "No motions" in f["reply"]


def test_list_motions_is_remembered():
    # "What motions can you do?" -> answered from the cached list, not the tool.
    a = base_answers(
        route=choice("simple_command", 0.95), tool=choice("list_motions", 0.99)
    )
    f = agent4.decide_fast(
        a,
        tool_names=TOOL_NAMES,
        robots=[ROBOT],
        default_robot=ROBOT,
        motions=["TaiChi", "Wave"],
    )
    assert f is not None
    assert f["kind"] == "remembered"
    assert f["args"] == {}
    assert "TaiChi" in f["reply"] and "Wave" in f["reply"]


def test_list_robots_is_remembered():
    a = base_answers(
        route=choice("status_query", 0.9), tool=choice("list_robots", 0.99)
    )
    f = agent4.decide_fast(
        a,
        tool_names=TOOL_NAMES,
        robots=["Awn", ROBOT],
        default_robot=ROBOT,
        motions=["TaiChi"],
    )
    assert f is not None
    assert f["kind"] == "remembered"
    assert "Awn" in f["reply"] and ROBOT in f["reply"]


# ---------------------------------------------------------------------------
# 2. Integration: full graph, stubbed Jev, real Webots MCP server
# ---------------------------------------------------------------------------

DATA_DIR = Path(agent4.MCP_SERVER_PATH).parent / "data"


async def integration():
    fake = FakeJev(
        {
            "lights": base_answers(
                route=choice("simple_command", 0.95),
                tool=choice("set_led_color", 1.0),
                led_stated=noul(0.9),
                led_color=choice("white", 0.9),
                robot=choice("Jame", 1.0),
                robot_stated=noul(0.05),  # not explicitly named -> default robot
            ),
            "STOP": base_answers(
                route=choice("emergency_stop", 0.97),
                stop_requested=noul(0.99),
                robot=choice("Jame", 1.0),
                robot_stated=noul(0.05),
            ),
            "battery": base_answers(
                route=choice("status_query", 0.9),
                tool=choice("get_robot_status", 0.95),
                robot=choice("Jame", 1.0),
                robot_stated=noul(0.05),
            ),
            "motions": base_answers(
                route=choice("simple_command", 0.95),
                tool=choice("list_motions", 1.0),
            ),
            "STOP": base_answers(
                route=choice("emergency_stop", 0.97),
                stop_requested=noul(0.99),
                robot=choice("Jame", 1.0),
                robot_stated=noul(0.05),
            ),
        }
    )
    # jev_client() builds a fresh client per call, so stub the factory itself.
    agent4.jev_client = lambda: fake

    client = MultiServerMCPClient(
        {
            "webots_mcp": {
                "command": "/Users/jamshid/.pyenv/shims/python",
                "args": [agent4.MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    )
    agent4._test_mcp_client = client

    try:
        graph = await agent4.make_graph()
        nodes = graph.get_graph().nodes
        assert {"jev_gate", "fast_exec", "llm_call", "tools"} <= set(nodes), nodes

        # --- Fast path: "Turn on the lights" (async: MCP tools are async-only) ---
        res = await graph.ainvoke({"messages": [HumanMessage(content="Turn on the lights")]})
        assert res.get("llm_calls", 0) == 0, "fast path must not call the LLM"
        led_call = [m for m in res["messages"] if isinstance(m, AIMessage) and m.tool_calls][-1]
        assert led_call.tool_calls[0]["name"] == "set_led_color", res["messages"]
        assert "LEDs set to white" in res["messages"][-1].content
        robot_led = led_call.tool_calls[0]["args"]["robot_name"]

        # The command actually landed in the MCP protocol's command file.
        cmd_file = DATA_DIR / robot_led / "commands.json"
        assert cmd_file.exists(), f"no command file written for {robot_led}"
        assert json.loads(cmd_file.read_text())["action"] == "set_leds"

        # --- Status path: "what is the battery level?" ---
        res = await graph.ainvoke(
            {"messages": [HumanMessage(content="what is the battery level?")]}
        )
        assert res.get("llm_calls", 0) == 0
        assert res["messages"][-1].type == "ai"

        # --- Remembered list: "what motions can you do?" ---
        # The agent must answer from the list cached at graph build: no LLM
        # call and no tool round trip (no AIMessage carries a tool call).
        res = await graph.ainvoke(
            {"messages": [HumanMessage(content="what motions can you do?")]}
        )
        assert res.get("llm_calls", 0) == 0
        list_tool_calls = [
            m
            for m in res["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        assert not list_tool_calls, (
            "list_motions should be answered from memory, not re-fetched"
        )
        assert "motion" in res["messages"][-1].content.lower()

        # --- Emergency path: "STOP!" (fast path -> stop_motion tool) ---
        res = await graph.ainvoke({"messages": [HumanMessage(content="STOP!")]})
        assert res.get("llm_calls", 0) == 0, "emergency stop must not call the LLM"
        stop_call = [m for m in res["messages"] if isinstance(m, AIMessage) and m.tool_calls][-1]
        assert stop_call.tool_calls[0]["name"] == "stop_motion", res["messages"]
        assert "stop flag set" in res["messages"][-1].content
        robot_stop = stop_call.tool_calls[0]["args"]["robot_name"]
        cmd_file = DATA_DIR / robot_stop / "commands.json"
        assert json.loads(cmd_file.read_text())["action"] == "stop_motion"

        # The MCP server now really blocks motion (safety net against any
        # stale in-flight run), and resume_motion unblocks it again.
        tools_by_name = {t.name: t for t in await client.get_tools()}
        blocked = await tools_by_name["play_motion"].ainvoke(
            {"robot_name": robot_stop, "motion_name": "TaiChi"}
        )
        blocked_text = (
            blocked
            if isinstance(blocked, str)
            else (
                "\n".join(
                    b["text"]
                    for b in blocked
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                )
                if isinstance(blocked, list)
                else json.dumps(blocked, ensure_ascii=False)
            )
        )
        assert "⛔" in blocked_text, blocked_text
        await tools_by_name["resume_motion"].ainvoke({"robot_name": robot_stop})

    finally:
        close = getattr(client, "close", None)
        if close:
            await close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"{len(tests)} unit tests passed")
    print("running integration (spawns the Webots MCP server)...")
    asyncio.run(integration())
    print("PASS integration: fast path executed with 0 LLM calls")


if __name__ == "__main__":
    main()