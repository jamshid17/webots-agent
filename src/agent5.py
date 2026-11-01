"""Perception-first Webots agent. Run with LangGraph or --check (no LLM key).

The LLM plans; MCP/controller provide fresh observations and execute skills.
No TypeSafe dependency: visual tasks always go through the reasoning loop.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(os.getenv('MCP_SERVER_PATH') or ROOT.parent/'webots-mcp/webots_mcp_server.py')
PROMPT = """You control Webots robots through MCP. Execute requests and report observed results.
Call list_robots to discover robot names; never invent a name or use placeholder coordinates.
For 'what do you see', object finding, or spatial questions, use get_object_recognition.
It returns Webots simulated recognition ground truth, not learned image recognition.
Use only returned labels/IDs. Confidence null means no learned confidence is provided.
Positions are in the camera frame (x forward, y left, z up), meters; bearing positive is left.
Observations describe the current view only. Empty detections do not prove free space.
On unavailable/stale observations explain the error; never invent observations.
For a requested search: observe, then if necessary scan head yaw -0.8, 0, +0.8
with pitch 0, observing after EACH move; stop when the target is found. At most
three head moves per request. Detection after a head move may need another read
while the head settles. Use image position/bearing to describe left/right.
Execute tools sequentially because the controller has a single command file.
Do not treat a queued command as confirmed physical completion. Verify where possible.
Re-observe after any motion before deciding what to do next. Never calculate a long
open-loop walking sequence from one observation. This prototype has no closed-loop
navigation/grasp skill: explain that limitation for approach/pick-up requests; do not
substitute unverified motion animations for obstacle-aware navigation or grasping.
Use basic motion/LED/head/arm tools when explicitly requested. Honor stop flags;
resume only on an explicit request. A stop request must call stop_motion immediately.
Do not call reset_robot_pose to teleport; it only commands the controller's initial joint pose.
Camera file paths from get_visual_perception are not image content; use recognition for objects.
"""


def mcp_client():
    return MultiServerMCPClient({'webots_mcp': {
        'command': os.getenv('MCP_PYTHON') or sys.executable,
        'args': [str(SERVER)], 'transport': 'stdio',
    }})


def decode_result(raw):
    """Normalize MCP adapter text blocks, structured results, or JSON strings."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    if isinstance(raw, list) and raw and all(isinstance(b, dict) and b.get('type') == 'text' for b in raw):
        values = [decode_result(b['text']) for b in raw]
        return values[0] if len(values) == 1 else values
    return raw


async def make_graph(config=None):
    """LangGraph API entry point; only the runtime config is a parameter."""
    return await _build_graph()


async def _build_graph(*, client=None, model=None):
    """Build the graph with optional dependency injection for offline tests."""
    client = client or mcp_client()
    tools = await client.get_tools()
    by_name = {t.name: t for t in tools}
    if 'get_object_recognition' not in by_name:
        raise RuntimeError('Install prototype/recognition/install.py: MCP server lacks get_object_recognition')
    if model is None:
        provider = os.getenv('AGENT5_PROVIDER', 'google')
        if provider == 'google':
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(model=os.getenv('AGENT5_MODEL', 'gemini-3.7-flash'),
                                            temperature=1.0, timeout=60, max_retries=2)
        elif provider == 'openai':
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model=os.getenv('AGENT5_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
                               timeout=60, max_retries=2)
        else:
            raise ValueError('AGENT5_PROVIDER must be google or openai')
    bound = model.bind_tools(tools)

    async def reason(state):
        return {'messages': [await bound.ainvoke([SystemMessage(content=PROMPT)] + state['messages'])]}

    async def execute(state):
        results = []
        calls = state['messages'][-1].tool_calls
        # If a model batches stop with movement, stop wins and other calls are cancelled.
        stopping = any(call['name'] == 'stop_motion' for call in calls)
        for call in calls:
            try:
                if stopping and call['name'] != 'stop_motion':
                    result = {'error': 'Cancelled because this batch contains a stop request'}
                else:
                    result = await by_name[call['name']].ainvoke(call['args'])
            except Exception as exc:
                result = {'error': str(exc)}
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            results.append(ToolMessage(content=content, tool_call_id=call['id'], name=call['name']))
        return {'messages': results}

    builder = StateGraph(MessagesState)
    builder.add_node('reason', reason)
    builder.add_node('tools', execute)
    builder.add_edge(START, 'reason')
    builder.add_conditional_edges('reason', lambda s: 'tools' if s['messages'][-1].tool_calls else END)
    builder.add_edge('tools', 'reason')
    return builder.compile()


async def check(robot_name=None):
    tools = {tool.name: tool for tool in await mcp_client().get_tools()}
    if 'get_object_recognition' not in tools:
        raise RuntimeError('Recognition MCP tool missing. Install prototype first.')
    robots = decode_result(await tools['list_robots'].ainvoke({}))
    if isinstance(robots, dict) and 'name' in robots:
        robots = [robots]
    print('Robots:', json.dumps(robots))
    if not robot_name:
        names = [r['name'] for r in robots if isinstance(r, dict) and 'name' in r] if isinstance(robots, list) else []
        if len(names) != 1:
            raise RuntimeError('Pass --robot NAME when zero or multiple robots are listed')
        robot_name = names[0]
    observation = decode_result(await tools['get_object_recognition'].ainvoke({'robot_name': robot_name}))
    print(json.dumps(observation, indent=2))
    if not isinstance(observation, dict) or not observation.get('available'):
        raise RuntimeError('No live recognition frame; open agent5-recognition.wbt and run simulation')
    return observation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check live MCP recognition without an LLM call or movement')
    parser.add_argument('--robot')
    parser.add_argument('--message', help='Run a single request through agent5')
    args = parser.parse_args()
    if args.check:
        asyncio.run(check(args.robot))
    elif args.message:
        async def run_message():
            graph = await make_graph()
            result = await graph.ainvoke({'messages': [HumanMessage(content=args.message)]}, {'recursion_limit': 30})
            print(result['messages'][-1].text)
        asyncio.run(run_message())
    else:
        parser.error('Use --check, --message TEXT, or launch graph agent5 with langgraph dev')
