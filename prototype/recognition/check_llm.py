"""Live model/MCP smoke test. Uses the configured agent5 model/API key."""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['LANGSMITH_TRACING'] = 'false'
from langchain.messages import HumanMessage
from src.agent5 import make_graph, mcp_client

async def main():
    graph = await make_graph()
    result = await graph.ainvoke({'messages':[HumanMessage(content='What objects can Jame see right now? Use built-in recognition.')]}, {'recursion_limit':12})
    calls = [call['name'] for message in result['messages'] if message.type == 'ai' for call in message.tool_calls]
    assert 'get_object_recognition' in calls, calls
    answer = result['messages'][-1].text
    print('Tool calls:', calls)
    print('Answer:', answer)
    assert 'red' in str(answer).lower() and 'blue' in str(answer).lower(), answer
    print('PASS live LLM -> MCP -> Webots recognition -> LLM answer')
    if '--search' in sys.argv:
        try:
            search = await graph.ainvoke({'messages': [HumanMessage(content='Find the green box for Jame by looking around with the head. Report only what recognition confirms.')]}, {'recursion_limit': 30})
            answer = search['messages'][-1].text
            print('Search answer:', answer)
            observations = [m.content for m in search['messages'] if m.type == 'tool' and m.name == 'get_object_recognition']
            assert any('green box' in frame for frame in observations), observations
            print('PASS live agent head search found green box')
        finally:
            tools = {t.name: t for t in await mcp_client().get_tools()}
            await tools['set_head_position'].ainvoke({'robot_name':'Jame', 'yaw':0, 'pitch':0})

asyncio.run(main())
