"""Offline contract/graph tests; --live additionally tests the running Webots demo."""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from types import SimpleNamespace

os.environ['LANGSMITH_TRACING'] = 'false'
from langchain.messages import AIMessage, HumanMessage
from src import agent5
from prototype.recognition.recognition_bridge import active_robots, publish, read_observation, snapshot


class RecognitionTests(unittest.TestCase):
    def test_robot_discovery_excludes_old_missing_and_corrupt_status(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, status in {
                'Jame': {'timestamp': time.time(), 'webots_connected': True,
                         'robot_position': {'x': 1}},
                'Awn': {'timestamp': time.time()-60, 'webots_connected': True},
                'Disconnected': {'timestamp': time.time(), 'webots_connected': False},
            }.items():
                folder = Path(directory)/name
                folder.mkdir()
                publish(folder/'status.json', status)
            (Path(directory)/'Empty').mkdir()
            broken = Path(directory)/'Broken'
            broken.mkdir()
            (broken/'status.json').write_text('{')
            self.assertEqual(active_robots(directory), [{'name': 'Jame', 'position': {'x': 1}}])

    def test_serialize_real_api_shape(self):
        obj = SimpleNamespace(getId=lambda: 1, getModel=lambda: 'red box',
            getPosition=lambda: [1, 1, 0], getPositionOnImage=lambda: [40, 60],
            getSizeOnImage=lambda: [20, 30], getSize=lambda: [.2, .3])
        camera = SimpleNamespace(getName=lambda: 'RecognitionCamera', hasRecognition=lambda: True,
            getWidth=lambda: 320, getHeight=lambda: 240, getRecognitionObjects=lambda: [obj])
        result = snapshot(camera, 12)
        self.assertEqual(result['objects'][0]['label'], 'red box')
        self.assertAlmostEqual(result['objects'][0]['distance_m'], 2**.5)
        self.assertGreater(result['objects'][0]['bearing_rad'], 0)
        self.assertIsNone(result['objects'][0]['confidence'])
        json.dumps(result, allow_nan=False)

    def test_unavailable_is_not_empty_detection(self):
        self.assertFalse(snapshot(None, 0)['available'])

    def test_fresh_stale_missing_and_robot_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'Jame'; folder.mkdir()
            publish(folder/'recognition.json', {'available': True, 'timestamp': time.time(), 'objects': []})
            self.assertTrue(read_observation(directory, 'Jame')['available'])
            self.assertFalse(read_observation(directory, 'Other')['available'])
            self.assertFalse(read_observation(directory, '../Jame')['available'])
            publish(folder/'recognition.json', {'available': True, 'timestamp': time.time()-20, 'objects': []})
            self.assertIn('stale', read_observation(directory, 'Jame')['error'])

    def test_decode_adapter(self):
        self.assertEqual(agent5.decode_result([{'type':'text', 'text':'{"available":true}'}]), {'available':True})


class FakeTool:
    def __init__(self, name, log):
        self.name, self.log = name, log
    async def ainvoke(self, args):
        self.log.append(self.name)
        await asyncio.sleep(0)
        return {'available': True, 'objects': []}


class FakeModel:
    def __init__(self, calls):
        self.calls, self.count = calls, 0
    def bind_tools(self, tools):
        return self
    async def ainvoke(self, messages):
        self.count += 1
        if self.count == 1:
            return AIMessage(content='', tool_calls=self.calls)
        assert any(getattr(m, 'type', None) == 'tool' for m in messages)
        return AIMessage(content='Observed result')


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_platform_factory_signature_and_delegation(self):
        from langgraph_api._factory_utils import _classify_factory

        arguments = _classify_factory(agent5.make_graph)
        config = {'configurable': {}}
        self.assertEqual(arguments(config, None), {'config': config})
        graph = object()
        with patch.object(agent5, '_build_graph', new=AsyncMock(return_value=graph)) as build:
            self.assertIs(await agent5.make_graph(**arguments(config, None)), graph)
            build.assert_awaited_once_with()

    async def run_graph(self, names):
        log = []
        tools = [FakeTool(name, log) for name in {'get_object_recognition', *names}]
        class Client:
            async def get_tools(self): return tools
        calls = [{'name': name, 'args': {'robot_name': 'Jame'}, 'id': str(i)} for i, name in enumerate(names)]
        graph = await agent5._build_graph(client=Client(), model=FakeModel(calls))
        result = await graph.ainvoke({'messages':[HumanMessage(content='test')]})
        self.assertEqual(result['messages'][-1].content, 'Observed result')
        return log, result

    async def test_tool_calls_are_sequential_and_observations_return_to_model(self):
        log, _ = await self.run_graph(['set_head_position', 'get_object_recognition'])
        self.assertEqual(log, ['set_head_position', 'get_object_recognition'])

    async def test_stop_cancels_batched_movements(self):
        log, result = await self.run_graph(['play_motion', 'stop_motion', 'set_head_position'])
        self.assertEqual(log, ['stop_motion'])
        self.assertEqual(len([m for m in result['messages'] if m.type == 'tool']), 3)

    async def test_missing_recognition_tool_fails_early(self):
        class Client:
            async def get_tools(self): return []
        with self.assertRaisesRegex(RuntimeError, 'lacks'):
            await agent5._build_graph(client=Client(), model=FakeModel([]))


async def live():
    observation = await agent5.check('Jame')
    labels = {o['label'] for o in observation['objects']}
    assert {'red box', 'blue box'} <= labels, labels
    tools = {tool.name: tool for tool in await agent5.mcp_client().get_tools()}
    # A bounded, sensor-driven search. We never move the base or use Supervisor data.
    try:
        await tools['set_head_position'].ainvoke({'robot_name':'Jame', 'yaw':.8, 'pitch':0})
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            await asyncio.sleep(.25)
            view = agent5.decode_result(await tools['get_object_recognition'].ainvoke({'robot_name':'Jame'}))
            if any(o['label'] == 'green box' for o in view.get('objects', [])):
                print('PASS live head search: green box detected after turning left')
                break
        else:
            raise AssertionError('Green box was not detected after left head turn')
    finally:
        await tools['set_head_position'].ainvoke({'robot_name':'Jame', 'yaw':0, 'pitch':0})
    print('PASS live MCP -> Webots controller -> recognition -> MCP')


if __name__ == '__main__':
    if '--live' in sys.argv:
        asyncio.run(live())
    else:
        unittest.main()
