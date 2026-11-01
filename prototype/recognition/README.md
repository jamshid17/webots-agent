# Agent5 recognition prototype

Agent5 uses Webots' built-in camera recognition through MCP. The controller
publishes fresh observations continuously; the LLM plans and reads results.
This is simulator ground truth for tagged objects, not a trained vision model.

## Run

1. Open `/Users/jamshid/Documents/uni/project1/webots-mcp/worlds/agent5-recognition.wbt`
   in Webots and run the simulation. It has one NAO named `Jame`, two boxes in
   front, and a green box outside the initial camera view, to the left.
2. From this repository run `env/bin/langgraph dev` (or your existing launcher).
3. In the chat UI select graph/assistant ID **agent5**. You can also set its
   `assistantId` URL query parameter to `agent5`. Existing graph `agent` still
   points to agent4.
4. Try:
   - "What objects can Jame see?"
   - "Which visible object is closest to Jame?"
   - "Find the green box by looking around."
   - "Turn Jame's head back to the center and look again."
   - "Set Jame's LEDs to blue."

The same Google model as agent4 is the default. Optional environment settings:
`AGENT5_PROVIDER=google|openai`, `AGENT5_MODEL`, `MCP_SERVER_PATH`, `MCP_PYTHON`.
OpenAI mode needs `OPENAI_API_KEY`; Google mode uses `GOOGLE_API_KEY`.

## Verification

```sh
LANGSMITH_TRACING=false env/bin/python test_agent5.py
LANGSMITH_TRACING=false env/bin/python src/agent5.py --check --robot Jame
LANGSMITH_TRACING=false env/bin/python test_agent5.py --live
env/bin/python prototype/recognition/check_llm.py --search
LANGSMITH_TRACING=false env/bin/python src/agent5.py --message 'What can Jame see?'
```

`--check` does not use an LLM or move the robot. `--live` checks initial red/blue
detections, turns the head left, verifies green detection, and restores the head.
The model smoke test uses the configured API key and may incur API charges.

## Components and limitations

- `src/agent5.py`: separate LangGraph reasoning/tool loop, sequential execution,
  stop precedence within batches, and perception-specific instructions.
- `recognition_bridge.py`: common serialization, atomic frame publication,
  read-only `get_object_recognition` MCP tool, 3-second freshness limit.
- Controller: enables recognition, publishes frames, uses the Webots robot name
  when the environment variable is absent, and polls the existing stop flag.
- Demo world: recognition camera in the NAO head slot; no changes to stock Nao
  PROTO or to the original MCP-test world. Camera follows head rotation.
- Objects provide model labels, IDs, camera-relative positions in meters
  (x forward, y left, z up), Euclidean center distances, bearings, pixel boxes,
  and timestamps. Confidence is null because no learned confidence exists.
- Detection covers tagged objects visible to this camera; it is not an obstacle
  map. Occlusion checking is enabled. Only the new world has this added camera.
- Walking, collision-free approach, and grasping are not implemented. Agent5
  explicitly reports this rather than issuing an unverified walking sequence.
- Existing motion tools use a single command file. Use one agent per robot;
  other clients can overwrite commands. Tool success may mean queued, not completed.
- Stop polling halts active animations but is not a balance controller or a
  hardware emergency stop. Falling is still possible with arbitrary motions.
- Existing `get_robot_status` reports joint targets rather than measured joints.
  Built-in recognition positions are measured from the simulated camera frame.
- Built-in recognition must eventually be replaced with real perception for
  hardware deployment. The structured observation contract is the replacement point.

## Reinstall / restore

`prepare.py` stages controller/server changes from the sibling project;
`install.py` installs the staged files. The first install backs up changed existing
files to `*.pre-agent5`. Restore those two original files to undo controller/server
changes; the added world and bridge files can be removed separately. Restart the
MCP process and Webots controller after changing either implementation.
