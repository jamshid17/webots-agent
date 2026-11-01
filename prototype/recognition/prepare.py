"""Stage reviewable changes; install.py performs the external writes."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'webots-mcp'

def original(relative):
    path = ROOT / relative
    backup = path.with_name(path.name + '.pre-agent5')
    return (backup if backup.exists() else path).read_text()

controller = original('controllers/my_controller_plus_mcp/my_controller_plus_mcp.py')
# Robot() must be instantiated only once; obtain the name before creating paths.
controller = controller.replace('robot_name = os.environ.get("WEBOTS_ROBOT_NAME")', 'robot = Robot()\nrobot_name = os.environ.get("WEBOTS_ROBOT_NAME") or robot.getName()')
controller = controller.replace('robot = Robot()\ntimestep', 'timestep')
controller = controller.replace('from controller import Motion, Robot', 'from controller import Motion, Robot\nfrom recognition_bridge import snapshot, publish')
controller = controller.replace('camera = robot.getDevice("CameraTop")', 'device_names = {robot.getDeviceByIndex(i).getName() for i in range(robot.getNumberOfDevices())}\n    camera = robot.getDevice("RecognitionCamera" if "RecognitionCamera" in device_names else "CameraTop")')
controller = controller.replace('camera.enable(timestep)', 'camera.enable(timestep)\n        if camera.hasRecognition():\n            camera.recognitionEnable(timestep)')
controller = controller.replace('    # Block commands during animation', '    # Poll the persistent stop flag before the animation busy check.\n    try:\n        stopped = json.loads((DATA_DIR / "stop_flag.json").read_text()).get("stopped", False)\n    except (OSError, ValueError):\n        stopped = False\n    if stopped:\n        if robot_state["current_motion"]:\n            robot_state["current_motion"].stop()\n            robot_state["current_motion"] = None\n        return\n    # Block commands during animation')
# Do not leave the transition loop oblivious to a stop request.
controller = controller.replace('        while robot.getTime() - start_time < transition_duration:', '''        while robot.getTime() - start_time < transition_duration:
            try:
                if json.loads((DATA_DIR / "stop_flag.json").read_text()).get("stopped"):
                    return
            except (OSError, ValueError):
                pass''')
controller = controller.replace('        "timestamp": current_time,', '        "timestamp": current_time,\n        "last_update": current_time,')
controller = controller.replace('        update_status()', '        update_status()\n        publish(DATA_DIR / "recognition.json", snapshot(camera, robot.getTime()))')
controller = controller.replace('        with open(STATUS_FILE, "w", encoding="utf-8") as f:\n            json.dump(status_data, f, indent=2, ensure_ascii=False)', '        publish(STATUS_FILE, status_data)')
(HERE/'my_controller_plus_mcp.py').write_text(controller)
server = original('webots_mcp_server.py').replace('print("herer")', '# Startup diagnostics must not corrupt MCP stdio.')
server = server.replace('# Initialization on load', 'from recognition_bridge import register_tools\nregister_tools(mcp, ROOT_DATA_DIR)\n\n# Initialization on load')
start = server.index('@mcp.tool()\ndef list_robots()')
end = server.index('\n\nfrom recognition_bridge import register_tools', start)
server = server[:start] + '@mcp.tool()\ndef list_robots() -> List[Dict[str, Any]]:\n    """List robots whose controllers published a heartbeat in the last 10 seconds.\n\n    Historical robot folders are excluded. Paused/stopped controllers expire.\n    """\n    from recognition_bridge import active_robots\n    return active_robots(ROOT_DATA_DIR)\n' + server[end:]
(HERE/'webots_mcp_server.py').write_text(server)
