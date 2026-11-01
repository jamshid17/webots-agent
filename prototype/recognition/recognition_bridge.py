"""Shared Webots recognition serialization and read-only MCP tools."""
import asyncio
import json
import math
import re
import time
from pathlib import Path


def snapshot(camera, simulation_time):
    result = {"source": "webots_builtin_recognition", "timestamp": time.time(),
              "simulation_time": simulation_time, "camera": camera.getName() if camera else None,
              "frame": "camera (x forward, y left, z up)", "objects": [],
              "available": bool(camera and camera.hasRecognition())}
    if not result["available"]:
        result["error"] = "Camera has no Recognition node. Open agent5-recognition.wbt."
        return result
    result["image_size"] = [camera.getWidth(), camera.getHeight()]
    for obj in camera.getRecognitionObjects():
        position = list(obj.getPosition())
        result["objects"].append({"id": obj.getId(), "label": obj.getModel(),
            "position_m": position, "distance_m": math.sqrt(sum(v*v for v in position)),
            "bearing_rad": math.atan2(position[1], position[0]),
            "bbox_center_px": list(obj.getPositionOnImage()),
            "bbox_size_px": list(obj.getSizeOnImage()), "size_m": list(obj.getSize()),
            "confidence": None})
    result["objects"].sort(key=lambda obj: obj["distance_m"])
    return result


def publish(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_observation(root, robot_name, max_age=3.0):
    if not re.fullmatch(r"[\w .-]+", robot_name) or robot_name in {".", ".."}:
        return {"available": False, "error": "Invalid robot name"}
    try:
        result = json.loads((Path(root) / robot_name / "recognition.json").read_text())
        age = time.time() - result["timestamp"]
        if not 0 <= age <= max_age:
            return {"available": False, "error": "Recognition is stale; run/unpause Webots", "age_s": age}
        return dict(result, age_s=age, robot_name=robot_name)
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False, "error": "No valid recognition frame; start the updated controller and recognition world"}


def active_robots(root, max_age=10.0):
    """Discover live controller heartbeats, never historical data directories."""
    now = time.time()
    robots = []
    for path in sorted(Path(root).glob('*/status.json')):
        try:
            status = json.loads(path.read_text(encoding='utf-8'))
            timestamp = status.get('timestamp', status.get('last_update', 0))
            if status.get('webots_connected') and 0 <= now - timestamp <= max_age:
                robots.append({'name': path.parent.name,
                               'position': status.get('robot_position')})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return robots


def register_tools(mcp, root):
    @mcp.tool()
    async def get_object_recognition(robot_name: str) -> dict:
        """Observe currently visible objects using Webots built-in recognition. Returns labels, camera-relative metric positions, distances and pixel boxes. Simulator ground truth, not learned vision. Fresh frames only; an empty list means no tagged object is visible, not that space is clear."""
        # Allow a newly requested head movement to begin; read only, no commands.
        await asyncio.sleep(0.25)
        return read_observation(root, robot_name)
