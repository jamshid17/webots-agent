"""Install staged prototype into the sibling Webots project, with backups."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / 'webots-mcp'
pairs = [
    (HERE/'webots_mcp_server.py', ROOT/'webots_mcp_server.py'),
    (HERE/'recognition_bridge.py', ROOT/'recognition_bridge.py'),
    (HERE/'my_controller_plus_mcp.py', ROOT/'controllers/my_controller_plus_mcp/my_controller_plus_mcp.py'),
    (HERE/'recognition_bridge.py', ROOT/'controllers/my_controller_plus_mcp/recognition_bridge.py'),
    (HERE/'agent5-recognition.wbt', ROOT/'worlds/agent5-recognition.wbt'),
]
for source, target in pairs:
    if target.exists() and source.read_bytes() == target.read_bytes():
        continue
    backup = target.with_name(target.name + '.pre-agent5')
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    print(f'Installed {target}')
