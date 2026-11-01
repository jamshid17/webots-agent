"""Enable existing recognition MCP/controller in MCP-test.wbt with a small world patch."""
import argparse
import difflib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3] / 'webots-mcp'
WORLD = ROOT / 'worlds/MCP-test.wbt'
CAMERA = '''  headSlot [
    Camera {
      translation 0.0587 0 0.0636
      rotation 0 1 0 0.0209
      name "RecognitionCamera"
      width 320
      height 240
      fieldOfView 1.064
      recognition Recognition { maxRange 5 occlusion 2 }
    }
  ]
'''


def patch_world(original):
    if 'name "RecognitionCamera"' in original:
        raise ValueError('World already has a recognition camera; refusing duplicate insertion')
    if original.count('Nao {\n') != 2 or original.count('Rock {\n') != 1:
        raise ValueError('Unexpected world structure; review before patching')
    result = original.replace('Nao {\n', 'Nao {\n' + CAMERA)
    # Rabbit and SoccerBall already contain recognitionColors in their stock PROTOs.
    # Rock does not expose that field: label a static parent Solid instead.
    rock = 'Rock {\n  translation 0.28 -1.33 0\n  scale 2\n}'
    if rock not in result:
        raise ValueError('Rock configuration changed; review before patching')
    wrapper = '''Solid {
  name "recognized rock"
  model "rock"
  recognitionColors [ 0.5 0.5 0.5 ]
  children [
    Rock {
      translation 0.28 -1.33 0
      scale 2
    }
  ]
}'''
    return result.replace(rock, wrapper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    original = WORLD.read_text()
    updated = patch_world(original)
    print(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                     fromfile=str(WORLD), tofile=str(WORLD))))
    if args.install:
        backup = WORLD.with_name(WORLD.name + '.pre-recognition')
        if not backup.exists():
            shutil.copy2(WORLD, backup)
        WORLD.write_text(updated)
        print(f'Installed. Backup: {backup}')
