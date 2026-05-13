from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import asyncio
from pathlib import Path

from omni.kit.asset_converter import AssetConverterContext, get_instance

MESH_DIR = Path.home() / "uav_indoor/assets/starling_2/meshes"

async def convert_one(src: Path, dst: Path):
    ctx = AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_cameras = True
    ctx.ignore_lights = True
    ctx.single_mesh = True
    ctx.use_meter_as_world_unit = False  # STL is in mm; preserve numeric coords for now

    task = get_instance().create_converter_task(str(src), str(dst), None, ctx)
    ok = await task.wait_until_finished()
    print(src.name, "->", dst.name, "OK" if ok else "FAILED")
    if not ok:
        print(task.get_status(), task.get_error_message())

async def main():
    for src in sorted(MESH_DIR.glob("*.stl")):
        dst = src.with_suffix(".usd")
        await convert_one(src, dst)

asyncio.get_event_loop().run_until_complete(main())

simulation_app.close()
