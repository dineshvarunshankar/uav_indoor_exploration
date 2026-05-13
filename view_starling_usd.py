from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.commands
from omni.kit.viewport.utility import get_active_viewport

from pxr import UsdGeom, Gf

usd_path = "/home/ubuntu/uav_indoor/assets/starling_2/starling_2.usd"

ctx = omni.usd.get_context()
ctx.open_stage(usd_path)

for _ in range(120):
    simulation_app.update()

stage = ctx.get_stage()

# Create a camera looking at the robot.
cam_path = "/World/Camera"
omni.kit.commands.execute(
    "CreatePrimWithDefaultXform",
    prim_type="Camera",
    prim_path=cam_path,
)

cam_prim = stage.GetPrimAtPath(cam_path)
xf = UsdGeom.Xformable(cam_prim)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.45, -0.65, 0.35))
xf.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 35.0))

cam = cam_prim
cam.GetAttribute("focalLength").Set(24.0)

viewport = get_active_viewport()
if viewport:
    viewport.camera_path = cam_path

print("Opened:", usd_path)
print("Camera:", cam_path)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
