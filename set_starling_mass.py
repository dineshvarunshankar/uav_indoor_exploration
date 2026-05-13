from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Gf

path = "/home/ubuntu/uav_indoor/assets/starling_2/starling_2.usd"
stage = Usd.Stage.Open(path)

VALUES = {
    "base_link": {
        "mass": 0.240,
        "inertia": (9.0e-4, 6.0e-4, 1.2e-3),
        "com": (0.0, 0.0, 0.0),
    },
    "front_left_prop": {
        "mass": 0.015,
        "inertia": (6.0e-6, 3.5e-6, 8.5e-6),
        "com": (0.0, 0.0, 0.0),
    },
    "front_right_prop": {
        "mass": 0.015,
        "inertia": (6.0e-6, 3.5e-6, 8.5e-6),
        "com": (0.0, 0.0, 0.0),
    },
    "back_left_prop": {
        "mass": 0.015,
        "inertia": (6.0e-6, 3.5e-6, 8.5e-6),
        "com": (0.0, 0.0, 0.0),
    },
    "back_right_prop": {
        "mass": 0.015,
        "inertia": (6.0e-6, 3.5e-6, 8.5e-6),
        "com": (0.0, 0.0, 0.0),
    },
}

total = 0.0
for name, vals in VALUES.items():
    prim = stage.GetPrimAtPath(f"/starling_2/{name}")
    mass_api = UsdPhysics.MassAPI.Apply(prim)

    mass_api.CreateMassAttr().Set(float(vals["mass"]))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*vals["inertia"]))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*vals["com"]))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    total += vals["mass"]
    print(name, vals)

print("TOTAL MASS:", total)
stage.GetRootLayer().Save()
simulation_app.close()
