from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf

ASSET_DIR = Path.home() / "uav_indoor/assets/starling_2"
MESH_DIR = ASSET_DIR / "meshes"
USD_PATH = ASSET_DIR / "starling_2.usd"

MESH_FILES = {
    "base_link": "base_link.usdz",
    "front_left_prop": "front_left_prop.usdz",
    "front_right_prop": "front_right_prop.usdz",
    "back_left_prop": "back_left_prop.usdz",
    "back_right_prop": "back_right_prop.usdz",
}

# Joint/body origins measured from your previous STL bounds, converted from
# Fusion frame (X left/right, Y up, Z front/back, mm) into Isaac frame
# (X front/back, Y left/right, Z up, m): (x,y,z)_isaac = (-Z, -X, Y) * 0.001
CENTERS = {
    "front_left_prop":  Gf.Vec3f( 0.053611006,  0.067263916, 0.011335515),
    "front_right_prop": Gf.Vec3f( 0.053543396, -0.067218193, 0.011335515),
    "back_left_prop":   Gf.Vec3f(-0.053543396,  0.067218193, 0.011335515),
    "back_right_prop":  Gf.Vec3f(-0.052976705, -0.065842438, 0.011335515),
}

def add_rigid_body(stage, path, mass, inertia, articulation_root=False, translate=None):
    prim = stage.DefinePrim(path, "Xform")
    if translate is not None:
        UsdGeom.Xformable(prim).AddTranslateOp().Set(translate)

    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    if articulation_root:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False)

    return prim

def reference_usdz(stage, prim_path, rel_asset_path, translate=None):
    prim = stage.DefinePrim(prim_path, "Xform")
    if translate is not None:
        UsdGeom.Xformable(prim).AddTranslateOp().Set(translate)
    prim.GetReferences().AddReference(rel_asset_path)
    return prim

def apply_collision_to_meshes(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.CollisionAPI.Apply(prim)
            count += 1
    return count

def add_fixed_joint(stage, child_name, pos):
    joint = UsdPhysics.FixedJoint.Define(stage, f"/starling_2/joints/base_link_to_{child_name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/starling_2/base_link")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/starling_2/{child_name}")])
    joint.CreateLocalPos0Attr(pos)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)
    return joint

ASSET_DIR.mkdir(parents=True, exist_ok=True)

stage = Usd.Stage.CreateNew(str(USD_PATH))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

root = UsdGeom.Xform.Define(stage, "/starling_2")
stage.SetDefaultPrim(root.GetPrim())
UsdGeom.Scope.Define(stage, "/starling_2/joints")

# Base: visual/collision references live under base_link.
add_rigid_body(stage, "/starling_2/base_link", 1.2, (0.013, 0.014, 0.013), articulation_root=True)
UsdGeom.Xform.Define(stage, "/starling_2/base_link/visuals")
UsdGeom.Xform.Define(stage, "/starling_2/base_link/collisions")
reference_usdz(stage, "/starling_2/base_link/visuals/base_link_asset", "./meshes/base_link.usdz")
reference_usdz(stage, "/starling_2/base_link/collisions/base_link_asset", "./meshes/base_link.usdz")
apply_collision_to_meshes(stage, "/starling_2/base_link/collisions/base_link_asset")

# Props: body prim is at rotor center. Referenced visual/collision is offset back
# by -center so geometry remains in the same assembled position.
for name in ["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"]:
    c = CENTERS[name]
    add_rigid_body(stage, f"/starling_2/{name}", 0.01, (1e-6, 1e-6, 1e-6), translate=c)
    UsdGeom.Xform.Define(stage, f"/starling_2/{name}/visuals")
    UsdGeom.Xform.Define(stage, f"/starling_2/{name}/collisions")

    offset = Gf.Vec3f(-c[0], -c[1], -c[2])
    reference_usdz(stage, f"/starling_2/{name}/visuals/{name}_asset", f"./meshes/{MESH_FILES[name]}", translate=offset)
    reference_usdz(stage, f"/starling_2/{name}/collisions/{name}_asset", f"./meshes/{MESH_FILES[name]}", translate=offset)
    apply_collision_to_meshes(stage, f"/starling_2/{name}/collisions/{name}_asset")
    add_fixed_joint(stage, name, c)

stage.GetRootLayer().Save()
print("Wrote", USD_PATH)

simulation_app.close()
