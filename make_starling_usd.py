from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf

ASSET_DIR = Path.home() / "uav_indoor/assets/starling_2"
MESH_DIR = ASSET_DIR / "meshes"
USD_PATH = ASSET_DIR / "starling_2.usd"

SCALE = 0.001  # STL mm -> meters

# Centers measured from STL bounds, in meters, Fusion coordinates.
# Your STL coordinate convention appears: X left/right, Y up, Z front/back.
CENTERS = {
    "front_left_prop":  (-0.067263916, 0.011335515, -0.053611006),
    "front_right_prop": ( 0.067218193, 0.011335515, -0.053543396),
    "back_left_prop":   (-0.067218193, 0.011335515,  0.053543396),
    "back_right_prop":  ( 0.065842438, 0.011335515,  0.052976705),
}

MESHES = {
    "base_link": "base_link.stl",
    "front_left_prop": "front_left_prop.stl",
    "front_right_prop": "front_right_prop.stl",
    "back_left_prop": "back_left_prop.stl",
    "back_right_prop": "back_right_prop.stl",
}

MASSES = {
    "base_link": 1.2,
    "front_left_prop": 0.01,
    "front_right_prop": 0.01,
    "back_left_prop": 0.01,
    "back_right_prop": 0.01,
}

INERTIAS = {
    "base_link": (0.013, 0.014, 0.013),
    "front_left_prop": (1e-6, 1e-6, 1e-6),
    "front_right_prop": (1e-6, 1e-6, 1e-6),
    "back_left_prop": (1e-6, 1e-6, 1e-6),
    "back_right_prop": (1e-6, 1e-6, 1e-6),
}

def add_rigid_body(stage, path, mass, inertia, articulation_root=False):
    prim = stage.DefinePrim(path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if articulation_root:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        physx_art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        physx_art.CreateEnabledSelfCollisionsAttr(False)
    return prim

def add_mesh_under(stage, body_name, mesh_file, visual=True, offset=(0,0,0)):
    folder = "visuals" if visual else "collisions"
    xform_path = f"/starling_2/{body_name}/{folder}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    mesh_path = f"{xform_path}/{body_name}_{folder}_mesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Reference the STL file. Isaac/Usd can load STL via asset resolver/importer path.
    rel = f"./meshes/{mesh_file}"
    mesh.GetPrim().GetReferences().AddReference(rel)

    xf = UsdGeom.Xformable(mesh.GetPrim())
    # Convert mm to m, then offset visual into body-local frame.
    xf.AddTranslateOp().Set(Gf.Vec3d(*offset))
    xf.AddScaleOp().Set(Gf.Vec3f(SCALE, SCALE, SCALE))

    if not visual:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh

def add_fixed_joint(stage, child_name, pos):
    joint_path = f"/starling_2/joints/base_link_to_{child_name}"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/starling_2/base_link")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/starling_2/{child_name}")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos))
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

# Base: geometry already in assembly frame; body at origin.
add_rigid_body(stage, "/starling_2/base_link", MASSES["base_link"], INERTIAS["base_link"], articulation_root=True)
add_mesh_under(stage, "base_link", MESHES["base_link"], visual=True, offset=(0,0,0))
add_mesh_under(stage, "base_link", MESHES["base_link"], visual=False, offset=(0,0,0))

# Props: body origin at rotor center; mesh offset negative center so visual stays in original assembly position.
for name, center in CENTERS.items():
    add_rigid_body(stage, f"/starling_2/{name}", MASSES[name], INERTIAS[name])
    neg = tuple(-v / SCALE for v in center)  # translate before scale is in STL mm coordinates
    add_mesh_under(stage, name, MESHES[name], visual=True, offset=neg)
    add_mesh_under(stage, name, MESHES[name], visual=False, offset=neg)
    add_fixed_joint(stage, name, center)

stage.GetRootLayer().Save()
print("Wrote", USD_PATH)

simulation_app.close()