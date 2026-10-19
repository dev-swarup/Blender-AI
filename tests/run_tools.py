from __future__ import annotations

import contextlib
import faulthandler
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import addon_utils
import bpy
import devimport
import numpy as np

devimport.install()
addon_utils.enable("blender_ai", default_set=True)

from blender_ai import state
from blender_ai.imaging.png import encode_png
from blender_ai.integrations import registry as integrations
from blender_ai.preferences import get_prefs
from blender_ai.tools import registry
from blender_ai.tools.registry import AsyncJob, ToolError, ToolOutput

RESULTS: list[tuple[str, bool, str]] = []
TMP = tempfile.mkdtemp(prefix="blender_ai_tests_")


def run(tool_name: str, /, **args: Any) -> Any:
    spec = registry.REGISTRY.get(tool_name)
    assert spec is not None, f"no tool {tool_name}"
    out = registry.call(spec, args)
    while isinstance(out, AsyncJob):
        out = registry.normalize(out.finish(out.work()))
    assert isinstance(out, ToolOutput)
    return out


def fails(tool_name: str, /, **args: Any) -> str:
    try:
        run(tool_name, **args)
    except ToolError as exc:
        return str(exc)
    raise AssertionError(f"{tool_name} should have raised ToolError")


def case(fn: Callable[[], None]) -> Callable[[], None]:
    def wrapper() -> None:
        try:
            fn()
        except Exception as exc:
            RESULTS.append((fn.__name__, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"))
        else:
            RESULTS.append((fn.__name__, True, ""))

    wrapper.__name__ = fn.__name__
    CASES.append(wrapper)
    return wrapper


CASES: list[Callable[[], None]] = []


def fresh() -> None:
    bpy.ops.wm.read_homefile(use_empty=True)


def ob(name: str) -> bpy.types.Object:
    found = bpy.data.objects.get(name)
    assert found is not None, f"missing object {name}"
    return found


@case
def catalog_schemas_are_valid() -> None:
    specs = registry.REGISTRY.specs()
    assert len(specs) >= 60, len(specs)
    for spec in specs:
        assert spec.description and len(spec.description) > 20, spec.name
        assert spec.input_schema["type"] == "object"
        for key in spec.input_schema.get("required", []):
            assert key in spec.input_schema["properties"], (spec.name, key)


@case
def loaded_and_each_agent_gets_its_toolset() -> None:
    from blender_ai.agent import roles
    from blender_ai.tools.registry import Mode

    st = state.get()
    assert st.ready and st.session is not None, "background mode loads everything during register()"
    session = st.session

    def names(mode: Mode, depth: int = 0) -> set[str]:
        return {s.name for s in session._toolset(roles.info(mode), depth)}

    model, scene, animate, material = (names(m) for m in roles.ORDER)
    for tools in (model, scene, animate, material):
        assert {"run_python", "spawn_agents", "create_primitive", "get_scene_info", "import_from_url"} <= tools
    assert {"bmesh_extrude", "smart_uv_unwrap", "create_curve", "add_modifier"} <= model
    assert {"add_light", "set_world", "scatter_on_surface", "add_physics"} <= scene
    assert {"camera_orbit", "insert_keyframe", "create_armature", "bake_physics"} <= animate
    assert {"generate_texture", "reference_to_pbr", "smart_uv_unwrap"} <= material
    assert "insert_keyframe" not in model and "bmesh_extrude" not in material
    assert "spawn_agents" not in names(Mode.MODEL, roles.MAX_DEPTH)
    every = set().union(model, scene, animate, material) - {"spawn_agents"}
    missing = {s.name for s in registry.REGISTRY.specs()} - every
    assert not missing, f"tools no agent can use: {sorted(missing)}"


@case
def primitives_and_transforms() -> None:
    fresh()
    for kind in ("cube", "uv_sphere", "ico_sphere", "cylinder", "cone", "torus", "plane", "grid", "circle", "monkey", "empty", "text", "bezier_curve"):
        run("create_primitive", kind=kind, name=f"P_{kind}", size=2.0)
    cube = ob("P_cube")
    assert len(cube.data.vertices) == 8 and cube.data.uv_layers, "cube with UVs"
    assert abs(cube.dimensions.x - 2.0) < 1e-4
    assert abs(ob("P_plane").dimensions.x - 2.0) < 1e-4, ob("P_plane").dimensions
    assert len(ob("P_torus").data.polygons) == 48 * 12
    run("create_primitive", kind="plane", name="Ground", size=30.0, collection="Terrain")
    assert [c.name for c in ob("Ground").users_collection] == ["Terrain"], "a missing collection is created"
    assert "Terrain" in bpy.context.scene.collection.children
    run("transform_object", name="P_cube", location=[1, 2, 3], rotation_deg=[0, 0, 90], scale=[2, 1, 1], apply_scale=True)
    assert tuple(round(v, 4) for v in cube.scale) == (1, 1, 1)
    assert abs(cube.dimensions.x - 4.0) < 1e-4
    run("set_object_props", name="P_uv_sphere", shading="smooth", new_name="Ball")
    assert ob("Ball").data.polygons[0].use_smooth
    run("set_object_props", name="Ball", shading="auto")


@case
def organise_scene() -> None:
    fresh()
    run("create_primitive", kind="cube", name="Box")
    run("duplicate_object", name="Box", count=3, offset=[2, 0, 0])
    assert len([o for o in bpy.data.objects if o.name.startswith("Box")]) == 4
    run("parent_objects", children=["Box.001", "Box.002"], parent="Box")
    assert ob("Box.001").parent == ob("Box")
    run("set_origin", name="Box.003", to="bounds_bottom")
    assert abs(ob("Box.003").location.z + 0.5) < 1e-4
    run("collection_ops", action="create", collection="Props", objects=["Box.003"])
    assert ob("Box.003").users_collection[0].name == "Props"
    run("select_objects", names=["Box", "Box.003"])
    run("join_objects", names=["Box", "Box.003"])
    assert "Box.003" not in bpy.data.objects
    run("delete_objects", names=["Box.002"])
    assert "Box.002" not in bpy.data.objects
    info = run("get_scene_info", detail=True).text
    assert "Box" in info and "Props" in info
    assert "Did you mean" in fails("transform_object", name="Boks", location=[0, 0, 0])


@case
def aim_and_constraints() -> None:
    fresh()
    run("create_primitive", kind="cube", name="Target", location=[0, 0, 0])
    run("add_camera", name="Cam", location=[5, -5, 3], target="Target")
    cam = ob("Cam")
    bpy.context.view_layer.update()
    forward = cam.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, -1))
    to_target = (ob("Target").location - cam.location).normalized()
    assert forward.normalized().dot(to_target) > 0.999
    run("add_object_constraint", name="Cam", type="TRACK_TO", target="Target")
    assert cam.constraints[0].type == "TRACK_TO"
    run("look_at", name="Cam", point=[0, 0, 5])


@case
def import_obj_round_trip() -> None:
    fresh()
    run("create_primitive", kind="monkey", name="Suzanne")
    path = os.path.join(TMP, "suzanne.obj")
    bpy.ops.wm.obj_export(filepath=path)
    fresh()
    out = run("import_asset", filepath=path, collection="Imported")
    assert '"count": 1' in out.text, out.text


@case
def mesh_editing() -> None:
    fresh()
    run(
        "create_mesh",
        name="Pyramid",
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1.5]],
        faces=[[3, 2, 1, 0], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    )
    assert len(ob("Pyramid").data.polygons) == 5
    run("create_primitive", kind="cube", name="Block", size=2)
    run("bmesh_extrude", name="Block", selection={"by": "normal", "direction": [0, 0, 1], "angle_deg": 10}, distance=1.0)
    assert abs(ob("Block").dimensions.z - 3.0) < 1e-4
    run("bmesh_inset", name="Block", selection={"by": "normal", "direction": [0, 0, 1]}, thickness=0.2, depth=-0.1)
    faces_before = len(ob("Block").data.polygons)
    run("bmesh_loop_cut", name="Block", axis="Z", cuts=2)
    assert len(ob("Block").data.polygons) > faces_before
    run("bmesh_bevel", name="Block", selection={"by": "sharp", "angle_deg": 60}, width=0.05, segments=2)
    run("transform_elements", name="Block", selection={"by": "position", "min": [-5, -5, 1.5], "max": [5, 5, 5]}, scale=[0.5, 0.5, 1])
    run("subdivide", name="Pyramid", cuts=1)
    run("merge_by_distance", name="Block")
    run("recalc_normals", name="Block")
    run("create_primitive", kind="cube", name="Wall", size=2)
    run("create_primitive", kind="cylinder", name="Hole", size=0.8, depth=4, rotation_deg=[90, 0, 0])
    verts = len(ob("Wall").data.vertices)
    run("bmesh_boolean", name="Wall", cutter="Hole", operation="DIFFERENCE")
    assert len(ob("Wall").data.vertices) > verts and not ob("Wall").modifiers
    import math

    ring = [[math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)] for i in range(8)]
    verts = [[x, y, 0.0] for x, y in ring] + [[x, y, 1.0] for x, y in ring]
    edges = [[i, (i + 1) % 8] for i in range(8)] + [[8 + i, 8 + (i + 1) % 8] for i in range(8)]
    run("create_mesh", name="Rings", vertices=verts, edges=edges)
    run("bridge_edge_loops", name="Rings", selection={"by": "all"})
    assert len(ob("Rings").data.polygons) == 8, len(ob("Rings").data.polygons)
    run("create_primitive", kind="cylinder", name="Tube", size=1, depth=2)
    run("delete_elements", name="Tube", selection={"by": "normal", "direction": [0, 0, 1], "angle_deg": 5}, kind="FACES")
    assert len(ob("Tube").data.polygons) == 32 + 1
    assert "matched no" in fails("bmesh_extrude", name="Block", selection={"by": "position", "min": [50, 50, 50], "max": [60, 60, 60]})


@case
def modifiers_and_uvs() -> None:
    fresh()
    run("create_primitive", kind="cube", name="Crate")
    out = run("add_modifier", name="Crate", type="BEVEL", settings={"width": 0.05, "segments": 3, "angle_limit": 40})
    assert '"segments": 3' in out.text
    run("set_modifier_prop", name="Crate", modifier="Bevel", settings={"segments": 2})
    run("apply_modifier", name="Crate", modifier="Bevel")
    assert len(ob("Crate").data.vertices) > 8 and not ob("Crate").modifiers
    run("add_modifier", name="Crate", type="SUBSURF", settings={"levels": 1})
    run("remove_modifier", name="Crate", modifier="Subsurf")
    assert "Did you mean" in fails("add_modifier", name="Crate", type="BEVEL", settings={"widht": 0.1})
    assert not ob("Crate").modifiers, "failed add_modifier must not leave a half-configured modifier"
    run("mark_seams", name="Crate", selection={"by": "sharp"})
    run("smart_uv_unwrap", names=["Crate"])
    run("smart_uv_unwrap", names=["Crate"], method="cube")
    run("pack_uvs", names=["Crate"])
    assert ob("Crate").data.uv_layers.active is not None


def _noise_png(path: str, size: int = 128) -> None:
    rng = np.random.default_rng(1)
    base = rng.random((size // 8, size // 8, 3)).astype(np.float32)
    img = np.kron(base, np.ones((8, 8, 1), dtype=np.float32))
    with open(path, "wb") as f:
        f.write(encode_png(img))


@case
def materials() -> None:
    fresh()
    run("create_primitive", kind="cube", name="Box")
    run("create_material", name="Red", base_color=[0.8, 0.05, 0.05], roughness=0.4, assign_to=["Box"])
    assert ob("Box").material_slots[0].material.name == "Red"
    run("create_material", name="Gold", base_color=[1, 0.8, 0.3], metallic=1.0)
    run("assign_material", objects=["Box"], material="Gold", faces={"by": "normal", "direction": [0, 0, 1]})
    assert any(p.material_index == 1 for p in ob("Box").data.polygons)
    run("set_principled", material="Gold", inputs={"Coat Weight": 0.5, "roughness": 0.2})
    assert "Did you mean" in fails("set_principled", material="Gold", inputs={"Rougness": 0.1})


@case
def pbr_from_reference_and_files() -> None:
    fresh()
    prefs = get_prefs()
    prefs.cache_dir = TMP
    run("create_primitive", kind="cube", name="Box")
    src = os.path.join(TMP, "ref.png")
    _noise_png(src)
    out = run("reference_to_pbr", image=src, material="Stone", objects=["Box"], tiling=2.0)
    assert out.images, "preview image returned"
    mat = bpy.data.materials["Stone"]
    images = {n.image.name for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}
    assert len(images) >= 5, images
    normal = next(n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and n.image.name.endswith("_normal"))
    assert normal.image.colorspace_settings.name != "sRGB"
    run("apply_pbr_set", material="Files", base_color=src, roughness=src, normal=src, directx_normal=True, objects=["Box"])
    assert ob("Box").material_slots[0].material.name == "Files"


@case
def generate_texture_with_fake_images_api() -> None:
    fresh()
    from blender_ai.tools import textures

    prefs = get_prefs()
    prefs.cache_dir = TMP
    prefs.provider = "OPENAI"
    prefs.openai_key = "test-key"
    prefs.image_model = "gpt-image-1"
    src = os.path.join(TMP, "gen.png")
    _noise_png(src, 64)
    with open(src, "rb") as f:
        data = f.read()
    seen: dict[str, Any] = {}

    def fake_generate(cfg: Any, prompt: str, reference: bytes | None = None) -> bytes:
        seen["prompt"], seen["reference"] = prompt, reference
        return data

    real = textures.image_api.generate
    textures.image_api.generate = fake_generate
    try:
        run("create_primitive", kind="plane", name="Floor", size=4)
        out = run("generate_texture", prompt="mossy cobblestones", material="Cobbles", objects=["Floor"])
    finally:
        textures.image_api.generate = real
        prefs.openai_key = ""
    assert "Seamless tileable" in seen["prompt"] and "mossy cobblestones" in seen["prompt"]
    assert "Cobbles" in out.text and ob("Floor").material_slots[0].material.name == "Cobbles"


@case
def lights_world_render() -> None:
    fresh()
    run("create_primitive", kind="monkey", name="Subject")
    for kind in ("POINT", "SUN", "SPOT", "AREA"):
        run("add_light", type=kind, name=f"L_{kind}", location=[2, -2, 3], target="Subject", size=0.5, color=[1, 0.9, 0.8])
    run("set_light", name="L_AREA", props={"energy": 750})
    assert bpy.data.lights["L_AREA"].energy == 750
    run("add_camera", name="Cam", location=[4, -4, 2], target="Subject", lens_mm=35)
    run("set_camera", name="Cam", focus_object="Subject", fstop=2.8)
    assert bpy.context.scene.camera == ob("Cam") and ob("Cam").data.dof.use_dof
    run("set_world", mode="color", color=[0.1, 0.1, 0.12])
    sky = run("set_world", mode="sky", sun_elevation_deg=20)
    assert "sky_type" in sky.text
    hdri_dir = os.path.join(bpy.utils.system_resource("DATAFILES"), "studiolights", "world")
    hdris = [f for f in os.listdir(hdri_dir) if f.endswith((".exr", ".hdr"))] if os.path.isdir(hdri_dir) else []
    if hdris:
        run("set_world", mode="hdri", hdri_path=os.path.join(hdri_dir, hdris[0]), rotation_deg=45)
    run("set_render", engine="workbench", resolution=[320, 240], film_transparent=True)
    preview = run("render_preview", width=160, engine="workbench")
    assert preview.images and preview.images[0].data[:4] == b"\x89PNG"
    assert bpy.context.scene.render.resolution_percentage == 100, "settings restored after preview"
    assert "3D Viewport" in fails("capture_viewport")
    assert "3D Viewport" in fails("render_turntable")


def _leg_rig() -> None:
    run("create_primitive", kind="cylinder", name="Leg", size=0.3, depth=2, location=[0, 0, 1], segments=16)
    run("subdivide", name="Leg", cuts=4)
    run(
        "create_armature",
        name="Rig",
        bones=[
            {"name": "thigh.L", "head": [0, 0, 2], "tail": [0, 0, 1]},
            {"name": "shin.L", "head": [0, 0, 1], "tail": [0, 0, 0.05], "parent": "thigh.L", "connect": True},
        ],
    )


@case
def rigging() -> None:
    fresh()
    _leg_rig()
    arm = ob("Rig")
    assert [b.name for b in arm.data.bones] == ["thigh.L", "shin.L"]
    run("add_bone", armature="Rig", name="foot.L", head=[0, 0, 0.05], tail=[0, -0.3, 0.0], parent="shin.L", connect=True)
    run("edit_bone", armature="Rig", bone="foot.L", roll_deg=10)
    run("mirror_bones", armature="Rig")
    assert "thigh.R" in arm.data.bones and arm.data.bones["shin.R"].parent.name == "thigh.R"
    out = run("parent_with_auto_weights", meshes=["Leg"], armature="Rig")
    assert ob("Leg").parent == arm and any(m.type == "ARMATURE" for m in ob("Leg").modifiers)
    assert '"weighted_vertices": 0' not in out.text, out.text
    run("set_vertex_group_weights", mesh="Leg", group="thigh.L", selection={"by": "position", "min": [-1, -1, 1.5], "max": [1, 1, 3]}, weight=1.0)
    run("add_ik_constraint", armature="Rig", bone="shin.L", chain_count=2)
    assert "IK_shin.L" in arm.data.bones and arm.pose.bones["shin.L"].constraints[0].type == "IK"
    run("add_bone_constraint", armature="Rig", bone="foot.L", type="LIMIT_ROTATION", props={"use_limit_x": True, "min_x": -30})
    run("add_shape_key", name="Leg", key_name="Bulge", selection={"by": "position", "min": [-1, -1, 0.7], "max": [1, 1, 1.3]}, scale=[1.3, 1.3, 1])
    run("set_shape_key_value", name="Leg", key="Bulge", value=1.0, frame=10)
    keys = ob("Leg").data.shape_keys.key_blocks
    assert [k.name for k in keys] == ["Basis", "Bulge"]
    run(
        "add_driver",
        target="Leg",
        shape_key="Bulge",
        expression="var * 2",
        variables=[{"name": "var", "type": "TRANSFORMS", "object": "Rig", "bone": "shin.L", "transform_type": "ROT_X"}],
    )
    assert ob("Leg").data.shape_keys.animation_data.drivers


@case
def animation_basics() -> None:
    fresh()
    _leg_rig()
    run("create_primitive", kind="cube", name="Box")
    run("insert_keyframe", name="Box", data_path="location", frame=1, value=[0, 0, 0])
    run("insert_keyframe", name="Box", data_path="location", frame=24, value=[0, 0, 3], interpolation="LINEAR")
    run("insert_keyframe", name="Box", data_path="rotation_euler", frame=24, value=[0, 0, 90])
    from blender_ai.tools.animation import fcurves_of

    paths = {f.data_path for f in fcurves_of(ob("Box"))}
    assert {"location", "rotation_euler"} <= paths, paths
    run("set_interpolation", name="Box", interpolation="BEZIER", data_path_contains="location")
    curve = run("edit_fcurve", name="Box", data_path="location", index=2, add_modifier={"type": "CYCLES"})
    assert "CYCLES" in curve.text
    run("pose_bone", armature="Rig", bone="shin.L", rotation_deg=[30, 0, 0], frame=12)
    run("set_frame_range", start=1, end=48, fps=30)
    assert bpy.context.scene.frame_end == 48 and bpy.context.scene.render.fps == 30
    run("nla_push_down", name="Box")
    assert ob("Box").animation_data.nla_tracks and ob("Box").animation_data.action is None
    run("create_action", name="Box", action_name="Spin")
    run("insert_keyframe", name="Box", data_path="rotation_euler", frame=1, value=[0, 0, 0])
    run("insert_keyframe", name="Box", data_path="rotation_euler", frame=10, value=[0, 0, 180])
    strip = run("nla_add_strip", name="Box", action="Spin", start_frame=30, repeat=3)
    assert '"frames"' in strip.text
    light_out = run("add_light", type="POINT", name="Lamp")
    assert "Lamp" in light_out.text
    run("insert_keyframe", name="Lamp", data_path="data.energy", frame=1, value=100)
    assert fcurves_of(bpy.data.lights["Lamp"])


@case
def camera_moves_and_bake() -> None:
    fresh()
    run("create_primitive", kind="monkey", name="Hero")
    run("add_camera", name="Cam", location=[6, -6, 3], target="Hero")
    out = run("camera_orbit", camera="Cam", target="Hero", frames=100)
    pivot = ob(out.text.split('"pivot": "')[1].split('"')[0])
    assert ob("Cam").parent == pivot
    run("add_camera", name="Dolly", location=[0, -8, 2], make_active=False)
    run("camera_dolly", camera="Dolly", end=[0, -4, 2], frames=50, target="Hero")
    run("add_camera", name="Flyer", make_active=False)
    run("camera_path", camera="Flyer", points=[[0, -8, 2], [6, 0, 3], [0, 8, 2]], frames=80, target="Hero")
    assert any(c.type == "FOLLOW_PATH" for c in ob("Flyer").constraints)
    run("bake_action", name="Flyer", start=1, end=20, clear_constraints=True)
    assert not any(c.type == "FOLLOW_PATH" for c in ob("Flyer").constraints)


@case
def rigify_and_walk_cycle() -> None:
    fresh()
    if not integrations.available("rigify"):
        raise AssertionError("Rigify should always be available (bundled)")
    out = run("add_rigify_metarig", kind="human", name="Meta", height=1.8)
    assert len(ob("Meta").data.bones) > 50, out.text[:200]
    gen = run("generate_rigify_rig", metarig="Meta")
    rig_name = gen.text.split('"rig": "')[1].split('"')[0]
    cycle = run("build_cycle", armature=rig_name, cycle="walk", frames=24)
    assert "thigh" in cycle.text, cycle.text
    from blender_ai.tools.animation import fcurves_of

    assert len(fcurves_of(ob(rig_name))) >= 8
    run("nla_push_down", name=rig_name)


@case
def helpers_installer_enables_and_explains() -> None:
    from blender_ai.integrations import installer

    pending = [h.name for h in installer.downloads()]
    if pending and not bpy.app.online_access:
        try:
            installer.install(set(integrations.HELPERS), interactive=False)
        except installer.InstallError as exc:
            assert "Online access is off" in str(exc), exc
        else:
            raise AssertionError(f"downloading {pending} offline should fail with a clear message")
    elif not pending:
        installer.install(set(integrations.HELPERS), interactive=False)
    ready = {h.id for h in integrations.HELPERS.values()} - {h.id for h in integrations.missing()}
    installed = {h.id for h in integrations.HELPERS.values() if integrations.installed_module(h.id)}
    assert installed <= ready, f"installed helpers must end up enabled: {installed - ready}"
    assert "rigify" in ready


@case
def helper_integrations_when_present() -> None:
    fresh()
    for module in ("bl_ext.blender_org.mpfb", "bl_ext.blender_org.looptools"):
        with contextlib.suppress(Exception):
            addon_utils.enable(module, default_set=True)
    report = integrations.report()
    if report["looptools"]["usable"]:
        run("create_primitive", kind="circle", name="Ring", size=1, segments=12, fill=False)
        run("transform_elements", name="Ring", selection={"by": "indices", "indices": [0]}, translate=[0.2, 0, 0])
        run("loop_circle", name="Ring", selection={"by": "all"})
    if report["mpfb"]["usable"]:
        out = run("mpfb_create_human", name="Person", gender=1.0, age=0.5, rig="default")
        assert "Person" in out.text
        assets = run("mpfb_list_assets", kind="skins")
        assert '"kind": "skins"' in assets.text
    print("helpers:", {k: v["usable"] for k, v in report.items()})


@case
def python_runs_unrestricted() -> None:
    fresh()
    code = "import os, math\nresult = len(D.objects) + math.floor(1.5)\nprint('hi', bool(os.sep))"
    out = run("run_python", code=code, purpose="test")
    assert "hi True" in out.text and "result = 1" in out.text, out.text
    error = fails("run_python", code="x = 1\nraise ValueError('nope')", purpose="test")
    assert "line 2" in error and "ValueError: nope" in error, error
    caps = run("get_capabilities").text
    assert "rigify" in caps and "mcp_servers" in caps and "image_generation" in caps


@case
def curves_and_text() -> None:
    fresh()
    tube = run("create_curve", points=[[0, 0, 0], [1, 0, 0.5], [2, 1, 0]], name="Cable", thickness=0.05).text
    assert '"type": "CURVE"' in tube and abs(ob("Cable").data.bevel_depth - 0.05) < 1e-6, tube
    mesh = run("create_curve", points=[[0, 0, 0], [0, 2, 0]], name="Pipe", kind="POLY", thickness=0.1, convert_to_mesh=True)
    assert ob("Pipe").type == "MESH" and len(ob("Pipe").data.vertices) > 8, mesh.text
    run("create_text", text="HELLO", name="Sign", size=1.0, extrude=0.1, rotation=[90, 0, 0])
    assert ob("Sign").type == "FONT" and ob("Sign").data.body == "HELLO"
    run("create_text", text="AI", name="Logo", convert_to_mesh=True)
    assert ob("Logo").type == "MESH"


@case
def scatter_with_geometry_nodes() -> None:
    fresh()
    run("create_primitive", kind="plane", name="Ground", size=10)
    run("create_primitive", kind="ico_sphere", name="Rock", size=0.3)
    run("create_primitive", kind="cone", name="Tree", size=0.5)
    out = run("scatter_on_surface", target="Ground", objects=["Rock", "Tree"], count=50, seed=3).text
    assert "Scatter" in out
    ground = ob("Ground")
    assert ground.modifiers[-1].type == "NODES"
    depsgraph = bpy.context.evaluated_depsgraph_get()
    instances = sum(1 for inst in depsgraph.object_instances if inst.is_instance and inst.parent.name == "Ground")
    assert 25 <= instances <= 100, instances


@case
def physics_rigid_body_and_cloth() -> None:
    fresh()
    run("create_primitive", kind="plane", name="Floor", size=6)
    run("create_primitive", kind="cube", name="Box", size=0.5, location=[0, 0, 3])
    run("add_physics", objects=["Floor"], type="rigid_body_passive")
    run("add_physics", objects=["Box"], type="rigid_body", mass=2, bounciness=0.3)
    assert ob("Box").rigid_body.type == "ACTIVE" and ob("Box").rigid_body.mass == 2
    assert ob("Floor").rigid_body.type == "PASSIVE"
    run("create_primitive", kind="grid", name="Sheet", size=2, location=[3, 0, 2])
    run("add_physics", objects=["Sheet"], type="cloth", cloth_preset="silk")
    assert any(m.type == "CLOTH" for m in ob("Sheet").modifiers)
    out = run("bake_physics", frame_start=1, frame_end=12).text
    assert "Baked" in out
    bpy.context.scene.frame_set(12)
    assert ob("Box").matrix_world.translation.z < 3.0, "the box fell"


@case
def import_model_from_url() -> None:
    import threading
    import zipfile
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    fresh()
    run("create_primitive", kind="cube", name="Crate", size=2)
    folder = os.path.join(TMP, "served")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "crate.obj")
    bpy.ops.wm.obj_export(filepath=path, export_materials=False)
    with zipfile.ZipFile(os.path.join(folder, "crate.zip"), "w") as archive:
        archive.write(path, "model/crate.obj")
    fresh()

    class Quiet(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, directory=folder, **k)

        def log_message(self, *a: Any) -> None: ...

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = run("import_from_url", url=f"{base}/crate.obj", name="Imported", location=[1, 2, 0], target_size=1.0)
        imported = ob("Imported")
        bpy.context.view_layer.update()
        lo = min((imported.matrix_world @ v.co).z for v in imported.data.vertices)
        size = max(imported.dimensions)
        assert abs(lo) < 1e-3 and abs(size - 1.0) < 1e-3, (lo, size, out.text)
        run("import_from_url", url=f"{base}/crate.zip", name="Zipped")
        assert "Zipped" in bpy.data.objects
        assert "http(s)" in fails("import_from_url", url="file:///etc/passwd")
    finally:
        httpd.shutdown()


@case
def new_images_are_packed_into_the_blend() -> None:
    fresh()
    prefs = get_prefs()
    assert prefs.pack_textures, "packing is on by default"
    outside = os.path.join(TMP, "users_own.png")
    with open(outside, "wb") as f:
        f.write(encode_png(np.full((4, 4, 3), 200, dtype=np.uint8)))
    mine = bpy.data.images.load(outside)
    host = state.get().session.host
    code = (
        f"img = D.images.load(r'{outside}', check_existing=False)\nimg.name = 'FromFile'\n"
        "gen = D.images.new('Painted', 8, 8)\ngen.pixels[:] = [0.5] * (8 * 8 * 4)\n"
        "D.images.new('Blank', 8, 8)\n"
    )
    host.execute(registry.REGISTRY.get("run_python"), {"code": code, "purpose": "test"})
    assert bpy.data.images["FromFile"].packed_file is not None
    assert bpy.data.images["Painted"].packed_file is not None, "painted generated images are packed"
    assert bpy.data.images["Blank"].packed_file is None, "blank generated images rebuild themselves"
    assert mine.packed_file is None, "images that existed before the tool call stay external"
    prefs.pack_textures = False
    try:
        code = f"D.images.load(r'{outside}', check_existing=False).name = 'Loose'"
        host.execute(registry.REGISTRY.get("run_python"), {"code": code, "purpose": "t"})
        assert bpy.data.images["Loose"].packed_file is None
    finally:
        prefs.pack_textures = True


@case
def agent_session_with_scripted_agent() -> None:
    import time

    fresh()
    from blender_ai.agent.config import AgentConfig
    from blender_ai.agent.loop import AgentSession
    from blender_ai.agent.messages import AgentEvent

    class Scripted:
        has_history = False

        def __init__(self, cfg: Any) -> None:
            self.results: list[Any] = []

        def reset(self) -> None: ...

        def run(self, run: Any, turn: Any) -> str:
            if turn.text.startswith("Build a cone"):
                run.emit(AgentEvent("tool", name="create_primitive", call_id="c1"))
                self.results.append(run.call_tool("c1", "create_primitive", {"kind": "cone", "name": "ChildCone"}))
                return "Built ChildCone at the origin."
            run.emit(AgentEvent("tool", name="create_primitive", call_id="t1"))
            self.results.append(run.call_tool("t1", "create_primitive", {"kind": "cube", "name": "AgentCube"}))
            self.results.append(run.feedback())
            task = {"agent": "model_builder", "task": "Build a cone named ChildCone.", "name": "Cone"}
            run.emit(AgentEvent("tool", name="spawn_agents", call_id="s1"))
            self.results.append(run.call_tool("s1", "spawn_agents", {"tasks": [task]}))
            run.emit(AgentEvent("text", key="2:0", text="Made a cube."))
            return "Made a cube."

    st = state.get()
    agents: list[Scripted] = []

    def factory(cfg: Any) -> Scripted:
        agents.append(Scripted(cfg))
        return agents[-1]

    host = st.session.host
    host.agent_config = lambda: AgentConfig("OPENAI", "http://localhost:1", "k", "m")
    host.preflight = lambda cfg: None
    session = AgentSession(host, st.bridge, agent_factory=factory)
    session.submit("make a cube")
    end = time.monotonic() + 20
    while session.busy and time.monotonic() < end:
        st.bridge.drain()
        time.sleep(0.01)
    assert not session.busy
    assert "AgentCube" in bpy.data.objects and "ChildCone" in bpy.data.objects
    assert agents[0].results[0].text.startswith("{") and not agents[0].results[0].is_error
    spawned = agents[0].results[2]
    assert not spawned.is_error and "Built ChildCone" in spawned.text, spawned.text
    assert any(e.kind.value == "agent" and e.text == "Model Builder · Cone" for e in session.log)
    assert session.log[-1].text == "Made a cube."


def main() -> int:
    faulthandler.enable()
    only = os.environ.get("BLENDER_AI_TESTS")
    for fn in CASES:
        if only and fn.__name__ not in only.split(","):
            continue
        print(f"RUN   {fn.__name__}", flush=True)
        fn()
        name, ok, detail = RESULTS[-1]
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else "\n      " + detail.replace("\n", "\n      ")), flush=True)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    addon_utils.disable("blender_ai")
    sys.exit(code)
