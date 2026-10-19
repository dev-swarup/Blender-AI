from __future__ import annotations

from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector

from .common import (
    SELECTION,
    bmesh_of,
    describe,
    euler_deg,
    get_mesh_object,
    link,
    pick_edges,
    pick_faces,
    pick_verts,
    require_selection,
    unique_name,
    world_bounds,
)
from .registry import ToolError, array, boolean, integer, number, obj, string, tool, vec3


def _sel(description: str) -> dict[str, Any]:
    return {**SELECTION, "description": description}


@tool(
    obj(
        {
            "name": string("Name for the new object."),
            "vertices": array(vec3(), "Vertex positions in local space (meters).", min_items=1),
            "faces": array(array({"type": "integer", "minimum": 0}, min_items=3), "Faces as vertex-index lists (counter-clockwise seen from outside)."),
            "edges": array(array({"type": "integer", "minimum": 0}, min_items=2, max_items=2), "Loose edges as index pairs."),
            "location": vec3("World location of the origin."),
            "rotation_deg": vec3("Rotation in degrees."),
            "smooth": boolean("Shade smooth.", default=False),
            "collection": string("Collection for the object, created if missing."),
        },
        ["name", "vertices"],
    ),
    group="mesh",
)
def create_mesh(
    name: str,
    vertices: list[list[float]],
    faces: list[list[int]] | None = None,
    edges: list[list[int]] | None = None,
    location: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    smooth: bool = False,
    collection: str | None = None,
) -> dict[str, Any]:
    """Create a mesh object from explicit vertices and faces. Use for custom profiles and shapes that
    primitives can't give you; keep faces as quads where you can."""
    n = len(vertices)
    for f in faces or []:
        if any(i >= n for i in f):
            raise ToolError(f"Face {f} uses a vertex index >= {n}.")
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in vertices], [tuple(e) for e in edges or []], [tuple(f) for f in faces or []])
    if me.validate(verbose=False):
        me.update()
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
    ob = bpy.data.objects.new(unique_name(name, bpy.data.objects), me)
    link(ob, collection)
    if location:
        ob.location = location
    if rotation_deg:
        ob.rotation_euler = euler_deg(rotation_deg)
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Faces to extrude."),
            "distance": number("Distance along the face normals, in meters (negative pushes in).", default=0.1),
            "vector": vec3("Move the extruded faces by this world-space vector instead of along normals."),
            "individual": boolean("Extrude each face separately along its own normal.", default=False),
        },
        ["name", "selection"],
    ),
    group="mesh",
)
def bmesh_extrude(name: str, selection: dict[str, Any], distance: float = 0.1, vector: list[float] | None = None, individual: bool = False) -> dict[str, Any]:
    """Extrude faces outwards (or inwards) as a region or individually."""
    ob = get_mesh_object(name)
    to_local = ob.matrix_world.to_3x3().inverted_safe()
    with bmesh_of(ob) as bm:
        faces = pick_faces(bm, ob, selection)
        require_selection(faces, "faces")
        if individual:
            res = bmesh.ops.extrude_discrete_faces(bm, faces=faces)
            for f in res["faces"]:
                move = (to_local @ Vector(vector)) if vector else f.normal * distance
                bmesh.ops.translate(bm, vec=move, verts=list(f.verts))
            count = len(res["faces"])
        else:
            normal = sum((f.normal for f in faces), Vector()).normalized()
            res = bmesh.ops.extrude_face_region(bm, geom=faces)
            verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
            move = (to_local @ Vector(vector)) if vector else normal * distance
            bmesh.ops.translate(bm, vec=move, verts=verts)
            bmesh.ops.delete(bm, geom=faces, context="FACES")
            count = len(faces)
    return {"extruded_faces": count, **describe(ob)}


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Faces to inset."),
            "thickness": number("Inset width in meters.", minimum=0.0, default=0.05),
            "depth": number("Raise (+) or lower (-) the inset faces, in meters.", default=0.0),
            "individual": boolean("Inset each face separately.", default=False),
        },
        ["name", "selection"],
    ),
    group="mesh",
)
def bmesh_inset(name: str, selection: dict[str, Any], thickness: float = 0.05, depth: float = 0.0, individual: bool = False) -> dict[str, Any]:
    """Inset faces, e.g. to make panels, frames, window recesses or a rim before extruding."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        faces = pick_faces(bm, ob, selection)
        require_selection(faces, "faces")
        if individual:
            bmesh.ops.inset_individual(bm, faces=faces, thickness=thickness, depth=depth, use_even_offset=True)
        else:
            bmesh.ops.inset_region(bm, faces=faces, thickness=thickness, depth=depth, use_even_offset=True)
    return {"inset_faces": len(faces), **describe(ob)}


@tool(
    obj(
        {
            "name": string("Mesh object (apply scale first for an even bevel)."),
            "selection": _sel("Edges to bevel; by=sharp picks edges over angle_deg."),
            "width": number("Bevel width in meters.", minimum=0.0, default=0.02),
            "segments": integer("Segments in the rounded profile.", minimum=1, maximum=32, default=2),
            "profile": number("Profile shape; 0.5 is round.", minimum=0.0, maximum=1.0, default=0.5),
            "vertices_only": boolean("Bevel vertices instead of edges.", default=False),
        },
        ["name", "selection"],
    ),
    group="mesh",
)
def bmesh_bevel(
    name: str, selection: dict[str, Any], width: float = 0.02, segments: int = 2, profile: float = 0.5, vertices_only: bool = False
) -> dict[str, Any]:
    """Bevel edges (or vertices) permanently. For an adjustable bevel prefer add_modifier BEVEL."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        if vertices_only:
            geom: list[Any] = pick_verts(bm, ob, selection)
        else:
            edges = pick_edges(bm, ob, selection)
            geom = edges + list({v for e in edges for v in e.verts})
        require_selection(geom, "elements")
        bmesh.ops.bevel(
            bm,
            geom=geom,
            offset=width,
            offset_type="OFFSET",
            segments=segments,
            profile=profile,
            affect="VERTICES" if vertices_only else "EDGES",
            clamp_overlap=True,
        )
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "axis": string("World axis the cuts are perpendicular to.", enum=["X", "Y", "Z"]),
            "cuts": integer("Number of evenly spaced cuts across the bounds.", minimum=1, maximum=64, default=1),
            "positions": array({"type": "number", "minimum": 0.0, "maximum": 1.0}, "Cut positions as 0..1 fractions of the bounds, instead of even spacing."),
        },
        ["name", "axis"],
    ),
    group="mesh",
)
def bmesh_loop_cut(name: str, axis: str, cuts: int = 1, positions: list[float] | None = None) -> dict[str, Any]:
    """Add edge loops by cutting the mesh with planes across one axis. Use to add supporting loops or split
    a shape into sections before extruding parts of it."""
    ob = get_mesh_object(name)
    i = "XYZ".index(axis)
    lo, hi = world_bounds([ob])
    fractions = positions or [(k + 1) / (cuts + 1) for k in range(cuts)]
    inv = ob.matrix_world.inverted()
    normal_local = (ob.matrix_world.to_3x3().transposed() @ Vector([1.0 if k == i else 0.0 for k in range(3)])).normalized()
    with bmesh_of(ob) as bm:
        for t in fractions:
            co = lo.copy()
            co[i] = lo[i] + (hi[i] - lo[i]) * t
            geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
            bmesh.ops.bisect_plane(bm, geom=geom, plane_co=inv @ co, plane_no=normal_local)
    return {"cuts": len(fractions), **describe(ob)}


@tool(
    obj(
        {
            "name": string("Mesh object to cut or combine."),
            "cutter": string("Mesh object used as the boolean operand."),
            "operation": string("Boolean operation.", enum=["DIFFERENCE", "UNION", "INTERSECT"], default="DIFFERENCE"),
            "apply": boolean("Apply right away (destructive). Otherwise a live modifier is kept.", default=True),
            "hide_cutter": boolean("Hide the cutter afterwards.", default=True),
            "delete_cutter": boolean("Delete the cutter after applying.", default=False),
        },
        ["name", "cutter"],
    ),
    group="mesh",
    destructive=True,
)
def bmesh_boolean(
    name: str, cutter: str, operation: str = "DIFFERENCE", apply: bool = True, hide_cutter: bool = True, delete_cutter: bool = False
) -> dict[str, Any]:
    """Cut holes or combine shapes with a boolean against another mesh (exact solver)."""
    ob = get_mesh_object(name)
    cut = get_mesh_object(cutter)
    mod = ob.modifiers.new(f"Bool {cut.name}", "BOOLEAN")
    mod.operation = operation
    mod.object = cut
    if hasattr(mod, "solver"):
        mod.solver = "EXACT"
    if apply:
        from .modifiers import apply_modifier_now

        apply_modifier_now(ob, mod.name)
    if delete_cutter and apply:
        bpy.data.objects.remove(cut, do_unlink=True)
    elif hide_cutter:
        cut.hide_set(True)
        cut.hide_render = True
        cut.display_type = "WIRE"
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Elements to move."),
            "translate": vec3("World-space offset in meters."),
            "scale": vec3("Scale factors about the selection's center (or pivot)."),
            "rotation_deg": vec3("Rotation in degrees about the selection's center (or pivot)."),
            "pivot": vec3("World-space pivot for scale/rotation."),
        },
        ["name", "selection"],
    ),
    group="mesh",
)
def transform_elements(
    name: str,
    selection: dict[str, Any],
    translate: list[float] | None = None,
    scale: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    pivot: list[float] | None = None,
) -> dict[str, Any]:
    """Move, scale or rotate part of a mesh (vertices of the selected elements), e.g. taper the top of a
    shape or lift a row of faces."""
    ob = get_mesh_object(name)
    mw = ob.matrix_world
    inv = mw.inverted()
    with bmesh_of(ob) as bm:
        verts = pick_verts(bm, ob, selection)
        require_selection(verts, "vertices")
        center_w = Vector(pivot) if pivot else mw @ (sum((v.co for v in verts), Vector()) / len(verts))
        center = inv @ center_w
        if scale or rotation_deg:
            space = Matrix.Translation(-center)
            if scale:
                bmesh.ops.scale(bm, vec=Vector(scale), space=space, verts=verts)
            if rotation_deg:
                rot = euler_deg(rotation_deg).to_matrix()
                local_rot = inv.to_3x3() @ rot @ mw.to_3x3()
                bmesh.ops.rotate(bm, cent=center, matrix=local_rot, verts=verts)
        if translate:
            bmesh.ops.translate(bm, vec=inv.to_3x3() @ Vector(translate), verts=verts)
    return {"moved_vertices": len(verts), **describe(ob)}


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Faces/edges to subdivide (default all)."),
            "cuts": integer("Cuts per edge.", minimum=1, maximum=10, default=1),
            "smooth": number("Smoothing of the new vertices.", minimum=0.0, maximum=1.0, default=0.0),
        },
        ["name"],
    ),
    group="mesh",
)
def subdivide(name: str, selection: dict[str, Any] | None = None, cuts: int = 1, smooth: float = 0.0) -> dict[str, Any]:
    """Subdivide edges/faces to add geometry for further shaping."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        edges = pick_edges(bm, ob, selection)
        require_selection(edges, "edges")
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, smooth=smooth, use_grid_fill=True)
    return describe(ob)


@tool(obj({"name": string("Mesh object."), "distance": number("Merge distance in meters.", minimum=0.0, default=0.0001)}, ["name"]), group="mesh")
def merge_by_distance(name: str, distance: float = 0.0001) -> str:
    """Merge vertices closer than a distance (clean up duplicates after joining or mirroring)."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        before = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
        removed = before - len(bm.verts)
    return f"Removed {removed} duplicate vertices from {ob.name}"


@tool(obj({"name": string("Mesh object."), "inside": boolean("Point normals inwards.", default=False)}, ["name"]), group="mesh")
def recalc_normals(name: str, inside: bool = False) -> str:
    """Make face normals consistent (outwards by default). Fixes dark or inside-out shading."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        if inside:
            bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    return f"Recalculated normals on {ob.name}"


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Elements to delete."),
            "kind": string("What to delete.", enum=["VERTS", "EDGES", "FACES", "FACES_ONLY"], default="FACES"),
        },
        ["name", "selection"],
    ),
    group="mesh",
    destructive=True,
)
def delete_elements(name: str, selection: dict[str, Any], kind: str = "FACES") -> dict[str, Any]:
    """Delete vertices, edges or faces, e.g. remove the hidden half before adding a mirror modifier."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        if kind == "VERTS":
            geom: list[Any] = pick_verts(bm, ob, selection)
        elif kind == "EDGES":
            geom = pick_edges(bm, ob, selection)
        else:
            geom = pick_faces(bm, ob, selection)
        require_selection(geom, "elements")
        context = {"VERTS": "VERTS", "EDGES": "EDGES", "FACES": "FACES", "FACES_ONLY": "FACES_ONLY"}[kind]
        bmesh.ops.delete(bm, geom=geom, context=context)
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Boundary edges of two (or more) open loops to bridge; by=boundary is typical."),
            "cuts": integer("Extra loops in the bridge.", minimum=0, maximum=32, default=0),
        },
        ["name", "selection"],
    ),
    group="mesh",
)
def bridge_edge_loops(name: str, selection: dict[str, Any], cuts: int = 0) -> dict[str, Any]:
    """Connect open edge loops with faces (close a gap between two openings)."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        edges = pick_edges(bm, ob, selection)
        require_selection(edges, "edges")
        res = bmesh.ops.bridge_loops(bm, edges=edges)
        if cuts:
            bmesh.ops.subdivide_edges(bm, edges=res["edges"], cuts=cuts, use_grid_fill=True)
    return describe(ob)


@tool(
    obj(
        {
            "name": string("Mesh object."),
            "selection": _sel("Edges to mark; by=sharp is a good start for hard-surface UVs."),
            "clear": boolean("Clear seams instead of marking them.", default=False),
        },
        ["name", "selection"],
    ),
    group="uv",
)
def mark_seams(name: str, selection: dict[str, Any], clear: bool = False) -> str:
    """Mark or clear UV seams on edges before unwrapping with method=unwrap."""
    ob = get_mesh_object(name)
    with bmesh_of(ob) as bm:
        edges = pick_edges(bm, ob, selection)
        for e in edges:
            e.seam = not clear
    return f"{'Cleared' if clear else 'Marked'} {len(edges)} seam edges on {ob.name}"
