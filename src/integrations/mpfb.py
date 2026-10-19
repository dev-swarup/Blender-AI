from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

import bpy

from ..tools.common import describe, get_object
from ..tools.registry import ToolError, boolean, integer, number, obj, string, tool, vec3
from .registry import enabled_module

RIGS = ["default", "default_no_toes", "game_engine", "game_engine_with_breast", "mixamo", "mixamo_unity", "cmu_mb", "rigify.human", "rigify.human_toes"]
MACROS = ("gender", "age", "muscle", "weight", "height", "proportions", "cupsize", "firmness")


def _services() -> ModuleType:
    module = enabled_module("mpfb")
    if module is None:
        raise ToolError("MPFB isn't enabled. Install it from extensions.blender.org and enable it.")
    try:
        return importlib.import_module(f"{module}.services")
    except ImportError as exc:
        raise ToolError(f"Couldn't load MPFB services ({exc}); this MPFB version may be unsupported.") from exc


def _humans(services: ModuleType) -> Any:
    return importlib.import_module(services.__name__ + ".humanservice").HumanService


def _call(what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"MPFB failed to {what}: {type(exc).__name__}: {exc}") from exc


def _sections(services: ModuleType) -> dict[str, dict[str, Any]]:
    sections = getattr(services, "ASSET_LIBRARY_SECTIONS", None) or getattr(services.assetservice, "ASSET_LIBRARY_SECTIONS", [])
    return {s["asset_subdir"]: s for s in sections if s.get("asset_type") in ("mhclo", "mhmat", "proxy")}


def _basemesh(name: str) -> bpy.types.Object:
    ob = get_object(name)
    if ob.type == "ARMATURE":
        child = next((c for c in ob.children if c.type == "MESH"), None)
        if child is None:
            raise ToolError(f"{name!r} is a rig with no body mesh.")
        ob = child
    if ob.type != "MESH":
        raise ToolError(f"{name!r} is not an MPFB human.")
    return ob


@tool(
    obj(
        {
            "name": string("Name for the human."),
            "gender": number("0 = female, 1 = male.", minimum=0, maximum=1, default=0.5),
            "age": number("0 = 1 year, 0.1875 = 11, 0.5 = 25, 1 = 90 years.", minimum=0, maximum=1, default=0.5),
            "muscle": number("Muscle tone.", minimum=0, maximum=1, default=0.5),
            "weight": number("Body weight.", minimum=0, maximum=1, default=0.5),
            "height": number("Height.", minimum=0, maximum=1, default=0.5),
            "proportions": number("Idealised (1) vs average (0) proportions.", minimum=0, maximum=1, default=0.5),
            "african": number("Ethnic blend weight.", minimum=0, maximum=1),
            "asian": number("Ethnic blend weight.", minimum=0, maximum=1),
            "caucasian": number("Ethnic blend weight.", minimum=0, maximum=1),
            "rig": string("Add a rig.", enum=["none", *RIGS], default="none"),
            "skin": string("Skin material name from mpfb_list_assets(kind='skins'), if asset packs are installed."),
            "location": vec3("Where the feet go."),
        }
    ),
    group="integrations",
    requires="mpfb",
)
def mpfb_create_human(
    name: str | None = None,
    gender: float = 0.5,
    age: float = 0.5,
    muscle: float = 0.5,
    weight: float = 0.5,
    height: float = 0.5,
    proportions: float = 0.5,
    african: float | None = None,
    asian: float | None = None,
    caucasian: float | None = None,
    rig: str = "none",
    skin: str | None = None,
    location: list[float] | None = None,
) -> dict[str, Any]:
    """Create a realistic human body with MPFB from macro sliders, optionally rigged and skinned. Use this
    for people instead of modelling them by hand."""
    services = _services()
    targets = services.TargetService
    humans = _humans(services)
    macro = _call("read default macros", targets.get_default_macro_info_dict)
    values = {"gender": gender, "age": age, "muscle": muscle, "weight": weight, "height": height, "proportions": proportions}
    for key, value in values.items():
        if key in macro:
            macro[key] = value
    races = {k: v for k, v in (("african", african), ("asian", asian), ("caucasian", caucasian)) if v is not None}
    if races and "race" in macro:
        total = sum(races.values()) or 1.0
        macro["race"] = {k: races.get(k, 0.0) / total for k in macro["race"]}
    body = _call("create the human", humans.create_human, macro_detail_dict=macro)
    if name:
        body.name = name
    top = body
    if rig != "none":
        armature = _call(f"add the {rig} rig", humans.add_builtin_rig, body, rig)
        if armature is None:
            raise ToolError(f"MPFB has no rig named {rig!r} in this version.")
        top = armature
    if skin:
        mpfb_add_asset(body.name, "skins", skin)
    if location:
        top.location = location
    return {"human": body.name, "rig": top.name if top is not body else None, **describe(top)}


@tool(
    obj({"human": string("The MPFB body mesh (or its rig)."), "rig": string("Rig to add.", enum=RIGS, default="default")}, ["human"]),
    group="integrations",
    requires="mpfb",
)
def mpfb_add_rig(human: str, rig: str = "default") -> dict[str, Any]:
    """Add an MPFB rig to a human made with MPFB, with weights already fitted. 'rigify.human' gives a
    Rigify metarig to generate; 'game_engine' or 'mixamo' suit export."""
    services = _services()
    humans = _humans(services)
    body = _basemesh(human)
    armature = _call(f"add the {rig} rig", humans.add_builtin_rig, body, rig)
    if armature is None:
        raise ToolError(f"MPFB has no rig named {rig!r} in this version.")
    return {"rig": armature.name, "bones": len(armature.data.bones)}


@tool(
    obj(
        {
            "kind": string("Asset library to list.", enum=["skins", "eyes", "eyebrows", "eyelashes", "hair", "teeth", "tongue", "clothes", "proxymeshes"]),
            "limit": integer("Maximum names to return.", minimum=1, maximum=500, default=100),
        },
        ["kind"],
    ),
    group="integrations",
    mutates=False,
    requires="mpfb",
)
def mpfb_list_assets(kind: str, limit: int = 100) -> dict[str, Any]:
    """List MPFB asset names (skins, hair, clothes, eyes...) available to mpfb_add_asset. Empty lists mean
    the MakeHuman asset packs aren't installed."""
    services = _services()
    assets = services.AssetService
    if kind == "skins":
        paths = _call("list skins", assets.list_mhmat_assets, kind)
    elif kind == "proxymeshes":
        paths = _call("list proxies", assets.list_proxy_assets, kind)
    else:
        paths = _call(f"list {kind}", assets.list_mhclo_assets, kind)
    found = sorted({os.path.splitext(os.path.basename(str(p)))[0] for p in paths})
    return {"kind": kind, "count": len(found), "names": found[:limit]}


@tool(
    obj(
        {
            "human": string("The MPFB body mesh (or its rig)."),
            "kind": string("Asset library.", enum=["skins", "eyes", "eyebrows", "eyelashes", "hair", "teeth", "tongue", "clothes", "proxymeshes"]),
            "asset": string("Asset name from mpfb_list_assets."),
            "fit_to_rig": boolean("Rig the asset to the human's armature if it has one.", default=True),
        },
        ["human", "kind", "asset"],
    ),
    group="integrations",
    requires="mpfb",
)
def mpfb_add_asset(human: str, kind: str, asset: str, fit_to_rig: bool = True) -> str:
    """Put an MPFB asset on a human: a skin material, eyes, hair, eyebrows, clothes or a proxy topology."""
    services = _services()
    assets = services.AssetService
    humans = _humans(services)
    body = _basemesh(human)
    if kind == "skins":
        candidates = _call("list skins", assets.list_mhmat_assets, "skins")
    elif kind == "proxymeshes":
        candidates = _call("list proxies", assets.list_proxy_assets, kind)
    else:
        candidates = _call(f"list {kind}", assets.list_mhclo_assets, kind)
    match = next((str(p) for p in candidates if os.path.splitext(os.path.basename(str(p)))[0].lower() == asset.lower()), None)
    if match is None:
        names = sorted(os.path.splitext(os.path.basename(str(p)))[0] for p in candidates)[:30]
        raise ToolError(f"No {kind} asset {asset!r}. Some available: {', '.join(names) or 'none (asset packs not installed)'}")
    if kind == "skins":
        _call("set the skin", humans.set_character_skin, match, body)
        return f"Skin {asset!r} applied to {body.name}"
    section = _sections(services).get(kind, {})
    object_type = section.get("object_type", kind.title())
    added = _call(f"add {asset}", humans.add_mhclo_asset, match, body, asset_type=object_type, set_up_rigging=fit_to_rig)
    return f"Added {kind} {asset!r} to {body.name}" + (f" as {added.name}" if added is not None and hasattr(added, "name") else "")
