"""Export depth / normal / clay control maps from the white-model scene.

Headless usage:
    blender -b assets/demo_white_model.blend -P blender/export_control_maps.py -- \
        --outdir results/raw/A --name A --res 512

Depth pass:
    visible mesh vertices -> 2%/98% camera-distance percentiles -> linear mist
    range -> invert -> near bright / far dark grayscale PNG.

Normal pass:
    Cycles Normal (world space, -1..1) * 0.5 + 0.5 -> RGB PNG.

Clay pass:
    material_override = white matte, render combined -> clay reference PNG.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/raw/demo")
    parser.add_argument("--name", default="demo")
    parser.add_argument("--res", type=int, default=512, choices=(512, 768))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--mist-start", type=float, default=None)
    parser.add_argument("--mist-end", type=float, default=None)
    parser.add_argument("--blend", default=None)
    parser.add_argument("--no-floor-depth", action="store_true")
    return parser.parse_args(argv)


def scene_depth_range(scene, camera):
    """Estimate depth range from mesh vertices visible inside the camera frustum."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    distances = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if obj.get("is_floor"):
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        try:
            for vert in mesh.vertices:
                world_co = obj_eval.matrix_world @ vert.co
                ndc = world_to_camera_view(scene, camera, world_co)
                if ndc.z <= 0 or not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
                    continue
                distances.append((world_co - cam_loc).length)
        finally:
            obj_eval.to_mesh_clear()
    if not distances:
        return 0.1, 100.0
    near = float(np.percentile(distances, 2))
    far = float(np.percentile(distances, 98))
    if far <= near:
        far = near + 1.0
    return near, far


def make_clay_material():
    mat = bpy.data.materials.new("ClayOverride")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.92, 0.92, 0.92, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.88
    spec = bsdf.inputs.get("Specular IOR Level")
    if spec is not None:
        spec.default_value = 0.1
    mat.diffuse_color = (0.92, 0.92, 0.92, 1.0)
    return mat


def invert_node(tree):
    invert = tree.nodes.new("CompositorNodeInvert")
    invert.inputs["Fac"].default_value = 1.0
    if "Invert Color" in invert.inputs:
        invert.inputs["Invert Color"].default_value = True
    return invert


def mix_node(tree, blend_type, constant):
    node = tree.nodes.new("CompositorNodeMixRGB")
    node.blend_type = blend_type
    node.inputs[0].default_value = 1.0
    node.inputs[2].default_value = constant
    return node


def add_output(tree, base_path, slot_name):
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = base_path
    out.file_slots[0].path = slot_name
    out.format.file_format = "PNG"
    out.format.color_mode = "RGB"
    out.format.color_depth = "8"
    out.format.compression = 15
    return out


def render_clay(scene, view_layer, outdir, name, clay_mat):
    view_layer.material_override = clay_mat
    scene.use_nodes = False
    scene.render.filepath = os.path.join(outdir, name + "_clay.png")
    bpy.ops.render.render(write_still=True)
    view_layer.material_override = None


def render_control_maps(scene, view_layer, outdir, name):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    inv = invert_node(tree)
    mul = mix_node(tree, "MULTIPLY", (0.5, 0.5, 0.5, 1.0))
    add = mix_node(tree, "ADD", (0.5, 0.5, 0.5, 1.0))
    depth_out = add_output(tree, outdir, name + "_depth_")
    normal_out = add_output(tree, outdir, name + "_normal_")

    depth_in = inv.inputs["Color"] if "Color" in inv.inputs else inv.inputs["Image"]
    tree.links.new(rl.outputs["Mist"], depth_in)
    tree.links.new(inv.outputs[0], depth_out.inputs[0])
    tree.links.new(rl.outputs["Normal"], mul.inputs[1])
    tree.links.new(mul.outputs[0], add.inputs[1])
    tree.links.new(add.outputs[0], normal_out.inputs[0])

    scene.render.filepath = os.path.join(outdir, name + "_unused.png")
    bpy.ops.render.render(write_still=False)


def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    args = parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if args.blend:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    camera = scene.camera
    camera.data.clip_start = 0.05
    camera.data.clip_end = 200.0
    near, far = scene_depth_range(scene, camera)
    mist_start = args.mist_start if args.mist_start is not None else near
    mist_end = args.mist_end if args.mist_end is not None else far
    if mist_end <= mist_start:
        mist_end = mist_start + max(1.0, far - near)
    scene.world.mist_settings.start = mist_start
    scene.world.mist_settings.depth = mist_end - mist_start
    scene.world.mist_settings.falloff = "LINEAR"

    view_layer.use_pass_mist = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_combined = True

    clay_mat = make_clay_material()
    t0 = datetime.now(timezone.utc)
    render_clay(scene, view_layer, outdir, args.name, clay_mat)
    floor = bpy.data.objects.get("Floor")
    if args.no_floor_depth and floor is not None:
        floor.hide_render = True
    render_control_maps(scene, view_layer, outdir, args.name)
    if args.no_floor_depth and floor is not None:
        floor.hide_render = False
    t1 = datetime.now(timezone.utc)

    depth_raw = latest(os.path.join(outdir, args.name + "_depth_*.png"))
    normal_raw = latest(os.path.join(outdir, args.name + "_normal_*.png"))
    if not depth_raw or not normal_raw:
        raise RuntimeError("compositor outputs missing: depth=%r normal=%r" % (depth_raw, normal_raw))
    depth_png = os.path.join(outdir, args.name + "_depth.png")
    normal_png = os.path.join(outdir, args.name + "_normal.png")
    shutil.copyfile(depth_raw, depth_png)
    shutil.copyfile(normal_raw, normal_png)

    meta = {
        "name": args.name,
        "resolution": args.res,
        "samples": args.samples,
        "camera_lens_mm": camera.data.lens,
        "depth_near_p2": near,
        "depth_far_p98": far,
        "mist_start": mist_start,
        "mist_end": mist_end,
        "mist_falloff": "LINEAR",
        "floor_hidden_for_depth": bool(args.no_floor_depth),
        "view_transform": scene.view_settings.view_transform,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "blender_version": bpy.app.version_string,
        "outputs": {
            "depth": depth_png,
            "normal": normal_png,
            "clay": os.path.join(outdir, args.name + "_clay.png"),
        },
    }
    with open(os.path.join(outdir, args.name + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("EXPORT_OK %s" % json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
