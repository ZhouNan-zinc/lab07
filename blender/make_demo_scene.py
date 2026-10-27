"""Build a white-model demo scene for the depth-control pipeline.

Run headless:
    blender -b -P blender/make_demo_scene.py -- --out assets/demo_white_model.blend --res 512

The scene contains a sphere, a torus, a beveled cube and a cylinder on a
large studio floor, lit by a three-point area-light setup and viewed by a
63 mm camera.  All materials are white/matte so that the clay render and the
depth/normal passes describe geometry only.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def matte_material(name, color=(0.85, 0.85, 0.85, 1.0), roughness=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    spec = bsdf.inputs.get("Specular IOR Level")
    if spec is not None:
        spec.default_value = 0.25
    mat.diffuse_color = color
    return mat


def build_scene(res):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    # Standard transform keeps the data passes (mist/normal) predictable.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.35
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.55, 0.60, 0.68, 1.0)
    bg.inputs[1].default_value = 0.22
    world.color = (0.55, 0.60, 0.68)

    body_mat = matte_material("WhiteMatte", (0.88, 0.88, 0.88, 1.0), 0.52)
    floor_mat = matte_material("FloorMatte", (0.62, 0.63, 0.66, 1.0), 0.72)
    clay_mat = matte_material("ClayWhite", (0.92, 0.92, 0.92, 1.0), 0.85)

    # A subdivided floor keeps enough visible vertices for the percentile
    # depth-range estimate used by export_control_maps.py.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=48, y_subdivisions=48, size=24, location=(0, 0, 0)
    )
    floor = bpy.context.object
    floor.name = "Floor"
    floor["is_floor"] = True
    floor.data.materials.append(floor_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.62, segments=64, ring_count=32, location=(-0.95, 0.05, 0.62)
    )
    sphere = bpy.context.object
    sphere.name = "Sphere"
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.52,
        minor_radius=0.19,
        major_segments=64,
        minor_segments=24,
        location=(0.28, 0.12, 0.78),
        rotation=(math.radians(64), 0.0, math.radians(-18)),
    )
    torus = bpy.context.object
    torus.name = "Torus"
    bpy.ops.object.shade_smooth()
    torus.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cube_add(size=0.86, location=(1.18, -0.30, 0.53))
    cube = bpy.context.object
    cube.name = "BeveledCube"
    cube.rotation_euler = (0.0, 0.0, math.radians(-16))
    bevel = cube.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.075
    bevel.segments = 4
    cube.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.30, depth=1.24, vertices=64, location=(-0.12, -1.22, 0.63)
    )
    cylinder = bpy.context.object
    cylinder.name = "Cylinder"
    cylinder.rotation_euler = (0.0, math.radians(5), 0.0)
    bpy.ops.object.shade_smooth()
    cylinder.data.materials.append(body_mat)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 63
    camera_data.sensor_width = 36
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (4.8, -6.0, 2.8)
    look_at(camera, (0.0, 0.0, 0.55))
    scene.camera = camera

    add_area_light("Key", (3.0, -3.2, 4.2), 130, 3.5, (0, 0, 0.5))
    add_area_light("Fill", (-3.6, -2.8, 2.2), 35, 5.0, (0, 0, 0.6))
    add_area_light("Rim", (0.5, 3.5, 3.2), 90, 2.2, (0, 0, 0.8))

    # Store the clay override in the file for export_control_maps.py.
    scene["clay_material"] = clay_mat.name
    scene["demo_resolution"] = res
    return scene


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="assets/demo_white_model.blend")
    parser.add_argument("--res", type=int, default=512, choices=(512, 768))
    args = parser.parse_args(argv)

    scene = build_scene(args.res)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(
        "SCENE_OK path=%s res=%d objects=%d"
        % (out_path, args.res, len(scene.objects))
    )


if __name__ == "__main__":
    main()
