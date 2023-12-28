import argparse
import colorsys
import io
import math
import os
import random
import sys
from contextlib import redirect_stdout

import bmesh
import bpy
import cv2
import numpy as np
import pandas as pd
from numpy.random import uniform
from tqdm import tqdm

C = bpy.context
D = bpy.data
O = bpy.ops

C.scene.render.engine = "CYCLES"
C.scene.render.film_transparent = True
C.scene.render.image_settings.file_format = "PNG"
C.scene.render.image_settings.color_depth = "8"

C.view_layer.cycles.use_denoising = True

D.scenes["Scene"].render.resolution_x = 1200  # 9  600, 900, 1200
D.scenes["Scene"].render.resolution_y = 1600  # 12 800, 1200, 1600
D.scenes["Scene"].render.resolution_percentage = 100
D.scenes["Scene"].render.dither_intensity = 1
D.scenes["Scene"].cycles.max_bounces = 12  # default is 12
D.scenes["Scene"].cycles.samples = 128  # default is 128

LOCATION_X = 0
LOCATION_Y = 0
LOCATION_Z = 1.8

CAMERA_LCATION_Z = 1.5

ROTATION_X = math.pi * 90 / 180
ROTATION_Y = 0
ROTATION_Z = 0

SHIT_MIN = -0.1
ROTATION_MIN = math.pi * 0 / 180
ROTATION_MAX = math.pi * 90 / 180

LIGHT_MIN, LIGHT_MAX = 0.01, 0.05
SCALE_MIN, SCALE_MAX = 0.2, 0.5
SHIFT_MAX, SHIFT_LIGHT = 0.1, 0.3
LIGHT_LOC_Z = 2.2

# light
LIGHT_COLOR = ["#FFC58F", "#FFF1E0", "#FFFFFF", "#C9E2FF"]  # normal light
NUM_LIGHT_COLORS = len(LIGHT_COLOR)

# path
object_paths = list(pd.read_csv("../materials/csv/object.csv")["path"])
paper_mesh_paths = list(pd.read_csv("../materials/csv/paper_mesh.csv")["path"])
doc_paths = list(pd.read_csv("../materials/csv/doc.csv")["path"])
normal_map_paths = list(pd.read_csv("../materials/csv/normal_map.csv")["path"])
envmap_indoor_paths = list(pd.read_csv("../materials/csv/envmap_indoor.csv")["path"])
envmap_outdoor_paths = list(pd.read_csv("../materials/csv/envmap_outdoor.csv")["path"])

NUM_OBJECTS = len(object_paths)
NUM_DOCUMENTS = len(doc_paths)
NUM_NORMALS = len(normal_map_paths)
NUM_INDOOR_ENVMAPS = len(envmap_indoor_paths)
NUM_OUTDOOR_ENVMAPS = len(envmap_outdoor_paths)
NUM_PLANES = len(paper_mesh_paths)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        rendering document images for shadow removal.
        """
    )
    parser.add_argument("--root_path", type=str, default="../", help="save path")
    parser.add_argument("--save_path", type=str, default="output", help="save path")
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Add --rendering option if you want to render document images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument("-gid", "--gpu_id", type=int, default=None)

    if "--" not in sys.argv:
        args = parser.parse_args()
    else:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        args, _ = parser.parse_known_args(argv)
    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


class DocumentRenderer:
    def __init__(self, root_path="../", save_path="output"):
        self.root_path = root_path
        self.save_path = save_path

    def set_gpu_or_cpu(self, device="GPU"):
        C.scene.cycles.device = device
        prefs = C.preferences
        cprefs = prefs.addons["cycles"].preferences

        for device in cprefs.devices:
            device.use = True

    def set_shadow_catcher(self, catcher=True):
        self.plane.cycles.is_shadow_catcher = catcher

    def hexToRGBA(self, hex):
        gamma = 2.2
        hex = hex.lstrip("#")
        lv = len(hex)
        fin = list(int(hex[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        r = pow(fin[0] / 255, gamma)
        g = pow(fin[1] / 255, gamma)
        b = pow(fin[2] / 255, gamma)
        fin.clear()
        fin.append(r)
        fin.append(g)
        fin.append(b)
        fin.append(1.0)
        return tuple(fin)

    def getRandomColor(self):
        h = np.random.uniform(0, 1.0)
        s = np.random.uniform(0, 0.3)  # lower is close to white
        t = list(colorsys.hsv_to_rgb(h, s, 1.0)) + [1.0]
        return t

    def reset_scene(self):
        C.scene.frame_set(0)
        O.object.select_all(action="SELECT")
        O.object.delete(use_global=False)

        self.data = [D.meshes, D.materials, D.textures, D.images, D.worlds, D.lights]
        for d in self.data:
            for block in d:
                if block.name == "Render Result":
                    continue
                d.remove(block)

    def create_plane(self, rotate=True):
        if rotate is True:
            z_rot_euler = math.pi / 24 * uniform(-1.0, 1.0)
        else:
            z_rot_euler = 0
        O.mesh.primitive_plane_add(
            location=(0, 0, 0),
            rotation=(
                0,
                0,
                z_rot_euler,
            ),  # force plane horizontal
            enter_editmode=False,
        )

        C.object.dimensions = (9 / 10, 12 / 10, 0)  # 9:12

        self.plane = D.objects["Plane"]
        self.planeMat = D.materials.new(name="PlaneMaterial")  # material of plane
        self.planeMat.use_nodes = True
        self.planeMatNodes = self.planeMat.node_tree.nodes  # node of plane

        self.plane.data.materials.append(self.planeMat)

    def import_plane(self):
        idx = np.random.randint(0, NUM_PLANES)
        self.imported_mesh = O.import_scene.obj(filepath=paper_mesh_paths[idx])

        self.plane = C.selected_objects[0]
        self.plane.name = "Plane"

        random_scale = uniform(0.5, 0.6)  # for Doc3D obj
        self.plane.scale = (
            random_scale,
            random_scale,
            random_scale,
        )

        self.plane.location = (0, 0, 0)
        x_rot_euler = math.pi / 24 * uniform(-1.0, 1.0)
        y_rot_euler = math.pi / 24 * uniform(-1.0, 1.0)
        z_rot_euler = math.pi / 24 * uniform(-1.0, 1.0)
        self.plane.rotation_euler = (
            math.pi + x_rot_euler,
            math.pi + y_rot_euler,
            -math.pi / 2 + z_rot_euler,
        )

        self.planeMat = D.materials.new(name="PlaneMaterial")  # material of paper
        self.planeMat.use_nodes = True
        self.planeMatNodes = self.planeMat.node_tree.nodes  # node of paper

        # add planeMat to materials
        self.plane.data.materials[0] = self.planeMat

    def subdivide(self, div):
        O.object.select_all(action="DESELECT")
        O.object.mode_set(mode="EDIT")
        plane_mesh = bmesh.from_edit_mesh(self.plane.data)

        for i in range(div):
            for v in plane_mesh.verts:
                v.select_set(True)
            O.mesh.subdivide()

        O.object.mode_set(mode="OBJECT")
        O.object.select_all(action="DESELECT")

    def set_document(self):
        self.planeTex = self.planeMatNodes.new(type="ShaderNodeTexImage")  # tex image
        self.planeMat.node_tree.links.new(
            self.planeMatNodes["Principled BSDF"].inputs["Base Color"],
            self.planeTex.outputs["Color"],
        )

    def load_document(self):
        # set doc
        self.plane_roughness = random.uniform(0.4, 0.6)  # roughness of BSDF of plane
        self.planeMatNodes["Principled BSDF"].inputs[
            7
        ].default_value = self.plane_roughness

        self.doc_idx = np.random.randint(0, NUM_DOCUMENTS)
        self.planeTex.image = D.images.load(doc_paths[self.doc_idx])

    def link_document(self):
        self.planeMat.node_tree.links.new(
            self.planeMatNodes["Principled BSDF"].inputs["Base Color"],
            self.planeTex.outputs["Color"],
        )

    def unlink_document(self):
        self.planeMat.node_tree.links.remove(
            self.planeMatNodes["Principled BSDF"].inputs["Base Color"].links[0]
        )

    def init_normalmap(self):
        # normal map
        self.normalTex = self.planeMatNodes.new(type="ShaderNodeTexImage")
        self.normalMap = self.planeMatNodes.new(type="ShaderNodeNormalMap")

        # link nomal image to normal map
        self.planeMat.node_tree.links.new(
            self.planeMatNodes["Normal Map"].inputs["Color"],
            self.normalTex.outputs["Color"],
        )

        # link normal map to BSDF
        self.planeMat.node_tree.links.new(
            self.planeMatNodes["Principled BSDF"].inputs["Normal"],
            self.normalMap.outputs["Normal"],
        )

    def load_normalmap(self):
        # import normal map image
        idx = np.random.randint(0, NUM_NORMALS)
        self.normalTex.image = D.images.load(normal_map_paths[idx])

        self.planeMatNodes["Normal Map"].inputs[0].default_value = random.uniform(
            0, 1
        )  # 0.5  # strength or normal map, 0 to 10

    def link_normalmap(self):
        self.planeMat.node_tree.links.new(
            self.planeMatNodes["Principled BSDF"].inputs["Normal"],
            self.normalMap.outputs["Normal"],
        )

    def unlink_normalmap(self):
        self.planeMat.node_tree.links.remove(
            self.planeMatNodes["Principled BSDF"].inputs["Normal"].links[0]
        )

    def set_plane_roughness_one(self):
        self.planeMatNodes["Principled BSDF"].inputs[7].default_value = 1

    def set_plane_roughness_org(self):
        self.planeMatNodes["Principled BSDF"].inputs[
            7
        ].default_value = self.plane_roughness

    def init_world(self):
        self.world = D.worlds.new(name="World")
        C.scene.world = self.world
        self.world.use_nodes = True
        self.worldNodes = self.world.node_tree.nodes

        self.bg = self.world.node_tree.nodes["Background"]
        self.env = self.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        self.world.node_tree.links.new(
            self.env.outputs["Color"], self.bg.inputs["Color"]
        )

    def link_envmap(self):
        self.world.node_tree.links.new(
            self.env.outputs["Color"], self.bg.inputs["Color"]
        )

    def unlink_envmap(self):
        self.world.node_tree.links.remove(self.bg.inputs["Color"].links[0])

    def set_constant_envmap(self, color=None):
        if color is None:
            self.env_color = self.getRandomColor()
        else:
            self.env_color = self.hexToRGBA(color)

        self.const_env = self.worldNodes.new(type="ShaderNodeRGB")
        self.world.node_tree.nodes["RGB"].outputs[0].default_value = self.env_color
        self.world.node_tree.links.new(
            self.const_env.outputs["Color"], self.bg.inputs["Color"]
        )

    def set_envmap_strength(self, strength=1):
        self.bg.inputs[1].default_value = strength

    def load_envmap(self):
        if np.random.random() > 0.5:
            self.env_idx = np.random.randint(0, NUM_INDOOR_ENVMAPS)
            self.env.image = D.images.load(envmap_indoor_paths[self.env_idx])
        else:
            self.env_idx = np.random.randint(0, NUM_OUTDOOR_ENVMAPS)
            self.env.image = D.images.load(envmap_outdoor_paths[self.env_idx])

        envmap_strength = random.uniform(0.01, 0.5)
        self.bg.inputs[1].default_value = envmap_strength

    def create_light_source(self):
        light_type = "Point"
        num_of_lights = np.random.randint(2, 4)
        single = np.random.randint(0, 2)
        if single == 1:  # single:multiple = 1:1
            num_of_lights = 1
        self.lights = []
        O.object.light_add(
            type=light_type.upper(),
            location=(0, 0, LIGHT_LOC_Z),
            radius=uniform(LIGHT_MIN, LIGHT_MAX),
        )  # default radius is 1.0
        D.lights[light_type].use_nodes = True
        # C.active_object.data.energy = 50
        self.lights.append(D.lights[light_type])

        for i in range(1, num_of_lights):
            shift = np.append(uniform(-SHIFT_LIGHT, SHIFT_LIGHT, size=(2,)), [0])
            O.object.light_add(
                type=light_type.upper(),
                location=([0, 0, 2.2] + shift),
                radius=uniform(LIGHT_MIN, LIGHT_MAX),
            )  # default radius is 1.0
            D.lights[light_type + f".00{str(i)}"].use_nodes = True
            # C.active_object.data.energy = 50

            self.lights.append(D.lights[light_type + f".00{str(i)}"])

        self.set_power_pof_light(power=50)

        # color of light
        # idx = np.random.randint(0, NUM_LIGHT_COLORS)
        # spot_color = LIGHT_COLOR[idx]

        self.set_light_color()  # or self.set_light_color(LIGHT_COLOR[idx])
        self.set_light_strength(strength_range=(1, 1.25))

    def set_light_color(self, spot_color=None):
        if spot_color is None:
            self.spot_color = self.getRandomColor()
        else:
            self.spot_color = self.hexToRGBA(spot_color)

        for light in self.lights:
            light.node_tree.nodes["Emission"].inputs[0].default_value = self.spot_color

    def set_light_strength(self, spot_strength=None, strength_range=(1, 1.25)):
        if spot_strength is None:
            self.spot_strength = random.uniform(*strength_range)
        else:
            self.spot_strength = spot_strength

        for light in self.lights:
            light.node_tree.nodes["Emission"].inputs[
                1
            ].default_value = self.spot_strength

    def set_power_pof_light(self, power=50):
        for light in self.lights:
            light.energy = power

    def set_camera(self):
        self.camera = D.objects.new("Camera", D.cameras.new("Camera"))
        self.camera.location = (0, 0, CAMERA_LCATION_Z)
        D.scenes[0].collection.objects.link(self.camera)
        C.scene.camera = self.camera

    def set_object(self):
        idx = np.random.randint(0, NUM_OBJECTS)
        self.imported_obj = O.import_scene.obj(filepath=object_paths[idx])

        self.obj = C.selected_objects[0]
        coord_x = random.uniform(-SHIFT_MAX, SHIFT_MAX)
        coord_y = random.uniform(-SHIFT_MAX, SHIFT_MAX)
        rotate = math.pi * uniform(-1.0, 1.0, size=(3,))
        self.obj.scale = uniform(SCALE_MIN, SCALE_MAX, size=(1,)).repeat(3)
        self.obj.location = (coord_x, coord_y, LOCATION_Z)
        self.obj.rotation_euler = rotate

    def object_visibility(self, hide=True):
        self.obj.hide_viewport = hide
        self.obj.hide_render = hide

    def add_subdivision_surface(self, div):
        self.plane.select_set(True)
        C.view_layer.objects.active = self.plane
        O.object.modifier_add(type="SUBSURF")
        C.object.modifiers["Subdivision"].render_levels = div
        C.object.modifiers["Subdivision"].levels = div

    def create_cube(self):
        O.object.mode_set(mode="OBJECT")
        O.object.select_all(action="DESELECT")

        O.mesh.primitive_cube_add(size=10, enter_editmode=False, location=(0, 0, 0))
        C.object.dimensions = (9 / 8, 12 / 8, 0.3)
        self.cube = D.objects["Cube"]

        C.view_layer.objects.active = self.cube
        self.cube.select_set(True)

        O.object.modifier_add(type="SOLIDIFY")
        C.object.modifiers["Solidify"].thickness = 0.2

        O.object.modifier_add(type="COLLISION")

        C.scene.frame_set(0)
        self.cube.keyframe_insert(data_path="scale", frame=0)
        C.scene.frame_set(100)
        vx = np.random.uniform(0.825, 0.85)
        vy = np.random.uniform(0.825, 0.85)
        O.transform.resize(value=(vx, vy, 1))
        self.cube.keyframe_insert(data_path="scale", frame=100)
        C.scene.frame_set(0)

        O.object.mode_set(mode="OBJECT")
        O.object.select_all(action="DESELECT")

    def hide_cube(self, hide=True):
        O.object.select_all(action="DESELECT")

        C.view_layer.objects.active = self.cube
        self.cube.select_set(True)
        C.object.hide_viewport = hide
        C.object.hide_render = hide

        O.object.select_all(action="DESELECT")

    def make_plane_paper(self):
        O.object.mode_set(mode="OBJECT")
        O.object.select_all(action="DESELECT")

        C.view_layer.objects.active = self.plane
        self.plane.select_set(True)

        O.object.modifier_add(type="CLOTH")

        # make it paper like
        C.object.modifiers["Cloth"].settings.effector_weights.gravity = 0
        C.object.modifiers["Cloth"].settings.quality = 15
        C.object.modifiers["Cloth"].settings.mass = 0.4
        C.object.modifiers["Cloth"].settings.tension_stiffness = 80
        C.object.modifiers["Cloth"].settings.compression_stiffness = 80
        C.object.modifiers["Cloth"].settings.shear_stiffness = 80
        C.object.modifiers["Cloth"].settings.bending_stiffness = 10
        C.object.modifiers["Cloth"].settings.tension_damping = 25
        C.object.modifiers["Cloth"].settings.compression_damping = 25
        C.object.modifiers["Cloth"].settings.shear_damping = 25
        C.object.modifiers["Cloth"].settings.bending_damping = 1
        C.object.modifiers["Cloth"].collision_settings.use_self_collision = True
        C.object.modifiers["Cloth"].collision_settings.self_friction = 15

        O.object.select_all(action="DESELECT")

    def bake(self):
        O.ptcache.bake_all(bake=True)

    def bending_plane(self, hide=True):
        self.make_plane_paper()
        self.create_cube()
        C.scene.frame_current = 100
        D.scenes["Scene"].frame_end = 100
        self.bake()
        self.hide_cube(hide)

    def rendering(self, file_name="test"):
        C.scene.render.filepath = os.path.join(
            self.root_path, self.save_path, file_name
        )
        O.render.render(write_still=True)

    def check_area(self, file_name):
        img = cv2.imread(
            os.path.join(self.root_path, self.save_path, file_name),
            cv2.IMREAD_UNCHANGED,
        )
        h, w, c = img.shape
        return np.sum(1 - img[:, :, 3] / 255) > (h * w) / 10

    def init_setting(self, bending):
        self.reset_scene()
        self.init_world()
        self.load_envmap()
        self.set_camera()
        self.create_light_source()

        if bending is True:
            if np.random.random() >= 0.25:
                self.import_plane()
                simulate = False
            else:
                self.create_plane(rotate=True)
                simulate = True
        else:
            self.create_plane(rotate=True)
            simulate = False

        self.set_document()
        self.init_normalmap()
        self.load_normalmap()

        # bending document
        if simulate:
            div = np.random.randint(4, 5)
            self.subdivide(div)
            self.bending_plane(hide=True)
            self.add_subdivision_surface(9 - div)

        elif bending is True:
            self.add_subdivision_surface(2)

        Renderer.set_object()
        Renderer.load_document()


def enable_gpus(device_type, use_cpus=False, gpu_id=None):
    preferences = C.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for i, device in enumerate(devices):
        if i == gpu_id or gpu_id is None:
            device.use = True
            activated_gpus.append(device.name)
        else:
            device.use = False

        print(device.name, device.use)

    cycles_preferences.compute_device_type = device_type
    C.scene.cycles.device = "GPU"


if __name__ == "__main__":
    args = get_arguments()
    set_seed(args.seed)
    root_path = args.root_path
    save_path = args.save_path
    rendering = args.rendering
    gpu_id = args.gpu_id

    enable_gpus(device_type="CUDA", gpu_id=gpu_id)

    rendered_types = [
        "shadowfree_image",
        "shadow_image_org",
        "shadow_mask_raw",
        "background_image",
    ]

    if not os.path.exists(os.path.join(root_path, save_path)):
        os.mkdir(os.path.join(root_path, save_path))
        for types in rendered_types:
            os.mkdir(os.path.join(root_path, save_path, types))

    Renderer = DocumentRenderer(root_path=root_path, save_path=save_path)
    Renderer.set_gpu_or_cpu(device="GPU")
    output = io.StringIO()

    for i in tqdm(range(0, 5000)):

        with redirect_stdout(output):
            if i % 2 == 0:
                bending = True
            else:
                bending = False

            Renderer.init_setting(bending)

            # shadowfree
            while True:
                Renderer.object_visibility(hide=True)
                Renderer.rendering(file_name=f"shadowfree_image/{str(i).zfill(5)}.png")
                if (
                    Renderer.check_area(
                        file_name=f"shadowfree_image/{str(i).zfill(5)}.png"
                    )
                    is True
                ):
                    Renderer.init_setting(bending)
                    continue
                break

            # shadow
            Renderer.object_visibility(hide=False)
            Renderer.rendering(file_name=f"shadow_image_org/{str(i).zfill(5)}.png")

            # bg image
            Renderer.unlink_document()
            Renderer.object_visibility(hide=True)
            Renderer.rendering(file_name=f"background_image/{str(i).zfill(5)}.png")

            # shadow matte
            Renderer.object_visibility(hide=False)
            Renderer.set_shadow_catcher(True)
            if rendering is True:
                Renderer.rendering(file_name=f"shadow_mask_raw/{str(i).zfill(5)}.png")
            Renderer.set_shadow_catcher(False)
