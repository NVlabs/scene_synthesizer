# Standard Library
import os

# Third Party
import networkx as nx
import numpy as np
import trimesh
import trimesh.transformations as tra

# Local Folder
from ..usd_import import get_meters_per_unit, get_stage
from ..utils import log
from . import usd_export
from ..constants import EDGE_KEY_METADATA

def export_usd(
    scene,
    fname,
    file_type="usd",
    folder=None,
    texture_dir="textures",
    export_materials=True,
    separate_assets=False,
    use_absolute_usd_paths=False,
    write_usd_object_files=True,
    include_light_nodes=False,
    ignore_layers=None,
    up_axis="Z",
    meters_per_unit=1.0,
    kilograms_per_unit=1.0,
    write_attribute_attached_state=False,
):
    """Export scene to USD file or return USD data.

    Args:
        scene (scene_synthesizer.Scene): Scene description.
        fname (str): The USD filename or None. If None the USD stage is returned.
        file_type (str, optional): Either 'usd', 'usda', or 'usdc'. Defaults to 'usd'.
        folder (str, optional): Only used if fname is None. The folder in which to export the USD(s). Affects the location of textures. Defaults to None.
        texture_dir (str, optional): Directory where texture files will be written (relative to USD main file). Defaults to "textures".
        export_materials (bool, optional): Whether to write out texture files. Defaults to True.
        separate_assets (bool, optional): If True, each asset in the scene will be exported to a separate USD file, named according to its object identifier. Defaults to False.
        use_absolute_usd_paths (bool, optional): If set to True will convert all referenced USD paths to absolute ones. Defaults to False.
        write_usd_object_files (bool, optional): Write USD files instead of referencing existing ones. Defaults to True.
        include_light_nodes (bool, optional): Whether to include a link for the light nodes. Defaults to False.
        ignore_layers (bool, optional): Whether to ignore scene layers. If None, scenes with one layer or less will ignore the layers. Defaults to None.
        up_axis (str, optional): Either 'Y' or 'Z'. Defaults to "Z".
        meters_per_unit (float, optional): Meters per unit. Defaults to 1.0.
        kilograms_per_unit (float, optional): Kilograms per unit. Defaults to 1.0.
        write_attribute_attached_state (bool, optional): Will write attribute 'AttachedState' for objects whose parent is not the world. Defaults to False.

    Returns:
        Usd.Stage (optional): If fname is None, will return the USD stage (created in memory).
        dict[str, Usd.Stage] (optional): If fname=None and separate_assets=True will return the additional assets as USD stages in a dictionary.
    """
    if fname is not None and folder is not None:
        log.warn(f"USD export: folder={folder} will be ignored since file name is already specified.")
    
    if fname is not None:
        folder = os.path.dirname(fname)
        fname = os.path.basename(fname)

    if folder is None or len(folder) == 0:
        folder = "."

    # Remember current scene configuration to re-apply it after export
    current_configuration = scene.get_configuration()
    position_from_joint = dict(zip(scene.get_joint_names(), current_configuration))
    scene.zero_configurations()

    if ignore_layers is None:
        num_layers = len(
            set(
                [
                    scene._scene.geometry[g].metadata["layer"]
                    for g in scene._scene.geometry
                    if "layer" in scene._scene.geometry[g].metadata
                ]
            )
        )
        ignore_layers = num_layers < 2


    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, texture_dir), exist_ok=True)

    if fname is None:
        stage = usd_export.create_stage_in_memory(up_axis=up_axis, meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit)
    else:
        fname = os.path.join(folder, fname)
        stage = usd_export.create_stage(fname, up_axis=up_axis, meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit)
    stage_meters_per_unit = get_meters_per_unit(stage)
    referenced_stages = {}
    referenced_stage_fnames = {}
    referenced_stage_transforms = {}

    if not write_usd_object_files:
        for obj_id in scene.get_object_names():
            # ensure that all nodes of an object do come from the same file
            geom_names = scene.get_geometry_names(obj_id=obj_id)
            file_paths = [
                scene.geometry[scene.graph[g][1]].metadata["file_path"]
                for g in geom_names
                if "file_path" in scene.geometry[scene.graph[g][1]].metadata
            ]
            if (
                len(set(file_paths)) == 1
                and len(file_paths) > 0
                and (
                    file_paths[0].lower().endswith(".usd")
                    or file_paths[0].lower().endswith(".usda")
                    or file_paths[0].lower().endswith(".usdc")
                    or file_paths[0].lower().endswith(".usdz")
                )
            ):
                referenced_stage_fnames[obj_id] = file_paths[0]

                # get origin of referenced transform
                origin_node = scene.graph.transforms.children[obj_id][0]
                object_transform = scene.get_transform(origin_node)
                referenced_stage_transforms[obj_id] = object_transform

    if separate_assets:
        for obj_id in scene.get_object_names():
            if obj_id not in referenced_stage_fnames:
                if fname is None:
                    referenced_stages[obj_id] = usd_export.create_stage_in_memory(up_axis=up_axis, meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit)
                else:
                    referenced_stages[obj_id] = usd_export.create_stage(
                        os.path.join(folder, obj_id + "." + file_type), up_axis=up_axis, meters_per_unit=meters_per_unit, kilograms_per_unit=kilograms_per_unit
                    )
                referenced_stage_fnames[obj_id] = "./" + obj_id + "." + file_type
                referenced_stage_transforms[obj_id] = scene.get_transform(obj_id)

    # add lights
    if include_light_nodes:
        for light in scene._scene.lights:
            if isinstance(light, dict):
                light_name = light['name']
                if light['type'] == "RectLight":
                    usd_export.add_rect_light(
                        stage=stage,
                        scene_path=f"/world/{light_name}",
                        transform=scene.get_transform(light_name),
                        **light,
                    )
                elif light['type'] == "DiskLight":
                    usd_export.add_disk_light(
                        stage=stage,
                        scene_path=f"/world/{light_name}",
                        transform=scene.get_transform(light_name),
                        **light,
                    )
            elif isinstance(light, trimesh.scene.lighting.PointLight):
                usd_export.add_sphere_light(
                    stage=stage,
                    scene_path=f"/world/{light.name}",
                    radius=1.0 if light.radius is None else light.radius,
                    intensity=light.intensity,
                    transform=scene.get_transform(light.name),
                )
            elif isinstance(light, trimesh.scene.lighting.DirectionalLight):
                usd_export.add_distant_light(
                    stage=stage,
                    scene_path=f"/world/{light.name}",
                    transform=scene.get_transform(light.name),
                    intensity=light.intensity,
                )
            

    # create object_links dictionary
    scene_object_links = {
        obj_id: _get_object_links(obj_id=obj_id, scene=scene) for obj_id in scene.get_object_names()
    }

    # create mapping between scenegraph nodes and USD names
    scenegraph_to_usd_primpath = nodes_to_primpaths(
        scene=scene, scene_object_links=scene_object_links
    )

    # add joint names to mapping
    for obj_id in scene.get_object_names():
        obj_id_usd = obj_id.replace("-", "_").replace(":", "_").replace(".", "_")
        for joint_id in scene.get_joint_names(obj_id=obj_id, include_fixed_floating_joints=True):
            prim_path = f"/world/{obj_id_usd}/" + "/".join(joint_id.split("/")[1:]).replace(
                "-", "_"
            ).replace(":", "_").replace(".", "_")
            scenegraph_to_usd_primpath[joint_id] = prim_path

    def map_scene_path(path):
        if path in list(scene.metadata["object_nodes"].keys()):
            _, object_link_roots, _, _ = scene_object_links[path]
            log.debug(f"Map {path} to {object_link_roots[0][0]}. Chosen among: {object_link_roots}")
            return object_link_roots[0][0]
        log.debug(f"Map {path} to itself.")
        return path

    # add links
    # for i, key in enumerate(self.graph.nodes_geometry):
    materials_table = {}

    def add_geometry_node(stage, scene_path, g, g_vis, i, transform):
        if len(scene_path) == 0:
            raise ValueError(f"Cannot add geometry for empty scene_path!")

        log.debug(f"Adding geometry: {scene_path}")

        visible = True
        if "layer" in g.metadata and "collision" == g.metadata["layer"]:
            visible = False

        is_primitive = False
        local_primitive_transform = np.eye(4)
        if hasattr(g, "primitive"):
            is_primitive = True
            if isinstance(g, trimesh.primitives.Sphere):
                primitive_params = {
                    "primitive": "Sphere",
                    "radius": g.primitive.radius,
                }
                local_primitive_transform = tra.translation_matrix(g.primitive.center)
            elif isinstance(g, trimesh.primitives.Box):
                primitive_params = {
                    "primitive": "Cube",
                    "extents": g.primitive.extents,
                }
                local_primitive_transform = g.primitive.transform
            elif isinstance(g, trimesh.primitives.Cylinder):
                primitive_params = {
                    "primitive": "Cylinder",
                    "radius": g.primitive.radius,
                    "height": g.primitive.height,
                }
                local_primitive_transform = g.primitive.transform
            elif isinstance(g, trimesh.primitives.Capsule):
                primitive_params = {
                    "primitive": "Capsule",
                    "radius": g.primitive.radius,
                    "height": g.primitive.height,
                }
                local_primitive_transform = (
                    tra.translation_matrix([0, 0, g.primitive.height * 0.5]) @ g.primitive.transform
                )
            else:
                log.warning(
                    f"Can't export {g.primitive} to USD. Will serialize to triangular mesh."
                )
                is_primitive = False

        material_textures = None
        main_color = None

        if is_primitive:
            # it's a primitive shape
            log.debug(f"Adding primitive: {primitive_params}. Translation: {transform[:3, 3]}.")
            prim = usd_export.add_primitive(
                stage=stage,
                scene_path=scene_path,
                transform=transform @ local_primitive_transform,
                **primitive_params,
            )

            # add material
            if g_vis.defined and hasattr(g_vis, "main_color"):
                material_textures = usd_export.PBRMaterial(
                    diffuse_color=g_vis.main_color[:3] / 255.0,
                )
                main_color = g_vis.main_color[:3] / 255.0
        else:
            # it's a mesh
            g.apply_transform(transform)

            uv_exists = hasattr(g_vis, "uv")  # and len(g_vis.uv) > 0

            log.debug(f"Adding to {scene_path}  (uv_exists: {uv_exists})")
            prim = usd_export.add_mesh(
                stage=stage,
                scene_path=scene_path,
                vertices=g.vertices,
                faces=g.faces,
                uvs=g_vis.uv if uv_exists else None,
                # face_uvs_idx=g.faces if uv_exists else None,
                # face_normals=g.face_normals,
                single_sided=True,
            )

            log.debug(f"Visual kind of exported mesh: {g_vis.kind}")
            if uv_exists:
                if not hasattr(g_vis.material, 'image'):
                    # This is a trimesh.visual.material.PBRMaterial
                    material = g_vis.material
                    diffuse_texture = None
                    if material.baseColorTexture is not None:
                        image_dims = len(np.asarray(material.baseColorTexture).shape)
                        if image_dims == 3:
                            diffuse_texture = np.transpose(
                                np.asarray(material.baseColorTexture) / 255.0, axes=(2, 0, 1)
                            )
                        else:
                            diffuse_texture = np.array(
                                [
                                    np.asarray(material.baseColorTexture) / 255.0,
                                    np.asarray(material.baseColorTexture) / 255.0,
                                    np.asarray(material.baseColorTexture) / 255.0,
                                ]
                            )

                    material_textures = usd_export.PBRMaterial(
                        diffuse_color=material.baseColorFactor[:3] / 255.0 if material.baseColorFactor is not None else None,
                        metallic_value=material.metallicFactor,
                        roughness_value=material.roughnessFactor,
                        diffuse_texture=diffuse_texture,
                    )
                elif g_vis.material.image is None:
                    # This is a trimesh.visual.material.SimpleMaterial
                    try:
                        material = g_vis.to_texture().material
                        material_textures = usd_export.PBRMaterial(
                            diffuse_texture=np.transpose(
                                np.asarray(material.image) / 255.0, axes=(2, 0, 1)
                            ),
                        )
                    except AttributeError:
                        material_textures = usd_export.PBRMaterial(
                            diffuse_color=g_vis.material.diffuse[:3] / 255.0,
                            specular_color=g_vis.material.specular[:3] / 255.0,
                            # is_specular_workflow=?  # if False, specular_color will be ignored
                            # How to translate:
                            # g_vis.material.glossiness
                            # g_vis.material.ambient
                            # g_vis.material.main_color
                        )

                else:
                    material = g_vis.material

                    image_dims = len(np.asarray(material.image).shape)
                    if image_dims == 3:
                        diffuse_texture = np.transpose(
                            np.asarray(material.image) / 255.0, axes=(2, 0, 1)
                        )
                    else:
                        diffuse_texture = np.array(
                            [
                                np.asarray(material.image) / 255.0,
                                np.asarray(material.image) / 255.0,
                                np.asarray(material.image) / 255.0,
                            ]
                        )
                    material_textures = usd_export.PBRMaterial(
                        diffuse_texture=diffuse_texture,
                    )
            else:
                if hasattr(g_vis, "material"):
                    if g_vis.material.image is None:
                        try:
                            material = g_vis.to_texture().material

                            material_textures = usd_export.PBRMaterial(
                                diffuse_texture=np.transpose(
                                    np.asarray(material.image) / 255.0,
                                    axes=(2, 0, 1),
                                ),
                                # diffuse_color=material.diffuse[:3] / 255.0,
                            )
                        except AttributeError:
                            material_textures = usd_export.PBRMaterial(
                                # diffuse_color=g_vis.main_color[:3] / 255.0,
                                diffuse_color=g_vis.material.diffuse[:3]
                                / 255.0,
                            )
                            main_color = g_vis.material.diffuse[:3] / 255.0

                        #     material_textures = usd_export.PBRMaterial(
                        #         diffuse_color=g_vis.material.diffuse[:3]
                        #         / 255.0,  # main_color[:3] / 255.0,
                        #     )
                    else:
                        material_textures = usd_export.PBRMaterial(
                            diffuse_texture=np.transpose(
                                np.asarray(g_vis.material.image) / 255.0,
                                axes=(2, 0, 1),
                            ),
                        )
                else:
                    material_textures = usd_export.PBRMaterial(
                        diffuse_color=g_vis.main_color[:3] / 255.0,
                    )
                    main_color = g_vis.main_color[:3] / 255.0

        if main_color is not None:
            display_color = usd_export.UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "displayColor", usd_export.Sdf.ValueTypeNames.Float2Array
            )
            log.debug(f"Adding displayColor: {main_color} for prim {prim}")
            display_color.Set(main_color)
        elif export_materials and (material_textures is not None):
            hash_value = 0
            if material_textures.diffuse_texture is not None:
                hash_value += hash(material_textures.diffuse_texture.data.tobytes())
            if material_textures.diffuse_color is not None:
                hash_value += hash(np.asarray(material_textures.diffuse_color).data.tobytes())
            if material_textures.specular_color is not None:
                hash_value += hash(np.asarray(material_textures.specular_color).data.tobytes())

            if hash_value == 0:
                assert False, "hash_value needs to be updated"

            if hash_value not in materials_table:
                write_to_file = True
                texture_scene_path = f"{'/'.join(scene_path.split('/')[:-1])}/material{i:04}"
                texture_file_prefix = f"texture{i:04}"
                materials_table[hash_value] = (texture_scene_path, texture_file_prefix)
            else:
                write_to_file = False
                texture_scene_path, texture_file_prefix = materials_table[hash_value]

            material_textures.write_to_usd(
                stage=stage,
                usd_dir=folder,
                scene_path=texture_scene_path,
                bound_prims=[prim],
                texture_dir=texture_dir,
                texture_file_prefix=texture_file_prefix,
                write_to_file=write_to_file,
            )

        log.debug(f"Set visibility to {visible} for prim {prim}.")
        if not visible and not ignore_layers:
            usd_export.set_visibility(stage=stage, prim=prim, visibility="invisible")

        log.debug(f"Adding physics scheme to: {scene_path}")
        usd_export.add_physics_schemas(
            stage=stage,
            scene_path=scene_path,
            collision_api=True,
        )

    def add_joint_to_usd(
        stage, parent_path, child_path, joint_info, body_0_transform, body_1_transform
    ):
        log.debug(
            f"Adding joint to USD stage: {parent_path} -> {child_path} (type: {joint_info['type']})"
        )
        if joint_info["type"] == "revolute" or joint_info["type"] == "continuous":
            joint_offset = trimesh.geometry.align_vectors(joint_info["axis"], [0, 0, 1])
            joint_offset_inv = tra.inverse_matrix(joint_offset)

            usd_export.add_joint(
                stage=stage,
                scene_path=scenegraph_to_usd_primpath[joint_info["name"]],
                body_0_path=parent_path,
                body_1_path=child_path,
                body_0_transform=body_0_transform @ joint_offset_inv,
                body_1_transform=body_1_transform @ joint_offset_inv,
                joint_axis="Z",
                limit_lower=np.rad2deg(joint_info["limit_lower"])
                if joint_info["type"] == "revolute"
                else -1e4,
                limit_upper=np.rad2deg(joint_info["limit_upper"])
                if joint_info["type"] == "revolute"
                else +1e4,
                joint_type="PhysicsRevoluteJoint",
                stiffness=joint_info.get("stiffness", None),
                damping=joint_info.get("damping", None),
            )
        elif joint_info["type"] == "prismatic":
            joint_offset = trimesh.geometry.align_vectors(joint_info["axis"], [0, 0, 1])
            joint_offset_inv = tra.inverse_matrix(joint_offset)

            usd_export.add_joint(
                stage=stage,
                scene_path=scenegraph_to_usd_primpath[joint_info["name"]],
                body_0_path=parent_path,
                body_1_path=child_path,
                body_0_transform=body_0_transform @ joint_offset_inv,
                body_1_transform=body_1_transform @ joint_offset_inv,
                joint_axis="Z",
                limit_lower=joint_info["limit_lower"],
                limit_upper=joint_info["limit_upper"],
                joint_type="PhysicsPrismaticJoint",
                stiffness=joint_info.get("stiffness", None),
                damping=joint_info.get("damping", None),
            )
        elif joint_info["type"] == "floating":
            pass
        elif joint_info["type"] == "fixed":
            usd_export.add_joint(
                stage=stage,
                scene_path=scenegraph_to_usd_primpath[joint_info["name"]],
                body_0_path=parent_path,
                body_1_path=child_path,
                body_0_transform=body_0_transform,
                body_1_transform=body_1_transform,
                joint_type="PhysicsFixedJoint",
                joint_axis=None,
            )
            fixed_joint_partitions.append(child_path)
        else:
            log.warning(f"Unknown joint type, won't convert to USD: {joint_info['type']}")
    
    # go through object graph
    cnt = 0
    for object_name in scene.get_object_names():
        if object_name in referenced_stage_fnames and object_name not in referenced_stages:
            continue

        current_stage = stage if not separate_assets else referenced_stages[object_name]

        # get the links
        object_links, object_link_roots, _, links_without_parent_joints = scene_object_links[
            object_name
        ]
        object_transform = scene.get_transform(object_name, frame_from=scene.graph.base_frame)

        scene_path = scenegraph_to_usd_primpath[object_name]
        log.debug(
            f"Adding xform to USD stage: {scene_path}  (original: {object_name}), translation: {object_transform[:3, 3]}"
        )
        usd_export.add_xform(
            stage=current_stage,
            scene_path=scene_path,
            transform=object_transform,
        )

        # iterate over object graph
        root_link_of_object = object_link_roots[links_without_parent_joints[0]][0]
        log.debug(f"Root link of object: {root_link_of_object}")
        for object_link, object_link_root in zip(object_links, object_link_roots):
            log.debug(
                f"Object link: {object_link}. Object link nodes: {object_link.nodes}. Object link"
                f" root nodes: {object_link_root}"
            )
            assert len(object_link_root) == 1

            # We remember the link offset and use it to transform all children and joints of the object's first link
            # (we need an identity transform for the first link of an articulated object, for isaac.core)
            if scene.is_articulated(object_name) and root_link_of_object in object_link.nodes:
                link_offset = scene.get_transform(root_link_of_object, frame_from=object_name)
            else:
                link_offset = np.eye(4)

            # Add node one collection node if link only contains geometries

            # Sort nodes to ensure reproducibility / same ordering within USD file
            for object_node in sorted(object_link.nodes):
                log.debug(f"Going through {object_node}.")
                if object_node == root_link_of_object and scene.is_articulated(object_name):
                    # Convention for articulated/reduced coordinate objects: The first link has the identity transform
                    log.debug(f"No link offset. Using identity transform.")
                    transform = np.eye(4)
                elif object_node == object_link_root[0]:
                    log.debug(
                        f"Transform: {object_node} <- {object_name}. Adding link offset:"
                        f" {link_offset[:3, 3]}. Euler:"
                        f" {tra.euler_from_matrix(link_offset[:3, :3])}"
                    )
                    transform = link_offset @ scene.get_transform(
                        object_node, frame_from=object_name
                    )
                else:
                    log.debug(
                        f"Transform: {object_node} <- {object_name}. Adding link offset:"
                        f" {link_offset[:3, 3]}. Euler:"
                        f" {tra.euler_from_matrix(link_offset[:3, :3])}"
                    )
                    transform = link_offset @ scene.get_transform(
                        object_node, frame_from=object_link_root[0]
                    )

                log.debug(
                    "Transform: "
                    f"translation: {transform[:3, 3]} "
                    f"rotation: {tra.euler_from_matrix(transform[:3, :3])}"
                )

                _, geom_name = scene.graph[object_node]
                if geom_name is not None:
                    # (1) copying is necessary otherwise the scene will be mutated
                    # (2) copying will not properly copy visual properties
                    # g = self.scene.geometry[geom_name]
                    # The `include_cache=True` seems to have fixed it(?)
                    # I copy the visual properties separately
                    if isinstance(scene.scene.geometry[geom_name], trimesh.primitives.Primitive):
                        g = scene.scene.geometry[geom_name].copy()
                    else:                        
                        g = scene.scene.geometry[geom_name].copy(include_cache=True)
                    
                    g_vis = scene.scene.geometry[geom_name].visual.copy()

                    scene_path = scenegraph_to_usd_primpath[object_node]

                    log.debug(f"Adding geometry to USD stage: {scene_path}   (original: {object_node})")
                    
                    scene_path_extension = ""
                    if len(object_link.nodes) == 1 and scene.is_articulated(object_name):
                        usd_export.add_xform(
                            stage=current_stage,
                            scene_path=scene_path,
                            transform=transform,
                        )
                        transform = np.eye(4)
                        scene_path_extension = '/' +scene_path.split("/")[-1]

                    add_geometry_node(
                        stage=current_stage,
                        scene_path=scene_path + scene_path_extension,
                        g=g,
                        g_vis=g_vis,
                        i=cnt,
                        transform=transform,
                    )
                    cnt += 1
                else:
                    scene_path = scenegraph_to_usd_primpath[object_node]
                    log.debug(
                        f"Adding xform to USD stage: {scene_path} (original: {object_node}), translation: {transform[:3, 3]}"
                    )
                    usd_export.add_xform(
                        stage=current_stage,
                        scene_path=scene_path,
                        transform=transform,
                    )

    # add joints
    # remember to put all into collision group
    fixed_joint_partitions = []
    for object_name in scene.get_object_names():
        if object_name in referenced_stage_fnames and object_name not in referenced_stages:
            continue

        current_stage = stage if not separate_assets else referenced_stages[object_name]

        # iterate over object graph
        (
            object_links,
            object_link_roots,
            object_joints,
            links_without_parent_joints,
        ) = scene_object_links[object_name]
        log.debug(f"{object_name} links: {[l.nodes for l in object_links]}")
        log.debug(f"{object_name} link roots: {object_link_roots}")
        log.debug(f"{object_name} joints: {object_joints}")
        log.debug(
            f"{object_name} root link(s):"
            f" {[object_link_roots[x] for x in links_without_parent_joints]}"
        )

        # check root and add joint (unless it's a floating one)
        edge = scene.graph.transforms.edge_data[
            (scene.graph.transforms.parents[object_name], object_name)
        ]
        root_link_of_object = object_link_roots[links_without_parent_joints[0]][0]
        child_node = object_link_roots[links_without_parent_joints[0]][0]

        parent_node = scene.graph.transforms.parents[object_name]

        edge_is_joint = (
            EDGE_KEY_METADATA in edge and edge[EDGE_KEY_METADATA] is not None and "joint" in edge[EDGE_KEY_METADATA]
        )
        if scene.is_articulated(object_name):
            if not edge_is_joint:
                # This is dangerous
                warning_msg = (
                    f"Object '{object_name}' is articulated but is not connected with a fixed or"
                    " floating joint. Will add a fixed joint to the scene with name "
                )

                # Find name that is not being used in the scene
                scene_joint_names = scene.get_joint_names(include_fixed_floating_joints=True)
                fixed_joint_name = f"{object_name}/fixed_world_joint"
                joint_cnt = 0
                while fixed_joint_name in scene_joint_names:
                    fixed_joint_name = f"{object_name}/fixed_world_joint_{joint_cnt}"
                    joint_cnt += 1

                log.warning(
                    warning_msg
                    + f"'{fixed_joint_name}'. From node"
                    f" '{scene.graph.transforms.parents[object_name]}' to '{object_name}'."
                )

                # Add a new fixed joint to the scene
                scene.add_joint(
                    parent_node=scene.graph.transforms.parents[object_name],
                    child_node=object_name,
                    type="fixed",
                    name=fixed_joint_name,
                )

                edge = scene.graph.transforms.edge_data[
                    (scene.graph.transforms.parents[object_name], object_name)
                ]

                # Ensure the name can be mapped to a USD entity
                scenegraph_to_usd_primpath[fixed_joint_name] = (
                    f"/world/{object_name.replace('-', '_').replace(':', '_').replace('.', '_')}/{fixed_joint_name.split('/')[-1]}"
                )

            joint_info = edge[EDGE_KEY_METADATA]["joint"]

            # Since the first link by convention is transformed by an identity
            # and will be identical to the object pose, the joint transforms
            # will also be identity
            body_0_transform = scene.get_transform(object_name)
            body_1_transform = np.eye(4)

            parent_path = scenegraph_to_usd_primpath[parent_node]
            child_path = scenegraph_to_usd_primpath[child_node]

            log.debug(f"{parent_node} ~ {parent_path}  -->  {child_node} ~ {child_path}")
            add_joint_to_usd(
                current_stage,
                parent_path=parent_path,
                child_path=child_path,
                joint_info=joint_info,
                body_0_transform=body_0_transform,
                body_1_transform=body_1_transform,
            )
        elif edge_is_joint:
            joint_info = edge[EDGE_KEY_METADATA]["joint"]

            body_0_transform = scene.get_transform(object_name)
            body_1_transform = np.eye(4)

            # parent_path = scenegraph_to_usd_primpath[parent_node]
            # child_path = scenegraph_to_usd_primpath[child_node]

            add_joint_to_usd(
                current_stage,
                parent_path="",  #  "" means connect to world ("/world" doesn't work)
                child_path=scenegraph_to_usd_primpath[object_name],  # child_path,
                joint_info=joint_info,
                body_0_transform=body_0_transform,
                body_1_transform=body_1_transform,
            )

            if joint_info["type"] == "floating" or joint_info["type"] == "fixed":
                log.debug(f"Adding physics schema to {scenegraph_to_usd_primpath[object_name]}")
                usd_export.add_physics_schemas(
                    stage=current_stage,
                    scene_path=scenegraph_to_usd_primpath[object_name],
                    rigid_body_api=True,
                    mass_api=True,
                )

        def map_to_link_root(path, object_links, object_link_roots):
            link_index = ([1 if path in l.nodes else 0 for l in object_links]).index(1)
            link_root = object_link_roots[link_index][0]
            return link_root
        
        for object_joint_indices in object_joints:
            parent_node, child_node = object_joints[object_joint_indices][0]
            joint_info = object_joints[object_joint_indices][1][EDGE_KEY_METADATA]["joint"]

            # TODO: clean up
            parent_partition = parent_node
            child_partition = child_node
            log.debug(f"parent_node = {parent_node}, child_node = {child_node}")

            # parent_partition = map_scene_path(parent_partition)
            # child_partition = map_scene_path(child_partition)
            parent_partition = map_to_link_root(parent_partition, object_links=object_links, object_link_roots=object_link_roots)
            child_partition = map_to_link_root(child_partition, object_links=object_links, object_link_roots=object_link_roots)
            log.debug(f"parent_partition = {parent_partition}, child_partition = {child_partition}")

            parent_node = parent_partition
            child_node = child_partition

            body_0_transform = scene._scene.graph.get(frame_to=child_node, frame_from=parent_node)[0]
            body_1_transform = np.eye(4)
            if parent_node == root_link_of_object:
                # We need to apply the link_offset for joints that are attached to the first link
                # and subsequent links (to account for the convention that the first link has identity transform)
                link_offset = scene.get_transform(root_link_of_object, frame_from=object_name)
                body_0_transform = link_offset @ body_0_transform

            parent_path = (
                "/world"
                if parent_partition == scene._scene.graph.base_frame
                else scenegraph_to_usd_primpath[parent_partition]
            )
            child_path = scenegraph_to_usd_primpath[child_partition]
            log.debug(f"parent_path = {parent_path}, child_path = {child_path}")

            # Add RigidBodyAPI, MassAPI
            # Note: this code path is only traversed by articulated objects
            if joint_info["type"] != "fixed":
                if parent_path != "/world":
                    log.debug(f"Adding physics schema to {parent_path}")
                    usd_export.add_physics_schemas(
                        stage=current_stage,
                        scene_path=parent_path,
                        rigid_body_api=False,
                        mass_api=False,
                    )
                log.debug(f"Adding physics schema to {child_path}")
                usd_export.add_physics_schemas(
                    stage=current_stage,
                    scene_path=child_path,
                    rigid_body_api=False,
                    mass_api=False,
                )

            add_joint_to_usd(
                stage=current_stage,
                parent_path=parent_path,
                child_path=child_path,
                joint_info=joint_info,
                body_0_transform=body_0_transform,
                body_1_transform=body_1_transform,
            )

    # Set joint names and states
    # This also adds the ArticulationRootAPI
    for obj_id in scene.get_object_names():
        if obj_id in referenced_stage_fnames and obj_id not in referenced_stages:
            continue
        current_stage = stage if not separate_assets else referenced_stages[obj_id]
        joint_names = scene.get_joint_names(obj_id=obj_id)

        joint_positions = list(map(position_from_joint.get, joint_names))

        # remove object prefix
        joint_names = ["/".join(j.split("/")[1:]) for j in joint_names]

        if len(joint_names) > 0:
            # add joint names and state to object
            joint_info_path = scenegraph_to_usd_primpath[obj_id]
            log.debug(f"Adding joint information to {joint_info_path}")
            usd_export.add_joint_info(
                stage=current_stage,
                scene_path=joint_info_path,
                names=joint_names,
                positions=joint_positions,
            )

    # Add rigid body api to all articulated bodies
    # And collision filtering between the links
    for obj_id in scene.get_object_names():
        if obj_id in referenced_stage_fnames and obj_id not in referenced_stages:
            continue

        current_stage = stage if not separate_assets else referenced_stages[obj_id]

        if scene.is_articulated(obj_id):
            # iterate through all links and make them rigid bodies
            object_links, object_link_roots, _, _ = scene_object_links[obj_id]
            for object_link_root in object_link_roots:
                usd_export.add_physics_schemas(
                    stage=current_stage,
                    scene_path=scenegraph_to_usd_primpath[object_link_root[0]],
                    rigid_body_api=True,
                    mass_api=True,
                )

            # collision filtering - can only be applied to RigidBodyAPI, CollisionAPI, or ArticulationAPI
            usd_export.add_physics_filtered_pairs_api(
                stage=current_stage,
                scene_path=scenegraph_to_usd_primpath[obj_id],
                list_of_target_prims=[
                    scenegraph_to_usd_primpath[roots[0]] for roots in object_link_roots
                ],
            )

    # fill main stage if separate assets are supposed to be exported
    # or if some assets are supposed to be not written
    for object_name in referenced_stage_fnames:
        # retrieve meters_per_unit from referenced_stage
        # to calculate scale
        referenced_stage = (
            referenced_stages[object_name]
            if object_name in referenced_stages
            else get_stage(referenced_stage_fnames[object_name])
        )
        stage_will_be_written = bool(object_name in referenced_stages)
        
        referenced_stage_scaling_factor = None
        if not stage_will_be_written:
            referenced_stage_scaling_factor = (
                get_meters_per_unit(referenced_stage) / stage_meters_per_unit
            )
            
            # Determine how much the geometry is scaled w.r.t. to its initial size
            geom_scales = []
            for object_geom_node_name in scene.get_geometry_names(obj_id=object_name):
                object_geom_name = scene.graph.transforms.node_data[object_geom_node_name]['geometry']
                if "extents" in scene.geometry[object_geom_name].metadata:
                    geom_scales.append(
                        scene.geometry[object_geom_name].extents
                        / scene.geometry[object_geom_name].metadata["extents"]
                    )
        
            geom_scales = np.array(geom_scales)
            
            # check if consistent
            if len(geom_scales) > 0:
                if not np.allclose(geom_scales - geom_scales[0], 0, atol=1e-4):
                    log.warning(
                        f"Geometries of referenced USD asset have different scales. Won't apply any"
                        f" scaling. Result maybe wrong."
                    )
                else:
                    if np.allclose(geom_scales[0] - geom_scales[0][0], 0.0):
                        # scalar scale
                        referenced_stage_scaling_factor *= geom_scales[0][0]
                    else:
                        # vector scale
                        referenced_stage_scaling_factor = (
                            geom_scales[0] * referenced_stage_scaling_factor
                        )
            
            # If 1.0 set to None
            if np.allclose(referenced_stage_scaling_factor, 1.0):
                referenced_stage_scaling_factor = None

        # TODO(ceppner): does not scale the articulation units
        scene_path = scenegraph_to_usd_primpath[object_name]
        # Note: This overwrites the transform/scale in the
        # referenced file.
        prim = usd_export.add_xform(
            stage=stage,
            scene_path=scene_path,
            transform=referenced_stage_transforms[object_name],
            scale=referenced_stage_scaling_factor,
        )

        if scene.is_articulated(object_name):
            prim.AddAppliedSchema("PhysicsArticulationRootAPI")
            prim.AddAppliedSchema("PhysxArticulationAPI")

            # overwrite joint state info for referenced files that are not written
            if object_name in referenced_stage_fnames and object_name not in referenced_stages:
                # Duplicate code from line 780, except for add_articulation_api
                joint_names = scene.get_joint_names(obj_id=object_name)
                joint_positions = list(map(position_from_joint.get, joint_names))

                # remove object prefix
                joint_names = ["/".join(j.split("/")[1:]) for j in joint_names]

                if len(joint_names) > 0:
                    # add joint names and state to object
                    joint_info_path = scenegraph_to_usd_primpath[object_name]
                    log.debug(f"Adding joint information to {joint_info_path}")

                    usd_export.add_joint_info(
                        stage=stage,
                        scene_path=joint_info_path,
                        names=joint_names,
                        positions=joint_positions,
                        add_articulation_api=False,
                    )

        refs = prim.GetReferences()
        # By default referenced asset paths will be relative
        if not use_absolute_usd_paths and os.path.isabs(
            referenced_stage_fnames[object_name]
        ):
            asset_path = os.path.relpath(
                referenced_stage_fnames[object_name], os.path.dirname(fname)
            )
        else:
            asset_path = referenced_stage_fnames[object_name]

        if separate_assets:
            refs.AddReference(
                assetPath=asset_path,
                primPath=scene_path,
            )
        else:
            refs.AddReference(
                assetPath=asset_path,
            )

    # Go through objects and check if they have a parent that is not the world
    # Then write out the parent as an attached state
    if write_attribute_attached_state:
        for object_name in scene.get_object_names():
            # Get parent
            parent_frame = scene.graph.transforms.parents[object_name]

            # If it's the world frame abort
            if parent_frame == scene.graph.base_frame:
                continue

            # Get Transform
            transform = scene.get_transform(object_name, frame_from=parent_frame)

            # Get prim
            prim_path = scenegraph_to_usd_primpath[object_name]
            prim = stage.GetPrimAtPath(prim_path)

            # Determine whether the parent is part of any stage that is about to be written
            # or whether it already exists in which case we need to rely on the node_data
            # that was provided during USDAsset import
            if obj_id in referenced_stage_fnames and obj_id not in referenced_stages:
                if not "prim_path" in scene.graph.transforms.node_data[parent_frame]:
                    raise RuntimeError(
                        f"The node property 'prim_path' is missing for node '{parent_frame}'."
                    )
                parent_prim_path = scene.graph.transforms.node_data[parent_frame]["prim_path"]
                # Build prim path based on imported source file and current object name
                parent_prim_path = (
                    scenegraph_to_usd_primpath[scene.get_object_name(parent_frame)]
                    + "/"
                    + "/".join(parent_prim_path.split("/")[2:])
                )
            else:
                parent_prim_path = scenegraph_to_usd_primpath[parent_frame]

            usd_export.create_attribute(
                prim=prim,
                attribute_name="AttachedState:parent",
                sdf_type=usd_export.Sdf.ValueTypeNames.String,
                value=parent_prim_path,
            )

            usd_export.create_attribute(
                prim=prim,
                attribute_name="AttachedState:transform",
                sdf_type=usd_export.Sdf.ValueTypeNames.Matrix4d,
                value=usd_export.Gf.Matrix4d(*transform.flatten().tolist()),
            )
    
    # set scene configuration back to original values
    scene.update_configuration(current_configuration)

    if fname is None:
        if len(referenced_stages) > 0:
            return stage, referenced_stages
        else:
            return stage
    else:
        for s in referenced_stages.values():
            s.Save()

        stage.Save()

def nodes_to_primpaths(scene, scene_object_links=None):
    scenegraph_to_usd_primpath = {}

    # calculate if not provided
    if scene_object_links is None:
        scene_object_links = {
            obj_id: _get_object_links(obj_id=obj_id, scene=scene)
            for obj_id in scene.get_object_names()
        }

    for node in scene.graph.nodes:
        if node == scene.graph.base_frame:
            scenegraph_to_usd_primpath[node] = "/world"
            continue

        obj_id = scene.get_object_name(node)
        if obj_id == "" or obj_id is None:
            log.debug(f"{node} not part of any object, skipping.")
            continue

        obj_id_usd = obj_id.replace("-", "_").replace(":", "_").replace(".", "_")

        if obj_id == node:
            prim_path = f"/world/{obj_id_usd}"
        else:
            obj_links, obj_roots, _, _ = scene_object_links[obj_id]

            link_id = [
                obj_root[0]
                for obj_link, obj_root in zip(obj_links, obj_roots)
                if node in obj_link.nodes
            ][0]
            link_id = "/".join(link_id.split("/")[1:])
            link_id = link_id.replace("-", "_").replace(":", "_").replace(".", "_")

            if f"{obj_id}/{link_id}" == node:
                prim_path = f"/world/{obj_id_usd}/{link_id}"
            else:
                rest_id = "/".join(node.split("/")[1:])
                rest_id = rest_id.replace("-", "_").replace(":", "_").replace(".", "_")
                prim_path = f"/world/{obj_id_usd}/{link_id}/{rest_id}"

        log.debug(f"Mapping scene graph node '{node}' to prim path '{prim_path}'")
        scenegraph_to_usd_primpath[node] = prim_path
    return scenegraph_to_usd_primpath


# Turn object into set of links
def _get_object_links(obj_id, scene):
    object_graph = nx.DiGraph()
    object_graph.add_nodes_from(scene.metadata["object_nodes"][obj_id])
    object_graph.remove_node(obj_id)
    for e in scene.graph.to_edgelist():
        if e[0] in object_graph and e[1] in object_graph:
            object_graph.add_edge(e[0], e[1], **e[2])
    to_remove = []
    tmp_joints = {}
    for e in object_graph.edges:
        if object_graph.edges[e].get(EDGE_KEY_METADATA) is not None and object_graph.edges[e].get(
            EDGE_KEY_METADATA
        ).get("joint", False):
            to_remove.append(e)
            tmp_joints[e] = object_graph.edges[e]
    # show_graph(object_graph)
    object_graph.remove_edges_from(to_remove)
    # show_graph(object_graph)
    object_links = [
        object_graph.subgraph(c).copy() for c in nx.weakly_connected_components(object_graph)
    ]
    object_link_roots = [
        [x for x, num in object_link.in_degree() if num == 0] for object_link in object_links
    ]
    object_joints = {}
    for j in tmp_joints:
        index_node_0 = next((i for i, obj in enumerate(object_links) if j[0] in obj.nodes), None)
        index_node_1 = next((i for i, obj in enumerate(object_links) if j[1] in obj.nodes), None)
        object_joints[(index_node_0, index_node_1)] = (j, tmp_joints[j])

    links_without_parent_joints = list(
        set(range(len(object_links))).difference([b for (a, b) in object_joints])
    )

    return object_links, object_link_roots, object_joints, links_without_parent_joints
