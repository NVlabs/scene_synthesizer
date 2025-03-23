# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import inspect
import os
import posixpath
import re
import warnings
from collections import namedtuple
from collections.abc import Callable

# Third Party
import numpy as np
import trimesh.transformations as tra

try:
    # Third Party
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
except ImportError:
    warnings.warn("Warning: module pxr not found", ImportWarning)

mesh_return_type = namedtuple(
    "mesh_return_type",
    [
        "vertices",
        "faces",
        "uvs",
        "face_uvs_idx",
        "face_normals",
        "materials_order",
        "materials",
        "display_color",
    ],
)


class MaterialError(Exception):
    pass


class MaterialReadError(MaterialError):
    pass


class MaterialNotSupportedError(MaterialError):
    pass


class MaterialLoadError(MaterialError):
    pass


class MaterialFileError(MaterialError):
    pass


class MaterialNotFoundError(MaterialError):
    pass


def _get_shader_parameters(shader, time):
    # Get shader parameters
    params = {}
    inputs = shader.GetInputs()
    for i in inputs:
        name = i.GetBaseName()
        params.setdefault(i.GetBaseName(), {})
        if UsdShade.ConnectableAPI.HasConnectedSource(i):
            connected_source = UsdShade.ConnectableAPI.GetConnectedSource(i)
            connected_inputs = connected_source[0].GetInputs()
            while connected_inputs:
                connected_input = connected_inputs.pop()
                if UsdShade.ConnectableAPI.HasConnectedSource(connected_input):
                    new_inputs = UsdShade.ConnectableAPI.GetConnectedSource(connected_input)[
                        0
                    ].GetInputs()
                    connected_inputs.extend(new_inputs)
                elif connected_input.Get(time=time) is not None:
                    params[name].setdefault(connected_input.GetBaseName(), {}).update(
                        {
                            "value": connected_input.Get(time=time),
                            "type": connected_input.GetTypeName().type,
                            "docs": connected_input.GetDocumentation(),
                        }
                    )
        else:
            params[name].update(
                {
                    "value": i.Get(time=time),
                    "type": i.GetTypeName().type,
                    "docs": i.GetDocumentation(),
                }
            )
    return params


class MaterialManager:
    """Material management utility.
    Allows material reader functions to be mapped against specific shaders. This allows USD import functions
    to determine if a reader is available, which material reader to use and which material representation to wrap the
    output with.

    Default registered readers:

    - UsdPreviewSurface: Import material with shader id `UsdPreviewSurface`. All parameters are supported,
      including textures. See https://graphics.pixar.com/usd/release/wp_usdpreviewsurface.html for more details
      on available material parameters.

    Example:
        >>> # Register a new USD reader for mdl `MyCustomPBR`
        >>> from kaolin.io import materials
        >>> dummy_reader = lambda params, texture_path, time: UsdShade.Material()
        >>> materials.MaterialManager.register_usd_reader('MyCustomPBR', dummy_reader)
    """

    _usd_readers = {}
    _obj_reader = None

    @classmethod
    def register_usd_reader(cls, shader_name, reader_fn):
        """Register a shader reader function that will be used during USD material import.

        Args:
            shader_name (str): Name of the shader
            reader_fn (Callable): Function that will be called to read shader attributes. The function must take as
                input a dictionary of input parameters, a string representing the texture path, and a time
                `(params, texture_path, time)` and typically return a `Material`
        """
        if shader_name in cls._usd_readers:
            warnings.warn(
                f"Shader {shader_name} is already registered. Overwriting previous definition."
            )

        if not isinstance(reader_fn, Callable):
            raise MaterialLoadError("The supplied `reader_fn` must be a callable function.")

        # Validate reader_fn expects 3 parameters
        if len(inspect.signature(reader_fn).parameters) != 3:
            raise ValueError(
                "Error encountered when validating supplied `reader_fn`. Ensure that "
                "the function takes 3 arguments: parameters (dict), texture_path (string) and time "
                "(float)"
            )

        cls._usd_readers[shader_name] = reader_fn

    @classmethod
    def read_from_file(cls, file_path, scene_path=None, texture_path=None, time=None):
        r"""Read USD material and return a Material object.
        The shader used must have a corresponding registered reader function.

        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Required only for reading USD files. Absolute path of UsdShade.Material prim
                within the USD file scene. Must be a valid ``Sdf.Path``.
            texture_path (str, optional): Path to textures directory. By default, the textures will be assumed to be
                under the same directory as the file specified by `file_path`.
            time (convertible to float, optional): Optional for reading USD files. Positive integer indicating the tim
                at which to retrieve parameters.

        Returns:
            (Material): Material object determined by the corresponding reader function.
        """
        if os.path.splitext(file_path)[1].lower() in [".usd", ".usda", ".usdc", ".usdz"]:
            if scene_path is None or not Sdf.Path(scene_path):
                raise MaterialLoadError(f"The scene_path `{scene_path}`` provided is invalid.")

            if texture_path is None:
                texture_file_path = os.path.dirname(file_path)
            elif not os.path.isabs(texture_path):
                usd_dir = os.path.dirname(file_path)
                texture_file_path = posixpath.join(usd_dir, texture_path)
            else:
                texture_file_path = texture_path

            stage = Usd.Stage.Open(file_path)
            material = UsdShade.Material(stage.GetPrimAtPath(scene_path))

            return cls.read_usd_material(material, texture_path=texture_file_path, time=time)

        elif os.path.splitext(file_path)[1] == ".obj":
            if cls._obj_reader is not None:
                return cls._obj_reader(file_path)
            else:
                raise MaterialNotSupportedError("No registered .obj material reader found.")

    @classmethod
    def read_usd_material(cls, material, texture_path, time=None):
        r"""Read USD material and return a Material object.
        The shader used must have a corresponding registered reader function. If no available reader is found,
        the material parameters will be returned as a dictionary.

        Args:
            material (UsdShade.Material): Valid USD Material prim
            texture_path (str, optional): Path to textures directory. If the USD has absolute paths
                to textures, set to an empty string. By default, the textures will be assumed to be
                under the same directory as the USD specified by `file_path`.
            time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

        Returns:
            (Material): Material object determined by the corresponding reader function.
        """
        if time is None:
            time = Usd.TimeCode.Default()

        if not UsdShade.Material(material):
            raise MaterialLoadError(
                f"The material `{material}` is not a valid UsdShade.Material object."
            )

        for surface_output in material.GetSurfaceOutputs():
            if not surface_output.HasConnectedSource():
                continue
            surface_shader = surface_output.GetConnectedSource()[0]
            shader = UsdShade.Shader(surface_shader)
            if not UsdShade.Shader(shader):
                raise MaterialLoadError(
                    f"The shader `{shader}` is not a valid UsdShade.Shader object."
                )

            if shader.GetImplementationSourceAttr().Get(time=time) == "id":
                shader_name = UsdShade.Shader(surface_shader).GetShaderId()
            elif shader.GetPrim().HasAttribute("info:mdl:sourceAsset"):
                # source_asset = shader.GetPrim().GetAttribute('info:mdl:sourceAsset').Get(time=time)
                shader_name = (
                    shader.GetPrim()
                    .GetAttribute("info:mdl:sourceAsset:subIdentifier")
                    .Get(time=time)
                )
            else:
                shader_name = ""
                warnings.warn(
                    f"A reader for the material defined by `{material}` is not yet implemented."
                )

            params = _get_shader_parameters(surface_shader, time)

            if shader_name not in cls._usd_readers:
                warnings.warn(
                    "No registered readers were able to process the material "
                    f"`{material}` with shader `{shader_name}`."
                )
                return params

            reader = cls._usd_readers[shader_name]
            return reader(params, texture_path, time)
        raise MaterialError(f"Error processing material {material}")


class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.

    Attributes:
        message (str)
    """

    __slots__ = ["message"]

    def __init__(self, message):
        self.message = message


def _get_flattened_mesh_attributes(stage, scene_path, with_materials, with_normals, time):
    """Return mesh attributes flattened into a single mesh."""
    stage_dir = os.path.dirname(str(stage.GetRootLayer().realPath))
    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    attrs = {}

    def _process_mesh(mesh_prim, ref_path, attrs):
        cur_first_idx_faces = sum([len(v) for v in attrs.get("vertices", [])])
        cur_first_idx_uvs = sum([len(u) for u in attrs.get("uvs", [])])
        mesh = UsdGeom.Mesh(mesh_prim)
        mesh_primvars = UsdGeom.PrimvarsAPI(mesh_prim)
        mesh_vertices = mesh.GetPointsAttr().Get(time=time)
        mesh_face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time=time)
        mesh_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time=time)
        mesh_st = mesh_primvars.GetPrimvar("st")
        mesh_subsets = UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(mesh_prim))
        mesh_material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]

        mesh_display_color = mesh_primvars.GetPrimvar("displayColor")
        if mesh_display_color and mesh_display_color.Get():
            display_color = mesh_display_color.Get()[0]
            attrs["display_color"] = np.array([
                display_color[0] * 255,
                display_color[1] * 255,
                display_color[2] * 255,
            ], dtype=np.uint8)

        # Parse mesh UVs
        if mesh_st:
            mesh_uvs = mesh_st.Get(time=time)
            mesh_uv_indices = mesh_st.GetIndices(time=time)
            mesh_uv_interpolation = mesh_st.GetInterpolation()
        mesh_face_normals = mesh.GetNormalsAttr().Get(time=time)

        # Parse mesh geometry
        if mesh_vertices:
            attrs.setdefault("vertices", []).append(np.array(mesh_vertices, dtype=np.float32))
        if mesh_vertex_indices:
            attrs.setdefault("face_vertex_counts", []).append(
                np.array(mesh_face_vertex_counts, dtype=np.int64)
            )
            vertex_indices = np.array(mesh_vertex_indices, dtype=np.int64) + cur_first_idx_faces
            attrs.setdefault("vertex_indices", []).append(vertex_indices)
        if with_normals and mesh_face_normals:
            attrs.setdefault("face_normals", []).append(
                np.array(mesh_face_normals, dtype=np.float32)
            )
        if mesh_st and mesh_uvs:
            attrs.setdefault("uvs", []).append(np.array(mesh_uvs, dtype=np.float32))
            if mesh_uv_interpolation in ["vertex", "varying"]:
                if not mesh_uv_indices:
                    # for vertex and varying interpolation, length of mesh_uv_indices should match
                    # length of mesh_vertex_indices
                    mesh_uv_indices = list(range(len(mesh_uvs)))
                mesh_uv_indices = np.array(mesh_uv_indices) + cur_first_idx_uvs
                face_uvs_idx = mesh_uv_indices[np.array(mesh_vertex_indices, dtype=np.int64)]
                attrs.setdefault("face_uvs_idx", []).append(face_uvs_idx)
            elif mesh_uv_interpolation == "faceVarying":
                if not mesh_uv_indices:
                    # for faceVarying interpolation, length of mesh_uv_indices should match
                    # num_faces * face_size
                    mesh_uv_indices = [
                        i for i, c in enumerate(mesh_face_vertex_counts) for _ in range(c)
                    ]
                else:
                    attrs.setdefault("face_uvs_idx", []).append(mesh_uv_indices + cur_first_idx_uvs)
            # elif mesh_uv_interpolation == 'uniform':
            else:
                raise NotImplementedError(
                    f"Interpolation type {mesh_uv_interpolation} is not currently supported"
                )

        # Parse mesh materials
        if with_materials:
            subset_idx_map = {}
            attrs.setdefault("materials", []).append(None)
            attrs.setdefault("material_idx_map", {})
            if mesh_material:
                mesh_material_path = str(mesh_material.GetPath())
                if mesh_material_path in attrs["material_idx_map"]:
                    material_idx = attrs["material_idx_map"][mesh_material_path]
                else:
                    try:
                        material = MaterialManager.read_usd_material(mesh_material, stage_dir, time)
                        material_idx = len(attrs["materials"])
                        attrs["materials"].append(material)
                        attrs["material_idx_map"][mesh_material_path] = material_idx
                    except MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                    except MaterialReadError as e:
                        warnings.warn(e.args[0])
            if mesh_subsets:
                for subset in mesh_subsets:
                    subset_material, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
                    subset_material_metadata = subset_material.GetPrim().GetMetadata("references")
                    mat_ref_path = ""
                    if ref_path:
                        mat_ref_path = ref_path
                    if subset_material_metadata:
                        asset_path = subset_material_metadata.GetAddedOrExplicitItems()[0].assetPath
                        mat_ref_path = os.path.join(ref_path, os.path.dirname(asset_path))
                    if not os.path.isabs(mat_ref_path):
                        mat_ref_path = os.path.join(stage_dir, mat_ref_path)
                    try:
                        kal_material = MaterialManager.read_usd_material(
                            subset_material, mat_ref_path, time
                        )
                    except MaterialNotSupportedError as e:
                        warnings.warn(e.args[0])
                        continue
                    except MaterialReadError as e:
                        warnings.warn(e.args[0])

                    subset_material_path = str(subset_material.GetPath())
                    if subset_material_path not in attrs["material_idx_map"]:
                        attrs["material_idx_map"][subset_material_path] = len(attrs["materials"])
                        attrs["materials"].append(kal_material)
                    subset_indices = np.array(subset.GetIndicesAttr().Get())
                    subset_idx_map[attrs["material_idx_map"][subset_material_path]] = subset_indices
            # Create material face index list
            if mesh_face_vertex_counts:
                for face_idx in range(len(mesh_face_vertex_counts)):
                    is_in_subsets = False
                    for subset_idx in subset_idx_map:
                        if face_idx in subset_idx_map[subset_idx]:
                            is_in_subsets = True
                            attrs.setdefault("materials_face_idx", []).extend(
                                [subset_idx] * mesh_face_vertex_counts[face_idx]
                            )
                    if not is_in_subsets:
                        if mesh_material:
                            attrs.setdefault("materials_face_idx", []).extend(
                                [material_idx] * mesh_face_vertex_counts[face_idx]
                            )
                        else:
                            # Assign to `None` material (ie. index 0)
                            attrs.setdefault("materials_face_idx", []).extend(
                                [0] * mesh_face_vertex_counts[face_idx]
                            )

    def _traverse(cur_prim, ref_path, attrs):
        metadata = cur_prim.GetMetadata("references")
        if metadata:
            ref_path = os.path.dirname(metadata.GetAddedOrExplicitItems()[0].assetPath)
        if UsdGeom.Mesh(cur_prim):
            _process_mesh(cur_prim, ref_path, attrs)
        # for child in cur_prim.GetChildren():
        #     _traverse(child, ref_path, attrs)

    _traverse(stage.GetPrimAtPath(scene_path), "", attrs)

    if not attrs.get("vertices"):
        warnings.warn(f"Scene object at {scene_path} contains no vertices.", UserWarning)

    # Only import vertices if they are defined for the entire mesh
    if (
        all([v is not None for v in attrs.get("vertices", [])])
        and len(attrs.get("vertices", [])) > 0
    ):
        attrs["vertices"] = np.concatenate(attrs.get("vertices"))
    else:
        attrs["vertices"] = None
    # Only import vertex index and counts if they are defined for the entire mesh
    if (
        all([vi is not None for vi in attrs.get("vertex_indices", [])])
        and len(attrs.get("vertex_indices", [])) > 0
    ):
        attrs["face_vertex_counts"] = np.concatenate(attrs.get("face_vertex_counts", []))
        attrs["vertex_indices"] = np.concatenate(attrs.get("vertex_indices", []))
    else:
        attrs["face_vertex_counts"] = None
        attrs["vertex_indices"] = None
    # Only import UVs if they are defined for the entire mesh
    if not all([uv is not None for uv in attrs.get("uvs", [])]) or len(attrs.get("uvs", [])) == 0:
        if len(attrs.get("uvs", [])) > 0:
            warnings.warn(
                "UVs are missing for some child meshes for prim at "
                f"{scene_path}. As a result, no UVs were imported."
            )
        attrs["uvs"] = None
        attrs["face_uvs_idx"] = None
    else:
        attrs["uvs"] = np.concatenate(attrs["uvs"])
        if attrs.get("face_uvs_idx", None):
            attrs["face_uvs_idx"] = np.concatenate(attrs["face_uvs_idx"])
        else:
            attrs["face_uvs_idx"] = None

    # Only import face_normals if they are defined for the entire mesh
    if (
        not all([n is not None for n in attrs.get("face_normals", [])])
        or len(attrs.get("face_normals", [])) == 0
    ):
        if len(attrs.get("face_normals", [])) > 0:
            warnings.warn(
                "Face normals are missing for some child meshes for "
                f"prim at {scene_path}. As a result, no Face Normals were imported."
            )
        attrs["face_normals"] = None
    else:
        attrs["face_normals"] = np.concatenate(attrs["face_normals"])

    if attrs.get("materials_face_idx") is None or max(attrs.get("materials_face_idx", [])) == 0:
        attrs["materials_face_idx"] = None
    else:
        attrs["materials_face_idx"] = np.array(attrs["materials_face_idx"])

    if all([m is None for m in attrs.get("materials", [])]):
        attrs["materials"] = None
    return attrs


def get_joint_transform(joint_prim):
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(joint_prim.GetStage())

    trans_gf_vec = joint_prim.GetAttribute("physics:localPos0").Get()
    orient_gf_quat = joint_prim.GetAttribute("physics:localRot0").Get()
    local_pos_0 = np.array(trans_gf_vec) if trans_gf_vec is not None else np.zeros(3)
    local_pos_0 *= meters_per_unit
    local_rot_0 = (
        np.eye(4)
        if orient_gf_quat is None
        else tra.quaternion_matrix([orient_gf_quat.real, *orient_gf_quat.imaginary])
    )
    local_transform_0 = tra.translation_matrix(local_pos_0) @ local_rot_0

    trans_gf_vec = joint_prim.GetAttribute("physics:localPos1").Get()
    orient_gf_quat = joint_prim.GetAttribute("physics:localRot1").Get()
    local_pos_1 = np.array(trans_gf_vec) if trans_gf_vec is not None else np.zeros(3)
    local_pos_1 *= meters_per_unit
    local_rot_1 = (
        np.eye(4)
        if orient_gf_quat is None
        else tra.quaternion_matrix([orient_gf_quat.real, *orient_gf_quat.imaginary])
    )
    local_transform_1 = tra.translation_matrix(local_pos_1) @ local_rot_1

    return local_transform_0, tra.inverse_matrix(local_transform_1)


def get_joint_position(joint_prim, default_value=0.0, time=None):
    """Get joint position when it is available, otherwise return default_value.

    Args:
        joint_prim (Usd.Prim): Joint prim.
        default_value (float, optional): The default joint position. Defaults to 0.0.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters. None indicates default TimeCode. Defaults to None.

    Returns:
        float: Joint position.
    """
    if time is None:
        time = Usd.TimeCode.Default()
    
    joint_position = default_value

    parent_prim = joint_prim.GetParent()
    joint_names = parent_prim.GetAttribute("joint_names").Get()
    joint_positions = parent_prim.GetAttribute("joint_positions").Get()
    if (
        joint_names is not None
        and joint_positions is not None
        and len(joint_names) == len(joint_positions)
    ):
        joint_names = np.array(joint_names).tolist()
        joint_positions = np.array(joint_positions)

        def index_containing_substring(the_list, substring):
            for i, s in enumerate(the_list):
                if substring in s:
                    return i
            return -1

        joint_position = joint_positions[index_containing_substring(joint_names, joint_prim.GetName())]
    return joint_position


def get_scale(prim):
    """Return 3D scale of the given prim in world frame.

    Args:
        prim (Usd.Prim): Prim.

    Returns:
        np.ndarray: 3D scale vector.
    """
    xformable = UsdGeom.Xformable(prim)
    global_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    scale = np.asarray(Gf.Transform(global_transform).GetScale())

    return scale


def get_pose(prim):
    """Get the pose of the given prim.
       Defaults to identity if is not defined.
       Note: This is the pose of the prim relative to its parent.

    Args:
        prim (Usd.Prim): Prim.

    Returns:
        np.ndarray: 4x4 homogeneous matrix
    """
    # Fix this, also the stage unit is missing
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(prim.GetStage())

    xformable = UsdGeom.Xformable(prim)
    local_transform = xformable.GetLocalTransformation().RemoveScaleShear()
    translation = local_transform.ExtractTranslation()
    rotation = local_transform.ExtractRotationQuat()

    xformable_parent = UsdGeom.Xformable(prim.GetParent())
    parent_transform = xformable_parent.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    scale_parent = np.array(Gf.Transform(parent_transform).GetScale())

    pose = tra.translation_matrix(
        scale_parent * translation * meters_per_unit
    ) @ tra.quaternion_matrix([rotation.real, *rotation.imaginary])

    return pose


def get_stage(file_path):
    """Return Usd.Stage object of USD file.

    Args:
        file_path (str): A file path to a .usd, .usda, or .usdc file.

    Returns:
        Usd.Stage: The USD stage.
    """
    assert os.path.exists(file_path)
    stage = Usd.Stage.Open(file_path)
    return stage


def get_root(stage):
    r"""Return the root prim scene path.

    Args:
        stage (Usd.Stage): A USD stage.

    Returns:
        (str): Root scene path.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve root scene path
        >>> root_prim = get_root('./new_stage.usd')
        >>> mesh = import_mesh('./new_stage.usd', root_prim)
        >>> mesh.vertices.shape
        torch.Size([9, 3])
        >>> mesh.faces.shape
        torch.Size([3, 3])
    """
    return stage.GetPseudoRoot().GetPath()


def traverse_instanced_children(prim):
    """Get every Prim child beneath `prim`, even if `prim` is instanced.

    From: https://github.com/ColinKennedy/USD-Cookbook/blob/master/tricks/traverse_instanced_prims/README.md

    Important:
        If `prim` is instanced, any child that this function yields will
        be an instance proxy.

    Args:
        prim (`pxr.Usd.Prim`): Some Prim to check for children.

    Yields:
        `pxr.Usd.Prim`: The children of `prim`.

    """
    for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
        yield child

        for subchild in traverse_instanced_children(child):
            yield subchild


def get_scene_paths(stage, scene_path_regex=None, prim_types=None, conditional=lambda x: True):
    r"""Return all scene paths contained in specified file. Filter paths with regular
    expression in `scene_path_regex` if provided.

    Args:
        stage (Usd.Stage): USD Stage.
        scene_path_regex (str, optional): Optional regular expression used to select returned scene paths.
        prim_types (list of str, optional): Optional list of valid USD Prim types used to
            select scene paths.
        conditional (function path: Bool): Custom conditionals to check

    Returns:
        (list of str): List of filtered scene paths.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve scene paths
        >>> get_scene_paths('./new_stage.usd', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0'), Sdf.Path('/World/Meshes/mesh_1'), Sdf.Path('/World/Meshes/mesh_2')]
        >>> get_scene_paths('./new_stage.usd', scene_path_regex=r'.*_0', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0')]
    """
    if scene_path_regex is None:
        scene_path_regex = ".*"
    if prim_types is not None:
        prim_types = [pt.lower() for pt in prim_types]

    scene_paths = []
    for p in traverse_instanced_children(stage.GetPseudoRoot()):
        is_valid_prim_type = prim_types is None or p.GetTypeName().lower() in prim_types
        is_valid_scene_path = re.match(scene_path_regex, str(p.GetPath()))
        passes_conditional = conditional(p)
        if is_valid_prim_type and is_valid_scene_path and passes_conditional:
            scene_paths.append(p.GetPath())
    return scene_paths


def import_primitive(stage, scene_path=None):
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    if scene_path is None:
        scene_path = get_root(stage=stage)

    prim = stage.GetPrimAtPath(scene_path)
    if not prim:
        raise ValueError(f'No prim found at "{scene_path}".')

    primitive_data = {
        "type": prim.GetTypeName(),
    }

    if primitive_data["type"] == "Cube":
        # assuming this is at the origin, scale is half the extents
        primitive_data["extents"] = [
            2 * x for x in prim.GetProperty("xformOp:scale").Get() * meters_per_unit
        ]
    elif primitive_data["type"] == "Sphere":
        primitive_data["radius"] = UsdGeom.Sphere(prim).GetRadiusAttr().Get() * meters_per_unit

        if prim.HasProperty("xformOp:scale"):
            primitive_data["radius"] *= prim.GetProperty("xformOp:scale").Get()[0]
    elif primitive_data["type"] == "Cylinder":
        primitive_data["radius"] = UsdGeom.Cylinder(prim).GetRadiusAttr().Get() * meters_per_unit
        primitive_data["height"] = UsdGeom.Cylinder(prim).GetHeightAttr().Get() * meters_per_unit

        if prim.HasProperty("xformOp:scale"):
            primitive_data["radius"] *= prim.GetProperty("xformOp:scale").Get()[0]
            primitive_data["height"] *= prim.GetProperty("xformOp:scale").Get()[2]
    elif primitive_data["type"] == "Capsule":
        primitive_data["height"] = UsdGeom.Capsule(prim).GetHeightAttr().Get() * meters_per_unit
        primitive_data["radius"] = UsdGeom.Capsule(prim).GetRadiusAttr().Get() * meters_per_unit

        if prim.HasProperty("xformOp:scale"):
            primitive_data["radius"] *= prim.GetProperty("xformOp:scale").Get()[0]
            primitive_data["height"] *= prim.GetProperty("xformOp:scale").Get()[2]

    return primitive_data


# Mesh Functions
def heterogeneous_mesh_handler_skip(*args):
    r"""Skip heterogeneous meshes."""
    return None


def heterogeneous_mesh_handler_empty(*args):
    """Return empty tensors for vertices and faces of heterogeneous meshes."""
    return (
        np.empty(shape=(0, 3), dtype=np.float32),
        np.empty(shape=(0,), dtype=np.int64),
        np.empty(shape=(0, 3), dtype=np.int64),
        np.empty(shape=(0, 2), dtype=np.float32),
        np.empty(shape=(0, 3), dtype=np.int64),
        np.empty(shape=(0, 3, 3), dtype=np.float32),
        np.empty(shape=(0,), dtype=np.int64),
    )


def heterogeneous_mesh_handler_naive_homogenize(vertices, face_vertex_counts, *features):
    r"""Homogenize list of faces containing polygons of varying number of edges to triangles using fan
    triangulation.
    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        *features: Variable length features that need to be handled. For example, faces and uvs.
    Returns:
        (list of np.array): Homogeneous list of attributes.
    """

    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr if isinstance(attr, list) else attr.tolist()
            idx = 0
            new_attr = []
            for face_vertex_count in face_vertex_counts:
                attr_face = attr[idx : (idx + face_vertex_count)]
                idx += face_vertex_count
                while len(attr_face) >= 3:
                    new_attr.append(attr_face[:3])
                    attr_face.pop(1)
            return np.array(new_attr)
        else:
            return None

    new_attrs = [_homogenize(a, face_vertex_counts) for a in features]
    new_counts = np.full(vertices.shape[0], dtype=np.int64, fill_value=3)

    return (vertices, new_counts, *new_attrs)


def import_mesh(
    stage,
    scene_path=None,
    with_materials=False,
    with_normals=False,
    heterogeneous_mesh_handler=None,
    time=None,
):
    r"""Import a single mesh from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face).
    All sub-meshes found under the `scene_path` are flattened to a single mesh. The following
    interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    Returns an unbatched representation.

    Args:
        stage (Usd.Stage): A USD stage.
        scene_path (str, optional): Scene path within the USD file indicating which primitive to import.
            If not specified, the all meshes in the scene will be imported and flattened into a single mesh.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

    Returns:

    namedtuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties (Not yet implemented)

    Example:
        >>> # Create a stage with some meshes
        >>> stage = export_mesh('./new_stage.usd', vertices=torch.rand(3, 3), faces=torch.tensor([[0, 1, 2]]),
        ... scene_path='/World/mesh1')
        >>> # Import meshes
        >>> mesh = import_mesh(file_path='./new_stage.usd', scene_path='/World/mesh1')
        >>> mesh.vertices.shape
        torch.Size([3, 3])
        >>> mesh.faces
        tensor([[0, 1, 2]])
    """
    if scene_path is None:
        scene_path = get_root(stage=stage)
    if time is None:
        time = Usd.TimeCode.Default()
    meshes_list = import_meshes(
        stage=stage,
        scene_paths=[scene_path],
        heterogeneous_mesh_handler=heterogeneous_mesh_handler,
        with_materials=with_materials,
        with_normals=with_normals,
        times=[time],
    )

    if len(meshes_list) == 0:
        return None

    return mesh_return_type(*meshes_list[0])


def get_meters_per_unit(stage):
    """Return meters_per_unit of a stage.

    Args:
        stage (Usd.Stage): A USD stage.

    Returns:
        float: The meters_per_unit property.
    """
    return UsdGeom.GetStageMetersPerUnit(stage)


def import_meshes(
    stage,
    scene_paths=None,
    with_materials=False,
    with_normals=False,
    heterogeneous_mesh_handler=None,
    times=None,
):
    r"""Import one or more meshes from a USD file in an unbatched representation.

    Supports homogeneous meshes (meshes with consistent numbers of vertices per face). Custom handling of
    heterogeneous meshes can be achieved by passing a function through the ``heterogeneous_mesh_handler`` argument.
    The following interpolation types are supported for UV coordinates: `vertex`, `varying` and `faceVarying`.
    For each scene path specified in `scene_paths`, sub-meshes (if any) are flattened to a single mesh.
    Returns unbatched meshes list representation. Prims with no meshes or with heterogenous faces are skipped.

    Args:
        stage (Usd.Stage): USD Stage.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, all prims of type `Mesh` will be imported.
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        heterogeneous_mesh_handler (function, optional): Optional function to handle heterogeneous meshes.
            The function's input and output must be  ``vertices`` (torch.FloatTensor), ``faces`` (torch.LongTensor),
            ``uvs`` (torch.FloatTensor), ``face_uvs_idx`` (torch.LongTensor), and ``face_normals`` (torch.FloatTensor).
            If the function returns ``None``, the mesh will be skipped. If no function is specified,
            an error will be raised when attempting to import a heterogeneous mesh.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:

    list of namedtuple of:
        - **vertices** (list of torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **uvs** (list of torch.FloatTensor): of shape (num_uvs, 2)
        - **face_uvs_idx** (list of torch.LongTensor): of shape (num_faces, face_size)
        - **face_normals** (list of torch.FloatTensor): of shape (num_faces, face_size, 3)
        - **materials** (list of kaolin.io.materials.Material): Material properties (Not yet implemented)

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Import meshes
        >>> meshes = import_meshes(file_path='./new_stage.usd')
        >>> len(meshes)
        3
        >>> meshes[0].vertices.shape
        torch.Size([3, 3])
        >>> [m.faces for m in meshes]
        [tensor([[0, 1, 2]]), tensor([[0, 1, 2]]), tensor([[0, 1, 2]])]
    """
    if scene_paths is None:
        scene_paths = get_scene_paths(stage=stage, prim_types=["Mesh"])

    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    vertices_list, faces_list, uvs_list, face_uvs_idx_list, face_normals_list = [], [], [], [], []
    materials_order_list, materials_list = [], []

    display_color_list = []

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    for scene_path, time in zip(scene_paths, times):
        mesh_attr = _get_flattened_mesh_attributes(
            stage, scene_path, with_materials, with_normals, time=time
        )
        vertices = mesh_attr["vertices"] * meters_per_unit
        face_vertex_counts = mesh_attr["face_vertex_counts"]
        faces = mesh_attr["vertex_indices"]
        uvs = mesh_attr["uvs"]
        face_uvs_idx = mesh_attr["face_uvs_idx"]
        face_normals = mesh_attr["face_normals"]
        materials_face_idx = mesh_attr["materials_face_idx"]
        materials = mesh_attr["materials"]

        display_color = mesh_attr.get("display_color", None)

        if faces is not None:
            if not np.all(face_vertex_counts == face_vertex_counts[0]):
                if heterogeneous_mesh_handler is None:
                    raise NonHomogeneousMeshError(
                        f"Mesh at {scene_path} is non-homogeneous "
                        f"and cannot be imported from {stage}."
                    )
                else:
                    mesh = heterogeneous_mesh_handler(
                        vertices,
                        face_vertex_counts,
                        faces,
                        uvs,
                        face_uvs_idx,
                        face_normals,
                        materials_face_idx,
                    )
                    if mesh is None:
                        continue
                    else:
                        (
                            vertices,
                            face_vertex_counts,
                            faces,
                            uvs,
                            face_uvs_idx,
                            face_normals,
                            materials_face_idx,
                        ) = mesh

            if faces.size > 0:
                faces = faces.reshape(-1, face_vertex_counts[0])

        if face_uvs_idx is not None and faces is not None and face_uvs_idx.shape[0] > 0:
            uvs = uvs.reshape(-1, 2)
            face_uvs_idx = face_uvs_idx.reshape(-1, faces.shape[1])
        if face_normals is not None and faces is not None and face_normals.shape[0] > 0:
            face_normals = face_normals.reshape(-1, faces.shape[1], 3)
        if faces is not None and materials_face_idx is not None:  # Create material order list
            materials_face_idx.view(-1, faces.shape[1])
            cur_mat_idx = -1
            materials_order = []
            for idx in range(len(materials_face_idx)):
                mat_idx = materials_face_idx[idx][0].item()
                if cur_mat_idx != mat_idx:
                    cur_mat_idx = mat_idx
                    materials_order.append([idx, mat_idx])
        else:
            materials_order = None

        vertices_list.append(vertices)
        faces_list.append(faces)
        uvs_list.append(uvs)
        face_uvs_idx_list.append(face_uvs_idx)
        face_normals_list.append(face_normals)
        materials_order_list.append(materials_order)
        materials_list.append(materials)

        display_color_list.append(display_color)

    params = [
        vertices_list,
        faces_list,
        uvs_list,
        face_uvs_idx_list,
        face_normals_list,
        materials_order_list,
        materials_list,
        display_color_list,
    ]
    return [
        mesh_return_type(v, f, uv, fuv, fn, mo, m, dc)
        for v, f, uv, fuv, fn, mo, m, dc in zip(*params)
    ]
