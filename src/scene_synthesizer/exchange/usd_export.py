# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From: https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/io/usd.py
#       https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/io/materials.py
# With additions:
#       - replace all pytorch calls with numpy
#       - def add_()

"""Export scene to USD format.
"""

# Standard Library
import itertools
import os
import posixpath
import re
import warnings
from abc import abstractmethod
from argparse import ArgumentError
from pathlib import Path

# Third Party
import numpy as np
import trimesh.transformations as tra
from PIL import Image

try:
    # Third Party
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
except ImportError:
    warnings.warn("Warning: module pxr not found", ImportWarning)


def get_root(file_path):
    r"""Return the root prim scene path.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

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
    stage = Usd.Stage.Open(file_path)
    return stage.GetPseudoRoot().GetPath()


def get_pointcloud_scene_paths(file_path):
    r"""Returns all point cloud scene paths contained in specified file. Assumes that point
    clouds are exported using this API.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (list of str): List of filtered scene paths.
    """
    geom_points_paths = get_scene_paths(file_path, prim_types=["Points"])
    point_instancer_paths = get_scene_paths(file_path, prim_types=["PointInstancer"])
    return geom_points_paths + point_instancer_paths


def get_scene_paths(file_path, scene_path_regex=None, prim_types=None, conditional=lambda x: True):
    r"""Return all scene paths contained in specified file. Filter paths with regular
    expression in `scene_path_regex` if provided.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
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
        [Sdf.Path('/world/Meshes/mesh_0'), Sdf.Path('/world/Meshes/mesh_1'), Sdf.Path('/world/Meshes/mesh_2')]
        >>> get_scene_paths('./new_stage.usd', scene_path_regex=r'.*_0', prim_types=['Mesh'])
        [Sdf.Path('/world/Meshes/mesh_0')]
    """
    stage = Usd.Stage.Open(file_path)
    if scene_path_regex is None:
        scene_path_regex = ".*"
    if prim_types is not None:
        prim_types = [pt.lower() for pt in prim_types]

    scene_paths = []
    for p in stage.Traverse():
        is_valid_prim_type = prim_types is None or p.GetTypeName().lower() in prim_types
        is_valid_scene_path = re.match(scene_path_regex, str(p.GetPath()))
        passes_conditional = conditional(p)
        if is_valid_prim_type and is_valid_scene_path and passes_conditional:
            scene_paths.append(p.GetPath())
    return scene_paths


def get_authored_time_samples(file_path):
    r"""
    Returns *all* authored time samples within the USD, aggregated across all primitives.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).

    Returns:
        (list)
    """
    stage = Usd.Stage.Open(file_path)
    scene_paths = get_scene_paths(file_path)
    res = set()
    for scene_path in scene_paths:
        prim = stage.GetPrimAtPath(scene_path)
        attr = prim.GetAttributes()
        res.update(set(itertools.chain.from_iterable([x.GetTimeSamples() for x in attr])))
    return sorted(res)


def create_stage(file_path, up_axis="Z", meters_per_unit=1.0, kilograms_per_unit=1.0):
    r"""Create a new USD file and return an empty stage.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        up_axis (['Y', 'Z']): Specify the stage up axis. Choose from ``['Y', 'Z']``.
        meters_per_unit (float): Meters per unit. Defaults to 1.
        kilograms_per_unit (float): Kilograms per unit. Defaults to 1.

    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd', up_axis='Z')
        >>> type(stage)
        <class 'pxr.Usd.Stage'>
    """
    assert os.path.exists(
        os.path.dirname(file_path)
    ), f"Directory {os.path.dirname(file_path)} not found."
    stage = Usd.Stage.CreateNew(str(file_path))
    world = stage.DefinePrim("/world", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    try:
        UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    except:
        warnings.warn(
            "Warning: UsdPhysics.SetStageKilogramsPerUnit not found. Most likely due to an old"
            " usd_core version."
        )

    return stage

def create_stage_in_memory(up_axis="Z", meters_per_unit=1.0, kilograms_per_unit=1.0):
    r"""Create a new USD stage in memory.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        up_axis (['Y', 'Z']): Specify the stage up axis. Choose from ``['Y', 'Z']``.
        meters_per_unit (float): Meters per unit. Defaults to 1.
        kilograms_per_unit (float): Kilograms per unit. Defaults to 1.

    Returns:
        (Usd.Stage)
    """
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/world", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    try:
        UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    except:
        warnings.warn(
            "Warning: UsdPhysics.SetStageKilogramsPerUnit not found. Most likely due to an old"
            " usd_core version."
        )

    return stage

def add_mesh(
    stage,
    scene_path,
    vertices=None,
    faces=None,
    uvs=None,
    face_uvs_idx=None,
    face_normals=None,
    double_sided=None,
    single_sided=None,
    time=None,
):
    r"""Add a mesh to an existing USD stage.

    Add a mesh to the USD stage. The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the mesh.
        scene_path (str): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = create_stage('./new_stage.usd')
        >>> mesh = add_mesh(stage, '/world/mesh', vertices, faces)
        >>> stage.Save()
    """
    if time is None:
        time = Usd.TimeCode.Default()

    usd_mesh = UsdGeom.Mesh.Define(stage, scene_path)

    if faces is not None:
        # num_faces = faces.size(0)
        num_faces = faces.shape[0]
        # face_vertex_counts = [faces.size(1)] * num_faces
        face_vertex_counts = [faces.shape[1]] * num_faces
        # faces_list = faces.view(-1).cpu().long().numpy()
        # faces_list = faces.view(-1).cpu().long().numpy()
        faces_list = faces.reshape(-1)
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces_list, time=time)
    if vertices is not None:
        # vertices_list = vertices.detach().cpu().float().numpy()
        vertices_list = vertices
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_list), time=time)
    if uvs is not None:
        interpolation = None
        # uvs_list = uvs.view(-1, 2).detach().cpu().float().numpy()
        uvs_list = uvs
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.Float2Array
        )
        pv.Set(uvs_list, time=time)
        if face_uvs_idx is not None:
            pv.SetIndices(
                # Vt.IntArray.FromNumpy(face_uvs_idx.view(-1).cpu().long().numpy()),
                Vt.IntArray.FromNumpy(face_uvs_idx.reshape(-1)),
                time=time,
            )
            interpolation = "faceVarying"
        else:
            # if vertices is not None and uvs.size(0) == vertices.size(0):
            if vertices is not None and uvs.shape[0] == vertices.shape[0]:
                interpolation = "vertex"
            # elif uvs.size(0) == faces.size(0):
            elif uvs.shape[0] == faces.shape[0]:
                interpolation = "uniform"
            # elif uvs.size(0) == len(faces_list):
            elif uvs.shape[0] == len(faces_list):
                interpolation = "faceVarying"

        if interpolation is not None:
            pv.SetInterpolation(interpolation)

    if face_uvs_idx is not None and uvs is None:
        raise ValueError('If providing "face_uvs_idx", "uvs" must also be provided.')

    if face_normals is not None:
        # face_normals = face_normals.view(-1, 3).cpu().float().numpy()
        usd_mesh.GetNormalsAttr().Set(face_normals, time=time)
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation("faceVarying")

    if double_sided is not None:
        usd_mesh.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(double_sided)
    if single_sided is not None:
        usd_mesh.GetPrim().CreateAttribute("singleSided", Sdf.ValueTypeNames.Bool).Set(single_sided)

    return usd_mesh.GetPrim()


def add_xform(stage, scene_path, transform=None, scale=None):
    """Add Xform to USD stage.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path ([type]): Absolute path of prim within the USD file scene.
        transform (np.ndarray, optional): 4x4 homogeneous matrix representing the pose of the xform. Defaults to None.
        scale (float or list[float], optional): Scalar or 3D scale value. Defaults to None.

    Return:
        xform_prim (Usd.Prim): The newly created prim.
    """
    xform_prim = stage.DefinePrim(scene_path, "Xform")
    if transform is not None:
        _set_transform_prim(xform_prim, transform, scale=scale)

    return xform_prim


def _set_transform_prim(prim, transform, scale=None):
    translation = tra.translation_from_matrix(transform).tolist()
    quaternion = tra.quaternion_from_matrix(transform)

    UsdGeom.XformCommonAPI(prim).SetTranslate(translation)

    # change orientation via quaternion
    xformable = UsdGeom.Xformable(prim)
    new_prim_orient_attr = xformable.AddOrientOp()
    new_prim_orient_attr.Set(Gf.Quatf(quaternion[0], quaternion[1:].tolist()))

    if scale is not None:
        scale_attr = xformable.AddScaleOp()
        if hasattr(scale, "__len__"):
            if len(scale) != 3:
                raise ValueError(
                    f"Scale argument needs to be either scalar or 3D. Current length: {len(scale)}"
                )
            scale_attr.Set(Gf.Vec3f(scale[0], scale[1], scale[2]))
        else:
            scale_attr.Set(Gf.Vec3f(scale, scale, scale))


def add_sphere_light(
    stage,
    scene_path,
    transform=None,
    radius=None,
    intensity=None,
):
    """Add a UsdLux.SphereLight to the scene.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the light prim.
        transform (np.ndarray, optional): 4x4 homogeneous matrix representing the pose of the light. Defaults to None.
        radius (float, optional): Radius of light. Defaults to None.
        intensity (float, optional): Intensity of light. Defaults to None.
    """
    sphere_light = UsdLux.SphereLight.Define(stage, scene_path)

    if radius is not None:
        sphere_light.CreateRadiusAttr(radius)
        sphere_light.GetPrim().CreateAttribute(
            "radius", Sdf.ValueTypeNames.Float, custom=False
        ).Set(radius)

    if intensity is not None:
        sphere_light.CreateIntensityAttr(intensity)
        sphere_light.GetPrim().CreateAttribute(
            "intensity", Sdf.ValueTypeNames.Float, custom=False
        ).Set(intensity)

    if transform is not None:
        _set_transform_prim(prim=sphere_light, transform=transform)

def add_distant_light(
    stage,
    scene_path,
    transform=None,
    intensity=None,
):
    """Add a UsdLux.DistantLight to the scene.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the light prim.
        transform (np.ndarray, optional): 4x4 homogeneous matrix representing the pose of the light. Defaults to None.
        intensity (float, optional): Intensity of light. Defaults to None.
    """
    distant_light = UsdLux.DistantLight.Define(stage, scene_path)

    if intensity is not None:
        distant_light.CreateIntensityAttr(intensity)
        distant_light.GetPrim().CreateAttribute(
            "intensity", Sdf.ValueTypeNames.Float, custom=False
        ).Set(intensity)

    if transform is not None:
        _set_transform_prim(prim=distant_light, transform=transform)

def add_rect_light(
    stage,
    scene_path,
    width=1.0,
    height=1.0,
    intensity=15000,
    shaping_cone_softness=1.0,
    shaping_cone_angle=180.0,
    transform=None,
    **kwargs,
):
    """Add a UsdLux.RectLight to the scene.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the light prim.
        extent (2-tuple, list[float]): Size of rectangular light.
        transform (np.ndarray, optional): 4x4 homogeneous matrix representing the pose of the light. Defaults to None.
        intensity (float, optional): Intensity of light. Defaults to None.
    """
    rect_light = UsdLux.RectLight.Define(stage, scene_path)
    prim = rect_light.GetPrim()
    
    rect_light.CreateHeightAttr(height)
    rect_light.CreateWidthAttr(width)

    prim.AddAppliedSchema("ShapingAPI")
    shaping_api = UsdLux.ShapingAPI(prim)
    shaping_api.CreateShapingConeAngleAttr(shaping_cone_angle)
    shaping_api.CreateShapingConeSoftnessAttr(shaping_cone_softness)

    if intensity is not None:
        rect_light.CreateIntensityAttr(intensity)
        prim.CreateAttribute(
            "intensity", Sdf.ValueTypeNames.Float, custom=False
        ).Set(intensity)

    if transform is not None:
        _set_transform_prim(prim=rect_light, transform=transform)

def add_disk_light(
    stage,
    scene_path,
    radius=1.0,
    intensity=60000,
    shaping_cone_softness=1.0,
    shaping_cone_angle=180.0,
    color_temperature=6500,
    enable_color_temperature=False,
    transform=None,
    **kwargs,
):
    """Add a UsdLux.DiskLight to the scene.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the light prim.
        extent (2-tuple, list[float]): Size of rectangular light.
        transform (np.ndarray, optional): 4x4 homogeneous matrix representing the pose of the light. Defaults to None.
        intensity (float, optional): Intensity of light. Defaults to None.
    """
    disk_light = UsdLux.DiskLight.Define(stage, scene_path)
    prim = disk_light.GetPrim()
    
    if radius is not None:
        disk_light.CreateRadiusAttr(radius)
        disk_light.GetPrim().CreateAttribute(
            "radius", Sdf.ValueTypeNames.Float, custom=False
        ).Set(radius)

    disk_light.CreateColorTemperatureAttr(color_temperature)
    disk_light.CreateEnableColorTemperatureAttr(enable_color_temperature)

    prim.AddAppliedSchema("ShapingAPI")
    shaping_api = UsdLux.ShapingAPI(prim)
    shaping_api.CreateShapingConeAngleAttr(shaping_cone_angle)
    shaping_api.CreateShapingConeSoftnessAttr(shaping_cone_softness)
    
    if intensity is not None:
        disk_light.CreateIntensityAttr(intensity)
        prim.CreateAttribute(
            "intensity", Sdf.ValueTypeNames.Float, custom=False
        ).Set(intensity)

    if transform is not None:
        _set_transform_prim(prim=disk_light, transform=transform)

def add_primitive(
    stage,
    scene_path,
    primitive,
    transform=None,
    **kwargs,
):
    """Add a primitive shape to the stage.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the object prim.
        primitive (str): Type of primitive. Must be 'Capsule', 'Cube', 'Cylinder', or 'Sphere'.
        transform (np.ndarray, optional): 4x4 homogenous matrix. Defaults to None.

    Raises:
        ArgumentError: Raised if primitive is unkonwn.

    Returns:
        Usd.Prim: The added prim.
    """
    # Add the prim
    new_prim = stage.DefinePrim(scene_path, primitive)

    if transform is not None:
        _set_transform_prim(prim=new_prim, transform=transform)

    if primitive == "Capsule":
        new_geom = UsdGeom.Capsule(new_prim)
        new_geom.CreateHeightAttr(kwargs["height"])
        new_geom.CreateRadiusAttr(kwargs["radius"])
    elif primitive == "Cube":
        scale_attr = UsdGeom.Xformable(new_prim).AddScaleOp()
        scale_attr.Set(Gf.Vec3f(list(np.array(kwargs["extents"]) * 0.5)))
    elif primitive == "Cylinder":
        new_geom = UsdGeom.Cylinder(new_prim)
        new_geom.CreateHeightAttr(kwargs["height"])
        new_geom.CreateRadiusAttr(kwargs["radius"])
    elif primitive == "Sphere":
        new_geom = UsdGeom.Sphere(new_prim)
        new_geom.CreateRadiusAttr(kwargs["radius"])
    else:
        raise ArgumentError(f"Unknown USD shape primitive: {primitive}")

    return new_prim


def add_joint(
    stage,
    scene_path,
    body_0_path,
    body_1_path=None,
    body_0_transform=None,
    body_1_transform=None,
    joint_type="PhysicsRevoluteJoint",
    joint_axis="Z",
    limit_lower=None,
    limit_upper=None,
    damping=None,
    stiffness=None,
    physx_joint_api=True,
):
    VALID_JOINT_TYPES = ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint"]
    if joint_type not in VALID_JOINT_TYPES:
        raise ValueError(
            f"Unknown USD joint_type '{joint_type}'.  Valid types: {VALID_JOINT_TYPES}."
        )
    
    # Add the prim
    joint_prim = stage.DefinePrim(scene_path, joint_type)

    # Add drive if stiffness or damping are defined
    if physx_joint_api and (damping is not None or stiffness is not None):
        if joint_type == "PhysicsRevoluteJoint":
            joint_prim.AddAppliedSchema("PhysicsDriveAPI:angular")
            drive_api = UsdPhysics.DriveAPI(joint_prim, 'angular')
        elif joint_type == "PhysicsPrismaticJoint":
            joint_prim.AddAppliedSchema("PhysicsDriveAPI:linear")
            drive_api = UsdPhysics.DriveAPI(joint_prim, 'linear')

        if damping is not None:
            if isinstance(damping, float):
                drive_api.CreateDampingAttr().Set(damping)
            else:
                warnings.warn(f"Damping is not of type float: {damping}")
        if stiffness is not None:
            if isinstance(damping, float):
                drive_api.CreateStiffnessAttr().Set(stiffness)
            else:
                warnings.warn(f"Stiffness is not of type float: {damping}")


    add_physics_schemas(stage=stage, scene_path=scene_path, physx_joint_api=physx_joint_api)

    if joint_type == "PhysicsRevoluteJoint":
        joint_api = UsdPhysics.RevoluteJoint(joint_prim)
    elif joint_type == "PhysicsPrismaticJoint":
        joint_api = UsdPhysics.PrismaticJoint(joint_prim)
    elif joint_type == "PhysicsFixedJoint":
        joint_api = UsdPhysics.FixedJoint(joint_prim)

    if joint_axis is not None:
        joint_api.CreateAxisAttr().Set(joint_axis)

    if body_0_path is not None and body_0_path != "":
        joint_api.CreateBody0Rel().AddTarget(body_0_path)

    if body_1_path is not None and body_1_path != "":
        joint_api.CreateBody1Rel().AddTarget(body_1_path)

    if body_0_transform is not None:
        joint_api.CreateLocalPos0Attr().Set(
            Gf.Vec3f(tra.translation_from_matrix(body_0_transform).tolist())
        )
        quat_0 = tra.quaternion_from_matrix(body_0_transform)
        joint_api.CreateLocalRot0Attr().Set(Gf.Quatf(quat_0[0], Gf.Vec3f(quat_0[1:].tolist())))

    if body_1_transform is not None:
        joint_api.CreateLocalPos1Attr().Set(
            Gf.Vec3f(tra.translation_from_matrix(body_1_transform).tolist())
        )
        quat_1 = tra.quaternion_from_matrix(body_1_transform)
        joint_api.CreateLocalRot1Attr().Set(Gf.Quatf(quat_1[0], Gf.Vec3f(quat_1[1:].tolist())))

    if limit_lower is not None:
        joint_api.CreateLowerLimitAttr().Set(limit_lower)

    if limit_upper is not None:
        joint_api.CreateUpperLimitAttr().Set(limit_upper)
    


def add_joint_info(stage, scene_path, names, positions, add_articulation_api=True):
    """Add joint names and position to the USD. And add articulation API.

    Args:
        stage (Usd.Stage): USD stage.
        scene_path (str): Path of the object prim.
        names (list[str]): List of joint names.
        positions (list[str]): List of joint positions. Needs to have same length as names.
        add_articulation_api (bool, optional): Whether to add PhysicsArticulationRootAPI and PhysxArticulationAPI. Defaults to True.

    Raises:
        ValueError: Raised if len(names) != len(positions).
    """
    if len(names) != len(positions):
        raise ValueError("List of joint names and positions have not the same length!")

    prim = stage.GetPrimAtPath(scene_path)

    if add_articulation_api:
        add_physics_schemas(stage=stage, scene_path=scene_path, articulation_api=True)

    joint_names_attr = prim.CreateAttribute("joint_names", Sdf.ValueTypeNames.StringArray)
    joint_names_attr.Set(names)
    joint_state_attr = prim.CreateAttribute("joint_positions", Sdf.ValueTypeNames.DoubleArray)
    joint_state_attr.Set(positions)


def add_physics_schemas(
    stage,
    scene_path,
    collision_api=False,
    rigid_body_api=False,
    mass_api=False,
    physx_joint_api=False,
    articulation_api=False,
):
    prim = stage.GetPrimAtPath(scene_path)
    if not prim.IsValid():
        raise ValueError(f"{scene_path} is not a valid prim. Can't add physics schema.")

    # Add API schemas to prim
    if collision_api:
        prim.AddAppliedSchema("PhysicsCollisionAPI")
    if rigid_body_api:
        prim.AddAppliedSchema("PhysicsRigidBodyAPI")
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
    if mass_api:
        prim.AddAppliedSchema("PhysicsMassAPI")
    if physx_joint_api:
        prim.AddAppliedSchema("PhysxJointAPI")
    
    if articulation_api:
        prim.AddAppliedSchema("PhysicsArticulationRootAPI")
        prim.AddAppliedSchema("PhysxArticulationAPI")


def add_physics_filtered_pairs_api(
    stage,
    scene_path,
    list_of_target_prims,
):
    # collision filtering - can only be applied to RigidBodyAPI, CollisionAPI, or ArticulationAPI
    prim = stage.GetPrimAtPath(scene_path)
    prim.AddAppliedSchema("PhysicsFilteredPairsAPI")

    filtered_pairs_api = UsdPhysics.FilteredPairsAPI(prim)

    for prim_path in list_of_target_prims:
        filtered_pairs_api.CreateFilteredPairsRel().AddTarget(
            stage.GetPrimAtPath(prim_path).GetPath()
        )


def set_kinematics_enable(stage, scene_path):
    prim = stage.GetPrimAtPath(scene_path)
    if not prim.IsValid():
        raise ValueError(f"{scene_path} is not a valid prim. Can't set_kinematics_enable.")
    prim.GetAttribute("physics:kinematicEnabled").Set(True)


def unset_kinematics_enable(stage, scene_path):
    prim = stage.GetPrimAtPath(scene_path)
    if not prim.IsValid():
        raise ValueError(f"{scene_path} is not a valid prim. Can't unset_kinematics_enable.")
    prim.GetAttribute("physics:kinematicEnabled").Set(False)


def set_visibility(stage, prim, visibility):
    """Set visibility token of a prim.

    Args:
        stage (Usd.Stage): USD stage.
        prim (Usd.Prim): USD prim whose visibility should be set.
        visibility (str): Either 'invisible' or 'inherited'.

    Raises:
        ValueError: Argument visibility is neither 'invisible' nor 'inherited'.
    """
    if visibility not in ("invisible", "inherited"):
        raise ValueError(
            "Token `visibility` can only be either 'invisible' or 'inherited'. Currently it's"
            f" {visibility}."
        )
    vis = UsdGeom.Imageable(prim).GetVisibilityAttr()
    vis.Set(visibility)


def export_mesh(
    file_path,
    scene_path="/world/Meshes/mesh_0",
    vertices=None,
    faces=None,
    uvs=None,
    face_uvs_idx=None,
    face_normals=None,
    materials=None,
    up_axis="Y",
    time=None,
):
    r"""Export a single mesh to USD.

    Export a single mesh defined by vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional): Absolute path of mesh within the USD file scene. Must be a valid ``Sdf.Path``.
            If no path is provided, a default path is used.
        vertices (torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
    Returns:
       (Usd.Stage)

    Example:
        >>> vertices = torch.rand(3, 3)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> stage = export_mesh('./new_stage.usd', vertices=vertices, faces=faces)
    """
    assert isinstance(scene_path, str)
    if time is None:
        time = Usd.TimeCode.Default()
    if os.path.exists(file_path):
        stage = Usd.Stage.Open(file_path)
    else:
        stage = create_stage(file_path, up_axis)
    add_mesh(stage, scene_path, vertices, faces, uvs, face_uvs_idx, face_normals, time=time)

    return stage


def export_meshes(
    file_path,
    scene_paths=None,
    vertices=None,
    faces=None,
    uvs=None,
    face_uvs_idx=None,
    face_normals=None,
    up_axis="Y",
    times=None,
):
    r"""Export multiple meshes to a new USD stage.

    Export multiple meshes defined by lists vertices and faces and save the stage to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Absolute paths of meshes within the USD file scene. Must have the same number of
            paths as the number of meshes ``N``. Must be a valid Sdf.Path. If no path is provided, a default path is used.
        vertices (list of torch.FloatTensor, optional): Vertices with shape ``(num_vertices, 3)``.
        faces (list of torch.LongTensor, optional): Vertex indices for each face with shape ``(num_faces, face_size)``.
            Mesh must be homogenous (consistent number of vertices per face).
        uvs (list of torch.FloatTensor, optional): of shape ``(num_uvs, 2)``.
        face_uvs_idx (list of torch.LongTensor, optional): of shape ``(num_faces, face_size)``. If provided, `uvs` must also
            be specified.
        face_normals (list of torch.Tensor, optional): of shape ``(num_vertices, num_faces, 3)``.
        up_axis (str, optional): Specifies the scene's up axis. Choose from ``['Y', 'Z']``.
        times (list of int, optional): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
    """
    stage = create_stage(file_path, up_axis)
    mesh_parameters = {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "face_uvs_idx": face_uvs_idx,
        "face_normals": face_normals,
    }
    supplied_parameters = {k: p for k, p in mesh_parameters.items() if p is not None}
    length = len(list(supplied_parameters.values())[0])
    assert all([len(p) == length for p in supplied_parameters.values()])
    if scene_paths is None:
        if not stage.GetPrimAtPath("/world/Meshes"):
            stage.DefinePrim("/world/Meshes", "Xform")
        scene_paths = [f"/world/Meshes/mesh_{i}" for i in range(len(vertices))]
    assert len(scene_paths) == length
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    for i, scene_path in enumerate(scene_paths):
        mesh_params = {k: p[i] for k, p in supplied_parameters.items()}
        add_mesh(stage, scene_path, **mesh_params)

    return stage


class Material:
    """Abstract material definition class.
    Defines material inputs and methods to export material properties.
    """

    @abstractmethod
    def write_to_usd(
        self,
        file_path,
        scene_path,
        bound_prims=None,
        # time=None,
        texture_dir=None,
        texture_file_prefix="",
        **kwargs,
    ):
        pass

    @abstractmethod
    def read_from_usd(self, file_path, scene_path, time=None):
        pass

    @abstractmethod
    def write_to_obj(self, obj_dir=None, texture_dir=None, texture_prefix=""):
        pass

    @abstractmethod
    def read_from_obj(self, file_path):
        pass


class PBRMaterial(Material):
    """Define a PBR material using USD Preview Surface.
    Usd Preview Surface (https://graphics.pixar.com/usd/docs/UsdPreviewSurface-Proposal.html)
    is a physically based surface material definition.
    Args:
        diffuse_color (tuple of floats): RGB values for `Diffuse` parameter (typically referred to as `Albedo`
            in a metallic workflow) in the range of `(0.0, 0.0, 0.0)` to `(1.0, 1.0, 1.0)`. Default value is grey
            `(0.5, 0.5, 0.5)`.
        roughness_value (float): Roughness value of specular lobe in range `0.0` to `1.0`. Default value is `0.5`.
        metallic_value (float): Typically set to `0.0` for non-metallic and `1.0` for metallic materials. Ignored
            if `is_specular_workflow` is `True`. Default value is `0.0`.
        specular_color (tuple of floats): RGB values for `Specular` lobe. Ignored if `is_specular_workflow` is
            `False`. Default value is white `(0.0, 0.0, 0.0)`.
        diffuse_texture (torch.FloatTensor): Texture for diffuse parameter, of shape `(3, height, width)`.
        roughness_texture (torch.FloatTensor): Texture for roughness parameter, of shape `(1, height, width)`.
        metallic_texture (torch.FloatTensor): Texture for metallic parameter, of shape `(1, height, width)`.
            Ignored if  `is_specular_workflow` is `True`.
        specular_texture (torch.FloatTensor): Texture for specular parameter, of shape `(3, height, width)`.
            Ignored if `is_specular_workflow` is `False`.
        normals_texture (torch.FloatTensor): Texture for normal mapping of shape `3, height, width)`.Normals
            maps create the illusion of fine three-dimensional detail without increasing the number of polygons.
            Tensor values must be in the range of `[(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)]`.
        is_specular_workflow (bool): Determines whether or not to use a specular workflow. Default
            is `False` (use a metallic workflow).
    """

    def __init__(
        self,
        diffuse_color=(0.5, 0.5, 0.5),
        roughness_value=0.5,
        metallic_value=0.0,
        specular_color=(0.0, 0.0, 0.0),
        diffuse_texture=None,
        roughness_texture=None,
        metallic_texture=None,
        specular_texture=None,
        normals_texture=None,
        displacement_texture=None,
        is_specular_workflow=False,
    ):
        self.diffuse_color = diffuse_color
        self.roughness_value = roughness_value
        self.metallic_value = metallic_value
        self.specular_color = specular_color
        self.diffuse_texture = diffuse_texture
        self.roughness_texture = roughness_texture
        self.metallic_texture = metallic_texture
        self.specular_texture = specular_texture
        self.normals_texture = normals_texture
        self.is_specular_workflow = is_specular_workflow

        self.shaders = {
            "UsdPreviewSurface": {
                "writer": self._write_usd_preview_surface,
                "reader": self._read_usd_preview_surface,
            },
        }

    def write_to_usd(
        self,
        stage,
        usd_dir,
        scene_path,
        bound_prims=None,
        texture_dir="",
        texture_file_prefix="",
        shader="UsdPreviewSurface",
        write_to_file=True,
    ):
        # time=None,
        r"""Write material to USD.
        Textures will be written to disk in the format
        `{usd_dir}/{texture_dir}/{texture_file_prefix}{attr}.png` where `attr` is one of
        [`diffuse`, `roughness`, `metallic`, `specular`, `normals`].
        Args:
            stage (Usd.Stage): Stage.
            usd_dir (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of material within the USD file scene. Must be a valid ``Sdf.Path``.
            shader (str, optional): Name of shader to write. If not provided, use UsdPreviewSurface.
            bound_prims (list of Usd.Prim, optional): If provided, bind material to each prim.
            time (int, optional): Positive integer defining the time at which the supplied parameters correspond to.
            texture_dir (str, optional): Subdirectory to store texture files. If not provided, texture files will be
                saved in the same directory as the USD file specified by `usd_dir`.
            texture_file_prefix (str, optional): String to be prepended to the filename of each texture file.
        """
        assert (
            shader in self.shaders
        ), f"Shader {shader} is not support. Choose from {list(self.shaders.keys())}."

        writer = self.shaders[shader]["writer"]
        return writer(
            stage,
            usd_dir,
            scene_path,
            bound_prims,
            texture_dir,
            texture_file_prefix,
            write_to_file,
        )

    def _write_usd_preview_surface(
        self,
        stage,
        usd_dir,
        scene_path,
        bound_prims,
        texture_dir,
        texture_file_prefix,
        write_to_file,
    ):
        """Write a USD Preview Surface material."""
        texture_dir = Path(texture_dir).as_posix()

        material = UsdShade.Material.Define(stage, scene_path)

        shader = UsdShade.Shader.Define(stage, f"{scene_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Create Inputs
        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        specular_input = shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f)
        metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        normal_input = shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
        is_specular_workflow_input = shader.CreateInput(
            "useSpecularWorkflow", Sdf.ValueTypeNames.Int
        )

        # Set constant values
        if self.diffuse_color is not None:
            diffuse_input.Set(tuple(self.diffuse_color))
        if self.roughness_value is not None:
            roughness_input.Set(self.roughness_value)
        if self.specular_color is not None:
            specular_input.Set(tuple(self.specular_color))
        if self.metallic_value is not None:
            metallic_input.Set(self.metallic_value)
        is_specular_workflow_input.Set(int(self.is_specular_workflow))

        # Export textures abd Connect textures to shader
        if self.diffuse_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f"{texture_file_prefix}diffuse.png")
            
            if write_to_file:
                self._write_image(self.diffuse_texture, posixpath.join(usd_dir, rel_filepath))

            texture = self._add_texture_shader(
                stage,
                f"{scene_path}/diffuse_texture",
                rel_filepath,
                material=material,
                channels_out=3,
                wrap_s="repeat",
                wrap_t="repeat",
            )
            
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
            diffuse_input.ConnectToSource(inputTexture)
        if self.roughness_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f"{texture_file_prefix}roughness.png")
            
            if write_to_file:
                self._write_image(self.roughness_texture, posixpath.join(usd_dir, rel_filepath))
            
            texture = self._add_texture_shader(
                stage,
                f"{scene_path}/roughness_texture",
                rel_filepath,
                material=material,
                channels_out=1,
            )
            inputTexture = texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
            roughness_input.ConnectToSource(inputTexture)
        if self.specular_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f"{texture_file_prefix}specular.png")
            
            if write_to_file:
                self._write_image(self.specular_texture, posixpath.join(usd_dir, rel_filepath))

            texture = self._add_texture_shader(
                stage,
                f"{scene_path}/specular_texture",
                rel_filepath,
                material=material,
                channels_out=3,
            )
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
            specular_input.ConnectToSource(inputTexture)
        if self.metallic_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f"{texture_file_prefix}metallic.png")
            
            if write_to_file:
                self._write_image(self.metallic_texture, posixpath.join(usd_dir, rel_filepath))

            texture = self._add_texture_shader(
                stage,
                f"{scene_path}/metallic_texture",
                rel_filepath,
                material=material,
                channels_out=1,
            )
            inputTexture = texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
            metallic_input.ConnectToSource(inputTexture)
        if self.normals_texture is not None:
            rel_filepath = posixpath.join(texture_dir, f"{texture_file_prefix}normals.png")
            
            if write_to_file:
                self._write_image(
                    ((self.normals_texture + 1.0) / 2.0),
                    posixpath.join(usd_dir, rel_filepath),
                )

            texture = self._add_texture_shader(
                stage,
                f"{scene_path}/normals_texture",
                rel_filepath,
                material=material,
                channels_out=3,
            )
            inputTexture = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Normal3f)
            normal_input.ConnectToSource(inputTexture)

        # create Usd Preview Surface Shader outputs
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader.CreateOutput("displacement", Sdf.ValueTypeNames.Token)

        # create material
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        material.CreateDisplacementOutput().ConnectToSource(shader.GetOutput("displacement"))

        # bind material to bound prims if provided
        if bound_prims is not None:
            for prim in bound_prims:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(material)
        
        return material

    def _add_texture_shader(
        self,
        stage,
        path,
        texture_path,
        material,
        channels_out=3,
        scale=None,
        bias=None,
        wrap_s=None,
        wrap_t=None,  # , time
    ):
        assert channels_out > 0 and channels_out <= 4

        texture = UsdShade.Shader.Define(stage, path)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        if scale is not None:
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
        if bias is not None:
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)

        
        if wrap_s is not None:
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap_s)
        if wrap_t is not None:
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap_t)

        channels = ["r", "b", "g", "a"]
        for channel in channels[:channels_out]:
            texture.CreateOutput(channel, Sdf.ValueTypeNames.Float)
        
        # Attach PrimvarReader to read UV texture coordinates
        st_reader = UsdShade.Shader.Define(stage, path + "_stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
        st_input.Set("st")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(st_input)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )

        return texture

    @staticmethod
    def _read_image(path):
        img = Image.open(str(path))
        # img_tensor = (
        #     (torch.FloatTensor(img.getdata())).reshape(*img.size, -1) / 255.0
        # ).permute(2, 0, 1)
        img_tensor = np.transpose(np.asarray(img) / 255.0, axes=(2, 0, 1))
        return img_tensor

    @staticmethod
    def _write_image(img_tensor, path):
        # img_tensor_uint8 = (
        #     (img_tensor * 255.0).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        # )
        img_tensor_uint8 = np.transpose((img_tensor * 255.0).clip(0, 255), axes=(1, 2, 0)).astype(
            np.uint8
        )
        # img = Image.fromarray(img_tensor_uint8.squeeze().cpu().numpy())
        img = Image.fromarray(img_tensor_uint8)
        img.save(path)

    def read_from_usd(self, file_path, scene_path, texture_path=None, time=None):
        r"""Read material from USD.
        Args:
            file_path (str): Path to usd file (\*.usd, \*.usda).
            scene_path (str): Absolute path of UsdShade.Material prim within the USD file scene.
                Must be a valid ``Sdf.Path``.
            texture_path (str, optional): Path to textures directory. If the USD has absolute paths
                to textures, set to an empty string. By default, the textures will be assumed to be
                under the same directory as the USD specified by `file_path`.
            time (int, optional): Positive integer indicating the time at which to retrieve parameters.
        """
        if time is None:
            time = Usd.TimeCode.Default()
        if texture_path is None:
            texture_file_path = os.path.dirname(file_path)
        else:
            usd_dir = os.path.dirname(file_path)
            texture_file_path = posixpath.join(usd_dir, texture_path)
        stage = Usd.Stage.Open(file_path)
        material = UsdShade.Material(stage.GetPrimAtPath(scene_path))
        assert material

        surface_shader = material.GetSurfaceOutput().GetConnectedSource()[0]
        shader = UsdShade.Shader(surface_shader)
        if shader.GetImplementationSourceAttr().Get(time=time) == "id":
            shader_name = UsdShade.Shader(surface_shader).GetShaderId()
        else:
            raise NotImplementedError
        inputs = surface_shader.GetInputs()

        reader = self.shaders[shader_name]["reader"]
        return reader(inputs, texture_file_path, time)

    def _read_usd_preview_surface(self, inputs, texture_file_path, time):
        """Read UsdPreviewSurface material."""
        texture_file_path = Path(texture_file_path).as_posix()
        for i in inputs:
            name = i.GetBaseName()
            while i.HasConnectedSource():
                i = i.GetConnectedSource()[0].GetInputs()[0]
            value = i.Get(time=time)
            itype = i.GetTypeName()

            if "diffuse" in name.lower() or "albedo" in name.lower():
                if itype == Sdf.ValueTypeNames.Color3f:
                    self.diffuse_color = tuple(value)
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.diffuse_texture = self._read_image(fp)
            elif "roughness" in name.lower():
                if itype == Sdf.ValueTypeNames.Float:
                    self.roughness_value = value
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.roughness_texture = self._read_image(fp)
            elif "metallic" in name.lower():
                if itype == Sdf.ValueTypeNames.Float:
                    self.metallic_value = value
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.metallic_texture = self._read_image(fp)
            elif "specular" in name.lower():
                if itype == Sdf.ValueTypeNames.Color3f:
                    self.specular_color = tuple(value)
                elif itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.specular_texture = self._read_image(fp)
                self.is_specular_workflow = True
            elif "specular" in name.lower() and "workflow" in name.lower():
                if itype == Sdf.ValueTypeNames.Bool:
                    self.is_specular_workflow = value
            elif "normal" in name.lower():
                if itype == Sdf.ValueTypeNames.Asset:
                    fp = posixpath.join(texture_file_path, value.path)
                    self.normals_texture = self._read_image(fp) * 2.0 - 1.0
        return self

def create_attribute(prim, attribute_name, sdf_type, value=None, time_code=None):
    """Create an attribute if it does not exist and set value.

    Args:
        prim (Usd.Prim): USD primitive.
        attribute_name (str): Name of the attribute.
        sdf_type (Sdf.ValueTypeName): Type of the attribute.
        value (Any, optional): Value of the attribute. Defaults to None.
        time_code (Usd.TimeCode, optional): None means Usd.TimeCode.Default(). Defaults to None.

    Returns:
        Usd.Attribute: Returns the attribute.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        attribute = prim.CreateAttribute(attribute_name, sdf_type)
    if value is not None:
        attribute.Set(value, time=time_code)
    return attribute


def add_mdl_material(stage, mtl_url, mtl_name=None, mtl_base_primpath="/world/Looks/", texture_scale=None, project_uvw=True):
    """Add MDL material to a stage

    Args:
        stage (Usd.Stage): The USD stage.
        mtl_url (str): String of the material URL.
        mtl_name (str, optional): Material name. If None will use everything after last slash. Defaults to None.
        mtl_base_primpath (str, optional): The base prim path where the material will be added. Defaults to "/world/Looks/".
        texture_scale (float, optional): Texture scale. Defaults to None.
        project_uvw (bool, optional): Whether to set project_uvw to True. Defaults to True.

    Returns:
        UsdShade.Material: The MDL material.
    """
    if mtl_name is None:
        mtl_name = mtl_url.split('/')[-1].split('.')[0]

    mtl_path = Sdf.Path(f"{mtl_base_primpath}{mtl_name}")

    mtl = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)

    # MDL shaders should use "mdl" sourceType
    shader.SetSourceAsset(mtl_url, "mdl")
    shader.SetSourceAssetSubIdentifier(mtl_name, "mdl")

    if project_uvw:
        input_project_uvw = shader.CreateInput('project_uvw', Sdf.ValueTypeNames.Bool)
        input_project_uvw.Set(1)

    if texture_scale is not None:
        input_texture_scale = shader.CreateInput('texture_scale', Sdf.ValueTypeNames.Float2)
        input_texture_scale.Set((texture_scale, texture_scale))

    # MDL materials should use "mdl" renderContext
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mtl.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    return mtl

def bind_material_to_prims(stage, material, prim_paths):
    """Bind materials to prims.

    Args:
        stage (Usd.Stage): The USD stage.
        material (UsdShade.Material): The material.
        prim_paths (list[str]): List of prim paths to bind the material to.
    """
    
    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(material)
    