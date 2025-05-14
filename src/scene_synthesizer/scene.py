# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""This module contains the Scene class which is the most important class in scene_synth used to construct your desired scene.
"""

# Standard Library
import copy
import functools
import itertools
import json
import re
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass as _dataclass
from typing import Optional

# Third Party
import numpy as np
import trimesh
import trimesh.path
import trimesh.transformations as tra
import trimesh.viewer

# Local Folder
from . import utils
from .assets import Asset, BoxAsset, BoxMeshAsset, BoxWithHoleAsset, TrimeshAsset, TrimeshSceneAsset
from .exchange import export
from .utils import log
from .constants import EDGE_KEY_METADATA, DEFAULT_JOINT_LIMIT_LOWER, DEFAULT_JOINT_LIMIT_UPPER

try:
    # Third Party
    from pyglet.app import run as _pyglet_app_run
except BaseException as E:
    _pyglet_app_run = utils.late_bind_exception(E)


@_dataclass
class SupportSurface:
    """A class for holding information about a support surface in the scene."""

    polygon: trimesh.path.polygons.Polygon
    facet_index: int
    node_name: str
    transform: np.ndarray
    coverage: Optional[float] = None
    covered: Optional[bool] = None


@_dataclass
class Container:
    """A class for holding information about a volume in the scene."""

    geometry: trimesh.Trimesh
    node_name: str
    transform: np.ndarray
    support_surface: Optional[SupportSurface] = None

class Scene(object):
    """Represents a scene, which is a scene graph, where nodes can have geometry attached and edges are transforms, potentially articulated by joints.
    Objects are subgraphs of the scene graph. A scene is constructed by adding objects to it. An object is considered an instance of an asset.
    A scene can have semantic information about specific surfaces, volumes and geometries.
    """

    def __init__(
        self,
        base_frame="world",
        trimesh_scene=None,
        seed=None,
        keep_collision_manager_synchronized=True,
        share_geometry=False,
        default_use_collision_geometry=True,
    ):
        """Create a new empty scene.

        Args:
            base_frame (str, optional): Name of the base frame. Defaults to "world".
            trimesh_scene (trimesh.Scene, optional): The trimesh scene object that this scene is based on. Defaults to None.
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
            keep_collision_manager_synchronized (bool, optional): Whether to keep the collision manager in sync (after each object add or removal). Defaults to True.
            share_geometry (bool, optional): Whether to share geometry between objects if hashes are the same. Can improve performance for large scenes with multiple objects of the same kind. Note, that unintended side effects can happen since e.g. visual properties are not part of the hash. Defaults to False.
            default_use_collision_geometry (bool or None, optional): The default value for use_collision_geometry when adding objects to the scene and not specifically set. Can be either True, False or None. Defaults to True.
        """
        self._rng = np.random.default_rng(seed)

        self._scene = (
            trimesh.Scene(base_frame=base_frame) if trimesh_scene is None else trimesh_scene
        )

        self._initialize_metadata()

        self._collision_manager = None
        self._collision_manager_hash = None
        self._keep_collision_manager_synchronized = keep_collision_manager_synchronized
        self._share_geometry = share_geometry
        self._default_use_collision_geometry = default_use_collision_geometry
        self.synchronize_collision_manager(reset=True)

    def _initialize_metadata(self):
        """Initialize scene's metadata, if not present."""
        initial_values = {
            "support_polygons": {},
            "containers": {},
            "parts": {},
            "object_nodes": {},
            "object_geometry_nodes": {},
        }
        for k, v in initial_values.items():
            if k not in self._scene.metadata:
                self._scene.metadata[k] = v

    @property
    def metadata(self):
        """The metadata of the scene in form of a dict.
        This holds all information that is not part of the trimesh.Scene object self.scene.

        Returns:
            dict: Metadata of the scene.
        """
        return self._scene.metadata

    @property
    def graph(self):
        """The scene graph.

        Returns:
            trimesh.scene.transforms.SceneGraph: The scene graph.
        """
        return self._scene.graph

    @property
    def scene(self):
        """The trimesh scene object that contains most scene information.

        Returns:
            trimesh.Scene: The scene.
        """
        return self._scene

    @property
    def geometry(self):
        """The trimesh geometries that this scene contains.

        Returns:
            OrderedDict[str, trimesh.Trimesh]: A dict of identifiers and trimesh.Trimesh's.
        """
        return self._scene.geometry

    @classmethod
    def single_object_scene(cls, asset, obj_id=None, base_frame="world", seed=None, **kwargs):
        """Creates a scene and adds an asset. Only for convenience.

        Args:
            asset (scene.Asset): Object asset.
            obj_id (str): Object identifier.
            base_frame (str, optional): Name of the base frame of the scene. Defaults to "world".
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
            **kwargs: Additional keyword arguments that will be piped to the add_object method.

        Returns:
            scene.Scene: A scene object.
        """
        my_scene = Scene(base_frame=base_frame, seed=seed)
        my_scene.add_object(asset=asset, obj_id=obj_id, **kwargs)

        return my_scene

    def copy(self):
        """Return a deep copy of the current scene.

        Returns:
            (scene.Scene): Copy of the current scene.
        """
        scene_copy = Scene(base_frame=self.graph.base_frame, trimesh_scene=self.scene.copy())
        scene_copy._scene.metadata = copy.deepcopy(self.metadata)

        return scene_copy

    def invalidate_scenegraph_cache(self):
        """Invalidate cache of the trimesh scene graph and run forward kinematics with current configuration."""
        utils.forward_kinematics(self._scene)
        utils.invalidate_scenegraph_cache(self._scene)

    def add_base_frame(self, name, **kwargs):
        """Add a new root node to the scene graph.

        Args:
            name (str): New name of root node / base frame.
            **matrix: Homogenous transform between new base frame and previous one.

        Raises:
            ValueError: If node name already exists in the scene graph.
        """
        if name in self._scene.graph.nodes:
            raise ValueError(f"Node '{name}' already exists in scene.")

        self._scene.graph.update(frame_from=name, frame_to=self._scene.graph.base_frame, **kwargs)
        self._scene.graph.base_frame = name

        self.invalidate_scenegraph_cache()

    def remove_base_frame(self, keep_transform=True):
        """Remove root of scene graph. Only works, if current root has one child.
        Its child will become the new root of the scene graph.

        Args:
            keep_transform (bool, optional): Fold the potentially removed transform into the edge of the roots child. Defaults to True.

        Raises:
            ValueError: Current root doesn't have one child.
        """
        children = self._scene.graph.transforms.children[self._scene.graph.base_frame]
        if len(children) != 1:
            raise ValueError(
                "Can't remove base frame since it has"
                f" {len(self._scene.graph.transforms.children[self._scene.graph.base_frame])} != 1"
                " children. Resulting graph wouldn't be a tree."
            )

        if len(self._scene.graph.nodes) <= 2:
            raise ValueError(
                "Can't remove base frame since the graph has less than 2 edges and the edge"
                " attributes of the removed edge couldn't be added to another one."
            )

        T_child = self._scene.graph.get(children[0])[0]
        attrib = self.graph.transforms.edge_data[(self._scene.graph.base_frame, children[0])]
        attrib.pop("matrix", None)
        self._scene.graph.transforms.remove_node(self._scene.graph.base_frame)

        self._scene.graph.base_frame = children[0]

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

        new_child = self._scene.graph.transforms.children[self._scene.graph.base_frame][0]
        self._scene.graph.update(new_child, **attrib)
        if keep_transform:
            T = self._scene.graph.get(new_child)[0]
            self._scene.graph.update(new_child, matrix=T_child @ T)

    def add_base_frame_transform(self, matrix):
        """Transform the base_frame of the scene, i.e., move the entire scene.

        Args:
            matrix (np.ndarray): 4x4 homogenous transform
        """
        base_frame_children = self._scene.graph.transforms.children[self._scene.graph.base_frame]

        for c in base_frame_children:
            T = self._scene.graph.get(frame_to=c)[0]

            # Keep existing meta information
            metadata = self.graph.transforms.edge_data.get((self._scene.graph.base_frame, c), {})
            metadata.update({'matrix': matrix @ T})

            self._scene.graph.update(
                frame_from=self.graph.base_frame, frame_to=c, **metadata
            )

        self.invalidate_scenegraph_cache()

    def is_watertight(self, obj_id, layers=None):
        """Check wheter object geometries are all watertight.

        Args:
            obj_id (str): Object identifier.
            layers (list[str]): Layers.

        Raises:
            ValueError: Unknown object identifier.
            ValueError: No geometries found.

        Returns:
            bool: Whether object geometries are all watertight.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't check watertightness.")
        
        geom_names = [self.graph[geom_node][1] for geom_node in self.metadata["object_geometry_nodes"][obj_id]]
        geometries = [
            geom_name
            for geom_name in geom_names
            if layers is None
            or self.scene.geometry[geom_name].metadata.get("layer", None) in layers
        ]

        if len(geometries) == 0:
            raise ValueError(
                f"Object '{obj_id}' has no geometries in layers {layers}. Can't check"
                " watertightness."
            )

        return all(self.scene.geometry[geom_name].is_watertight for geom_name in geometries)

    def get_volume(self, obj_id):
        """Get volume of a single object in the scene.

        Args:
            obj_id (str): Object identifier.

        Raises:
            ValueError: Unknown object identifier.

        Returns:
            float: Volume of the object.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't get volume.")

        geometries = [self.graph[gn][1] for gn in self.metadata["object_geometry_nodes"][obj_id]]

        return sum(
            utils.get_mass_properties(self.scene.geometry[geom_name])[3] for geom_name in geometries
        )

    def set_mass(self, obj_id, mass):
        """Sets mass of a single object in the scene. Note, that in case the object consists of multiple geometries, masses are distributed according to volume. If meshes are not watertight this migth lead to unexpected distributions.
        Internally, this method sets the density according to the mass to ensure the two are always consistent.

        Args:
            obj_id (str): Object identifier.
            mass (float): Mass of the object.

        Raises:
            ValueError: Unknown object identifier.
            ValueError: Object has no geometries.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't set mass.")

        geometries = [self.graph[gn][1] for gn in self.metadata["object_geometry_nodes"][obj_id]]

        if len(geometries) == 0:
            raise ValueError(f"Object '{obj_id}' has no geometries. Can't set mass.")

        total_volume = self.get_volume(obj_id)
        for geom_name in geometries:
            geom = self.scene.geometry[geom_name]
            # We simplify the following:
            # geom_mass = mass * geom.volume / total_volume
            # geom_density = geom_mass / geom.volume
            # to:
            geom.density = mass / total_volume

    def get_mass(self, obj_id):
        """Get mass of a single object in the scene.

        Args:
            obj_id (str): Object identifier.

        Raises:
            ValueError: Unknown object identifier.

        Returns:
            float: Mass of the object.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't get mass.")

        geometries = [self.graph[gn][1] for gn in self.metadata["object_geometry_nodes"][obj_id]]
        return sum(
            utils.get_mass_properties(self.scene.geometry[geom_name])[0] for geom_name in geometries
        )

    def set_density(self, obj_id, density):
        """Set density of a single object in the scene.

        Args:
            obj_id (str): Object identifier.
            density (float): Density of the object.

        Raises:
            ValueError: Unkown object identifier.
            ValueError: Object has no geometries.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't set density.")

        geometries = [self.graph[gn][1] for gn in self.metadata["object_geometry_nodes"][obj_id]]

        if len(geometries) == 0:
            raise ValueError(f"Object '{obj_id}' has no geometries. Can't set density.")

        for geom_name in geometries:
            geom = self.scene.geometry[geom_name]
            geom.density = density

    def get_density(self, obj_id):
        """Get density of a single object in the scene.

        Args:
            obj_id (str): Object identifier.

        Raises:
            ValueError: Unknown object identifier.
            ValueError: Object has no geometries.

        Returns:
            float: Density of the object.
        """
        if obj_id not in self.metadata["object_nodes"]:
            raise ValueError(f"Unknown object '{obj_id}'. Can't get density.")

        geometries = [self.graph[gn][1] for gn in self.metadata["object_geometry_nodes"][obj_id]]
        if len(geometries) == 0:
            raise ValueError(f"Object '{obj_id}' has no geometries. Can't get density.")

        return self.scene.geometry[geometries[0]].density

    def get_bounds(self, query=None, frame=None):
        """Return bounds for subscene defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this scene, or list of objects. None means entire scene. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.

        Returns:
            np.ndarray: A 2x3 matrix of minimum and maximum coordinates for each dimension.
        """
        # select geometry according to query
        if query is None:
            node_names = set(self.scene.graph.nodes_geometry)
        elif type(query) is list or type(query) is tuple:
            node_names = []
            for k in query:
                if k in self.metadata["object_nodes"]:
                    node_names.extend(self.metadata["object_nodes"][k])
                else:
                    node_names.append(k)
            node_names = set(self.scene.graph.nodes_geometry).intersection(set(node_names))
        else:
            node_names = utils.select_sublist(
                query=query, all_items=self.scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        all_bounds = []
        for n in node_names:
            T, geomn = self.graph.get(n)
            bounds_w = trimesh.transform_points(self.scene.geometry[geomn].bounds, T)
            all_bounds.append(bounds_w)
        all_bounds = np.vstack(all_bounds)

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            all_bounds = trimesh.transform_points(all_bounds, T)

        return np.array([np.min(all_bounds, axis=0), np.max(all_bounds, axis=0)])

    def get_extents(self, query=None, frame=None):
        """Return extents for subscene defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this scene, or list of objects. None means entire scene. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the extents of each dimension.
        """
        return np.diff(self.get_bounds(query=query, frame=frame), axis=0)[0]

    def get_center_mass(self, query=None, frame=None):
        """Return center of mass for subscene defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this scene, or list of objects. None means entire scene. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the center of mass of the queried subscene.
        """
        if query is None:
            node_names = set(self.scene.graph.nodes_geometry)
        elif type(query) is list or type(query) is tuple:
            node_names = []
            for k in query:
                if k in self.metadata["object_nodes"]:
                    node_names.extend(self.metadata["object_nodes"][k])
                else:
                    node_names.append(k)
            node_names = set(self.scene.graph.nodes_geometry).intersection(set(node_names))
        else:
            node_names = utils.select_sublist(
                query=query, all_items=self.scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        result = utils.center_mass(trimesh_scene=self.scene, node_names=node_names)

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            result = tra.translation_from_matrix(T @ tra.translation_matrix(result))

        return result

    def get_centroid(self, query=None, frame=None):
        """Return centroid for subscene defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this scene, or list of objects. None means entire scene. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the centroid of the queried subscene.
        """
        if query is None:
            node_names = set(self.scene.graph.nodes_geometry)
        elif type(query) is list or type(query) is tuple:
            node_names = []
            for k in query:
                if k in self.metadata["object_nodes"]:
                    node_names.extend(self.metadata["object_nodes"][k])
                else:
                    node_names.append(k)
            node_names = set(self.scene.graph.nodes_geometry).intersection(set(node_names))
        else:
            node_names = utils.select_sublist(
                query=query, all_items=self.scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        total_area = 0.0
        total_centroid = np.zeros(3)
        for n in node_names:
            T, geomn = self.graph.get(n)
            area = self.scene.geometry[geomn].area
            total_centroid += area * (T @ np.append(self.scene.geometry[geomn].centroid, 1.0))[:3]
            total_area += area
        result = total_centroid / total_area

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            result = tra.translation_from_matrix(T @ tra.translation_matrix(result))

        return result

    def get_reference_frame(self, xyz, query=None, frame=None):
        """Return reference frame for subscene defined through nodes selected by query.

        Args:
            xyz (tuple[str]): A 3-tuple/list of ['top', 'center', 'bottom', 'com', 'centroid']
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this scene, or list of objects. None means entire scene. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.

        Raises:
            ValueError: Unknown reference string.

        Returns:
            np.ndarray: A 4x4 homogenous matrix.
        """
        # This seems to be called from one other place.
        translation = np.zeros(3)

        if any(keyword in xyz for keyword in ("top", "center", "bottom", "left", "right", "front", "back")):
            bounds = self.get_bounds(query=query, frame=frame)
        if "com" in xyz:
            center_mass = self.get_center_mass(query=query, frame=frame)
        if "centroid" in xyz:
            centroid = self.get_centroid(query=query, frame=frame)

        for i, alignment in enumerate([xyz[0].lower(), xyz[1].lower(), xyz[2].lower()]):
            if alignment in ("top", "right", "back"):
                translation[i] = bounds[1, i]
            elif alignment in ("bottom", "left", "front"):
                translation[i] = bounds[0, i]
            elif alignment == "center":
                translation[i] = (bounds[0, i] + bounds[1, i]) / 2.0
            elif alignment == "com":
                translation[i] = center_mass[i]
            elif alignment == "centroid":
                translation[i] = centroid[i]
            else:
                raise ValueError(
                    f"Unknown reference along axis {i}: '{alignment}'.  Use either: top/right/back,"
                    " bottom/left/front, center, com, centroid"
                )

        return tra.translation_matrix(translation)

    def remove_object(self, obj_id_regex):
        """Remove object(s) from the scene.

        Args:
            obj_id_regex (str): A regular expression (or just an object identifier). All objects that match this string will be removed.

        Raises:
            ValueError: No object identifier matches the regular expression string.
        """
        if utils.is_regex(obj_id_regex):
            x = re.compile(obj_id_regex)
            obj_ids = list(filter(x.search, self.metadata["object_nodes"].keys()))
        else:
            obj_ids = [obj_id_regex] if obj_id_regex in self.metadata["object_nodes"].keys() else []

        if len(obj_ids) == 0:
            raise ValueError(f"No objects with id '{obj_id_regex}' in scene.")

        log.debug(f"Deleting {len(obj_ids)} objects: {obj_ids}")
        
        remove_nodes = []
        remove_geometries = []
        for obj_id in obj_ids:
            log.debug(f"Removing object {obj_id}")

            # remember geometries and nodes to remove
            remove_nodes.extend(self._scene.metadata["object_nodes"][obj_id])
            remove_geometries.extend(self.graph[gn][1] for gn in self._scene.metadata["object_geometry_nodes"][obj_id])

            # remove support_polygons
            for k in tuple(self._scene.metadata["support_polygons"]):
                indices_to_remove = []
                for i, support_data in enumerate(self._scene.metadata["support_polygons"][k]):
                    if support_data.node_name in self._scene.metadata["object_nodes"][obj_id]:
                        indices_to_remove.append(i)

                for index in sorted(indices_to_remove, reverse=True):
                    del self._scene.metadata["support_polygons"][k][index]

                # delete empty entries
                if len(self._scene.metadata["support_polygons"][k]) == 0:
                    del self._scene.metadata["support_polygons"][k]

            # remove containers
            for k in tuple(self._scene.metadata["containers"]):
                indices_to_remove = []
                for i, support_data in enumerate(self._scene.metadata["containers"][k]):
                    if support_data.node_name in self._scene.metadata["object_nodes"][obj_id]:
                        indices_to_remove.append(i)

                for index in sorted(indices_to_remove, reverse=True):
                    del self._scene.metadata["containers"][k][index]

                # delete empty entries
                if len(self._scene.metadata["containers"][k]) == 0:
                    del self._scene.metadata["containers"][k]

            # remove parts
            for k in tuple(self._scene.metadata["parts"]):
                indices_to_remove = []
                for i, geom_node_name in enumerate(self._scene.metadata["parts"][k]):
                    if (
                        geom_node_name
                        in self._scene.metadata["object_geometry_nodes"][obj_id]
                    ):
                        indices_to_remove.append(i)

                for index in sorted(indices_to_remove, reverse=True):
                    del self._scene.metadata["parts"][k][index]

                # delete empty entries
                if len(self._scene.metadata["parts"][k]) == 0:
                    del self._scene.metadata["parts"][k]

            del self._scene.metadata["object_nodes"][obj_id]
            del self._scene.metadata["object_geometry_nodes"][obj_id]
        
        # remove nodes
        [self._scene.graph.transforms.remove_node(node_name) for node_name in set(remove_nodes)]
        
        # remove unreferenced geometry
        [self.geometry.pop(geom_name, None) for geom_name in set(remove_geometries) if geom_name not in self.graph.geometry_nodes.keys()]

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

        self.synchronize_collision_manager()

    def collapse_nodes(self, keep_object_root_nodes=True):
        """Remove nodes from the scene graph that do not have geometries attached and no adjacent edges with joints.

        Args:
            keep_object_root_nodes (bool, optional): Whether to keep nodes that are object roots. Defaults to True.
                    Note: USD export relies on a node with the same name as the object.
                    The result could create a graph that can't be exported to USD, if keep_object_root_nodes=False.
        """
        camera_node = self._scene.camera.name
        light_nodes = [x.name for x in self._scene.lights]

        deleted_nodes = []

        # Copy nodes so that they are not deleted while iterating over them
        node_list = list(self._scene.graph.nodes)
        for n in node_list:
            if n == self._scene.graph.base_frame or n in light_nodes or n == camera_node:
                # Do not remove base frame node, camera, or lights
                continue

            if keep_object_root_nodes and n in self.metadata["object_nodes"].keys():
                continue

            parent = self._scene.graph.transforms.parents[n]
            edge_data = self._scene.graph.transforms.edge_data[(parent, n)]
            articulated_node = (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
            )
            if n not in self._scene.graph.geometry_nodes and not articulated_node:
                # node does not have geometry attached to it and will be removed
                deleted_nodes.append(n)

                if n in self._scene.graph.transforms.children:
                    T_parent = self._scene.graph.get(frame_from=parent, frame_to=n)[0]

                    # node has leaves and must rewire graph accordingly
                    children = {
                        x: self._scene.graph.get(frame_from=parent, frame_to=x)[0]
                        for x in self._scene.graph.transforms.children[n]
                    }
                    # convert joint origin transforms
                    children_extras = {}
                    for x in self._scene.graph.transforms.children[n]:
                        edge_data = self._scene.graph.transforms.edge_data[(n, x)]
                        children_extras[x] = {EDGE_KEY_METADATA: edge_data.get(EDGE_KEY_METADATA, None)}
                        if (
                            EDGE_KEY_METADATA in edge_data
                            and edge_data[EDGE_KEY_METADATA] is not None
                            and "joint" in edge_data[EDGE_KEY_METADATA]
                        ):
                            if "origin" in edge_data[EDGE_KEY_METADATA]["joint"]:
                                children_extras[x][EDGE_KEY_METADATA]["joint"]["origin"] = (
                                    T_parent @ children_extras[x][EDGE_KEY_METADATA]["joint"]["origin"]
                                )

                    self._scene.graph.transforms.remove_node(n)

                    # connect all children to the node's parent
                    for child_node, transform in children.items():
                        self._scene.graph.transforms.add_edge(
                            parent, child_node, matrix=transform, **children_extras[child_node]
                        )
                else:
                    # it's a leaf node and can be removed without problems
                    self._scene.graph.transforms.remove_node(n)

                # invalidate cache of SceneGraph
                utils.invalidate_scenegraph_cache(self._scene)

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

        for n in deleted_nodes:
            for k in self.metadata["object_nodes"]:
                if n in self.metadata["object_nodes"][k]:
                    self.metadata["object_nodes"][k].remove(n)

    def rename_joints(self, mappings):
        """Rename joints in the scene graph.

        Args:
            mappings (list[(str, str)]): List of tuples of joints to be renamed and their new names respectively.
        """
        # check if mapping is valid
        old_names = [x for x, _ in mappings]
        new_names = [y for _, y in mappings]

        joint_names = self.get_joint_names(include_fixed_floating_joints=True)
        for old_name, new_name in mappings:
            if old_names.count(old_name) > 1:
                raise ValueError(
                    "Mapping for rename_joints needs to be unique. Currently contains multiple"
                    f" mappings for joint `{old_name}`."
                )

            if new_names.count(new_name) > 1:
                raise ValueError(
                    "Mapping for rename_joints needs to be valid. Currently contains multiple"
                    f" mappings to same name `{new_name}`."
                )

            if not old_name in joint_names:
                raise ValueError(
                    f"Mapping for rename_joints needs to be valid. Joint {old_name} does not exist"
                    " in the scene."
                )

            if new_name in joint_names:
                raise ValueError(
                    f"Mapping for rename_joints needs to be valid. Joint {new_name} already exists"
                    " in the scene."
                )

        scene_edge_data = self._scene.graph.transforms.edge_data
        joint_map = {}
        for k in scene_edge_data:
            edge_data = scene_edge_data[k]
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
            ):
                joint_data = edge_data[EDGE_KEY_METADATA]["joint"]
                joint_map[joint_data["name"]] = k

        # Apply the name mapping
        for old_name, new_name in mappings:
            scene_edge_data[joint_map[old_name]][EDGE_KEY_METADATA]["joint"]["name"] = new_name

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

    def rename_nodes(self, mappings):
        """Rename nodes in the scene graph.

        Args:
            mappings (list[(str, str)]): List of tuples of nodes to be renamed and their new names respectively.
        """
        # make sure node is in the graph
        # and new name is not yet existent
        for old_name, new_name in mappings:
            if old_name not in self._scene.graph.nodes or new_name in self._scene.graph.nodes:
                raise ValueError(
                    f"Mapping {old_name}->{new_name} not possible. Since new name already exists or"
                    " old name doesn't."
                )

        for old_name, new_name in mappings:
            # Fix all metadata
            for obj_id, obj_node_names in self.metadata["object_nodes"].items():
                if old_name in obj_node_names:
                    self.metadata["object_nodes"][obj_id] = [
                        n for n in obj_node_names if n != old_name
                    ] + [new_name]

            # 'containers'
            # 'parts'
            # 'support_polygons'

            is_base_frame = self.graph.base_frame == old_name

            # remember node data
            parent = (
                self._scene.graph.transforms.parents[old_name]
                if old_name in self._scene.graph.transforms.parents
                else None
            )
            children = (
                self._scene.graph.transforms.children[old_name]
                if old_name in self._scene.graph.transforms.children
                else []
            )
            attr_node = copy.deepcopy(self._scene.graph.transforms.node_data[old_name])
            attr_edge = (
                copy.deepcopy(self._scene.graph.transforms.edge_data[(parent, old_name)])
                if parent
                else None
            )

            # add new node and delete old one
            self._scene.graph.transforms.node_data[new_name] = attr_node
            self._scene.graph.transforms.node_data.pop(old_name)

            if parent:
                # add new incoming edge, delete old one
                self._scene.graph.transforms.edge_data[(parent, new_name)] = attr_edge
                self._scene.graph.transforms.edge_data.pop((parent, old_name))

                # update parents
                self._scene.graph.transforms.parents[new_name] = parent
                self._scene.graph.transforms.parents.pop(old_name)

            # add new outgoing edge, delete old ones, update parents
            for child in children:
                attr_edge = self._scene.graph.transforms.edge_data[(old_name, child)].copy()
                self._scene.graph.transforms.edge_data[(new_name, child)] = attr_edge
                self._scene.graph.transforms.edge_data.pop((old_name, child))

                self._scene.graph.transforms.parents[child] = new_name

            # check for base frame
            if is_base_frame:
                self.graph.base_frame = new_name

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

    def rename_geometries(self, mappings):
        """Rename geometries in the scene graph.

        Args:
            mappings (list[(str, str)]): List of tuples of nodes to be renamed and their new names respectively.
        """
        mapping_dict = {}
        for old_name, new_name in mappings:
            # make sure node is in the graph
            # and new name is not yet existent
            if old_name not in self.scene.geometry or new_name in self.scene.geometry:
                raise ValueError(
                    f"Mapping {old_name}->{new_name} not possible. Since new name already exists or"
                    " old name doesn't."
                )

            mapping_dict[old_name] = new_name

        for old_name, new_name in mappings:
            for node in self.graph.geometry_nodes[old_name]:
                # change node data
                attrib = self.graph.transforms.node_data[node]
                attrib["geometry"] = new_name

                # change edge data (not sure why this is redundant)
                if node in self.graph.transforms.parents:
                    parent_node = self.graph.transforms.parents[node]
                    attrib = self.graph.transforms.edge_data[(parent_node, node)]
                    attrib["geometry"] = new_name

            # Fix metadata
            for obj_id, obj_geom_names in self.metadata["object_geometry_nodes"].items():
                if old_name in obj_geom_names:
                    self.metadata["object_geometry_nodes"][obj_id] = [
                        n for n in obj_geom_names if n != old_name
                    ] + [new_name]

        # change geometry store
        self.scene.geometry = OrderedDict(
            [
                (mapping_dict[k], v) if k in mapping_dict else (k, v)
                for k, v in self.scene.geometry.items()
            ]
        )

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

    def topological_sorted_object_nodes(self):
        """Returns a list of object node names that is ordered topologically:
        Leaf --> .... --> Root (scene.base_frame).

        Returns:
            list(str): List of object node names in topolical order.
        """
        obj_nodes = list(self.metadata["object_nodes"].keys())
        topo_sorted = list(trimesh.graph.nx.topological_sort(self.graph.to_networkx()))[::-1]

        return [obj_node for obj_node in topo_sorted if obj_node in obj_nodes]

    def flatten(self):
        """Flatten the scene graph such that all object nodes are children of the base_frame."""
        for obj_id in self.topological_sorted_object_nodes():
            parent = self.graph.transforms.parents[obj_id]

            if parent != self.graph.base_frame:
                T = self.get_transform(obj_id)

                attr_edge = copy.deepcopy(self._scene.graph.transforms.edge_data[(parent, obj_id)])
                self.graph.transforms.edge_data.pop((parent, obj_id))
                self.graph.transforms.children[parent].remove(obj_id)

                attr_edge["matrix"] = T
                self.graph.transforms.edge_data[(self.graph.base_frame, obj_id)] = attr_edge

                self.graph.transforms.parents[obj_id] = self.graph.base_frame
                self.graph.transforms.children[self.graph.base_frame].append(obj_id)

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

    def simplify_node_names(self):
        """Change node names such that all names of the format {namespace}/{identifier} are simplified to {namespace} if only one node with this namespace exists."""
        # collect all namespaces to check if we can simplify
        namespaces = {}
        for n in self._scene.graph.nodes:
            if "/" in n:
                ns = n.split("/")[0]
                if ns in namespaces:
                    namespaces[ns].append(n)
                else:
                    namespaces[ns] = [n]

        mappings = []
        for ns, names in namespaces.items():
            if len(names) == 1 and ns not in self._scene.graph.nodes:
                # Only one name is in that namespace which means we can simplify it
                mappings.append((names[0], ns))

        self.rename_nodes(mappings=mappings)

    def add_walls(
        self,
        dimensions,
        thickness=0.15,
        overhang=0.0,
        offset=0.0,
        hole_extents=(0, 0),
        hole_offset=(0, 0),
        use_primitives=True,
        object_ids=None,
        use_collision_geometry=True,
        joint_type=None,
    ):
        """Add walls/box assets to any combination of the six faces of the scene volume.

        Args:
            dimensions (list[str]): In which dimension to add a wall. Any subset of ['x', '-x', 'y', '-y', 'z', '-z'].
            thickness (float, optional): Thickness of the wall. Defaults to 0.15.
            overhang (float, optional): The amount of overhang of the wall. Can be a single scalar or a len(dimensions)-dim list. Defaults to 0.0.
            offset (float, optional): The distance between wall and closest scene geometry. Can be a single scalar or a len(dimensions)-dim list. Defaults to 0.0.
            hole_extents (list[float], optional): 2-dim extents of the hole in the wall. Defaults to (0, 0).
            hole_offset (tuple[float], optional): 2-dim position offset of the hole in the wall. Defaults to (0, 0).
            use_primitives (bool, optional): Whether to create a mesh or use a primitive. Defaults to True.
            object_ids (list[str], optional): A list of object names used for adding walls, needs to have same length as dimensions. If None, names will be "wall_{dim}". Defaults to None
            use_collision_geometry (bool, optional): Defaults to True.
            joint_type (str, optional): Defaults to None.

        Raises:
            ValueError: If overhang is not a float, or a list of length len(dimensions).
            ValueError: If object_ids is not None and len(object_ids) != len(dimensions).
        """
        overhangs = None
        offsets = None
        dim_to_index = {'x': 0, '-x': 1, 'y': 2, '-y': 3, 'z': 4, '-z': 5}

        if isinstance(overhang, float):
            overhangs = [overhang] * 6
        elif isinstance(overhang, list) and len(overhang) == len(dimensions):
            overhangs = [0.0] * 6
            for i, d in enumerate(dimensions):
                overhangs[dim_to_index[d]] = overhang[i]

        if overhangs is None:
            raise ValueError("Overhang needs to be a float or a list with len(dimensions) floats.")
        
        if isinstance(offset, float):
            offsets = [offset] * 6
        elif isinstance(offset, list) and len(offset) == len(dimensions):
            offsets = [0.0] * 6
            for i, d in enumerate(dimensions):
                offsets[dim_to_index[d]] = offset[i]
        
        if offsets is None:
            raise ValueError("Offset needs to be a float or a list with len(dimensions) floats.")
        offsets = np.array(offsets)

        if object_ids is not None and len(object_ids) != len(dimensions):
            raise ValueError("If object_ids is set it needs to have as many elements as dimensions.")

        scene_bounds = self.get_bounds()
        scene_extents = self.get_extents()

        scene_bounds[0] -= offsets[1::2]
        scene_bounds[1] += offsets[0::2]

        scene_extents += (offsets[1::2] + offsets[0::2])

        extents = {
            "x": [
                thickness,
                scene_extents[1] + overhangs[2] + overhangs[3],
                scene_extents[2] + overhangs[4] + overhangs[5],
            ],
            "y": [
                scene_extents[0] + overhangs[0] + overhangs[1],
                thickness,
                scene_extents[2] + overhangs[4] + overhangs[5],
            ],
            "z": [
                scene_extents[0] + overhangs[0] + overhangs[1],
                scene_extents[1] + overhangs[2] + overhangs[3],
                thickness,
            ],
        }
        
        if all(hole_extents):  # check if hole is not zero
            hole_extents_per_dim = {
                "x": [None, hole_extents[0], hole_extents[1]],
                "y": [hole_extents[0], None, hole_extents[1]],
                "z": [None, hole_extents[0], hole_extents[1], None],
            }
        for k in list(extents.keys()):
            extents[f"-{k}"] = extents[k]

            if all(hole_extents):
                hole_extents_per_dim[f"-{k}"] = hole_extents_per_dim[k]

        if object_ids is None:
            object_ids = [f"wall_{dim}" for dim in dimensions]

        for dim, obj_id in zip(dimensions, object_ids):
            if all(hole_extents):
                wall = BoxWithHoleAsset(
                    *extents[dim],
                    *hole_extents_per_dim[dim],
                    hole_offset=hole_offset,
                    use_primitives=use_primitives,
                )
            else:
                if use_primitives:
                    wall = BoxAsset(extents=extents[dim])
                else:
                    wall = BoxMeshAsset(extents=extents[dim])

            if dim == 'x':
                transform = tra.translation_matrix(scene_bounds[1] * np.array([1.0, 0.0, 0.0]) + [thickness/2.0, 0, 0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0, 0.5, 0.5]))
            elif dim == '-x':
                transform = tra.translation_matrix(scene_bounds[0] * np.array([1.0, 0.0, 0.0]) - [thickness/2.0, 0, 0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0, 0.5, 0.5]))
            elif dim == 'y':
                transform = tra.translation_matrix(scene_bounds[1] * np.array([0.0, 1.0, 0.0]) + [0, thickness/2.0, 0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0.5, 0, 0.5]))
            elif dim == '-y':
                transform = tra.translation_matrix(scene_bounds[0] * np.array([0.0, 1.0, 0.0]) - [0, thickness/2.0, 0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0.5, 0, 0.5]))
            elif dim == 'z':
                transform = tra.translation_matrix(scene_bounds[1] * np.array([0.0, 0.0, 1.0]) + [0, 0, thickness/2.0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0.5, 0.5, 0]))
            elif dim == '-z':
                transform = tra.translation_matrix(scene_bounds[0] * np.array([0.0, 0.0, 1.0]) - [0, 0, thickness/2.0] + (scene_bounds[1] + scene_bounds[0]) * np.array([0.5, 0.5, 0]))

            self.add_object(
                wall,
                obj_id,
                transform=transform,
                use_collision_geometry=use_collision_geometry,
                joint_type=joint_type,
            )

    def distance_matrix_geometry(self, nodes_geometry=None, return_names=False):
        """Calculate pair-wise distances of the specified geometries in the scene.

        Args:
            nodes_geometry (list[str], optional): List of geometry nodes to include. None includes all geometry nodes. Defaults to None.
            return_names (bool, optional): Whether to return the geometry node names that each row/column refers to. Defaults to False.

        Returns:
            np.ndarray: NxN matrix of distances where N = len(nodes_geometry).
            list[str]: List of geometry node names that each row/column in the matrix refers to. Only returned if return_names == True.
        """
        # create collision managers for distance checks
        if nodes_geometry is None:
            nodes_geometry = self.graph.nodes_geometry
        
        collision_managers = []
        for n in nodes_geometry:
            T, geom_name = self.graph[n]
            collision_managers.append(trimesh.collision.CollisionManager())
            collision_managers[-1].add_object('object', self.geometry[geom_name], transform=T)
        
        distance_matrix = np.zeros((len(nodes_geometry), len(nodes_geometry)))

        for i in range(len(nodes_geometry)):
            for j in range(len(nodes_geometry)):
                distance_matrix[i, j] = collision_managers[i].min_distance_other(collision_managers[j])

        if return_names:
            return distance_matrix, nodes_geometry

        return distance_matrix


    def connected_component_geometry(self, node_name, max_distance=1e-3, exclude_node_name=True):
        """Return the list of geometry nodes that are within a maximum distance of the geometry node node_name.
        Can be used to identify which geometries/objects are touching a specific one.

        Args:
            node_name (str, list[str]): A single geometry node identifier or a list of identifiers.
            max_distance (float, optional): Maximum distance to be considered part of the same connected component. Defaults to 1e-3.
            exclude_node_name (bool, optional): Whether to exclude node_name(s) itself in the returned set. Defaults to True.

        Raises:
            ValueError: Raised if node_name is not a geometry node in the scene.

        Returns:
            list[str]: The list of geometry nodes in the scene that are within max_distance of node_name.
        """
        distance_matrix, all_node_names = self.distance_matrix_geometry(nodes_geometry=None, return_names=True)

        if type(node_name) not in [list, tuple]:
            node_names = [node_name]
        else:
            node_names = node_name

        for n in node_names:
            if n not in all_node_names:
                raise ValueError(f"Node {node_name} is not a geometry node.")
        
        node_names_ids = [all_node_names.index(n) for n in node_names]
        connected_component = set(node_names + [n for i, n in enumerate(all_node_names) for node_name_id in node_names_ids if distance_matrix[node_name_id, i] <= max_distance])

        if exclude_node_name:
            connected_component -= set(node_names)

        return list(connected_component)

    def connected_components_geometry(self, nodes_geometry=None, max_distance=1e-3):
        """Get all connected components of the specified list of geometry nodes.
        Can be used to identify "islands" of geometry touching each other.

        Args:
            nodes_geometry (list[str]): A list of geometry nodes for which to compute the connected components.
            max_distance (float, optional): Maximum distance to be considered part of the same connected component. Defaults to 1e-3.

        Returns:
            list[list[str]]: A list of connected components.
        """
        if nodes_geometry is None:
            nodes_geometry = self.graph.nodes_geometry
        
        def get_all_connected_components(graph):
            already_seen = set()
            result = []
            for node in graph:
                if node not in already_seen:
                    connected_group, already_seen = get_connected_component(node, already_seen)
                    result.append(connected_group)
            return result

        def get_connected_component(node, already_seen):
            result = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                already_seen.add(node)
                nodes.update(n for n in graph[node] if n not in already_seen)
                result.append(node)
            return result, already_seen

        # create collision managers for distance checks
        collision_managers = {}
        for n in nodes_geometry:
            T, geom_name = self.graph[n]
            collision_managers[n] = trimesh.collision.CollisionManager()
            collision_managers[n].add_object('object', self.geometry[geom_name], transform=T)
        
        graph = {
            n: {m for m in nodes_geometry if m != n and collision_managers[m].min_distance_other(collision_managers[n]) <= max_distance}
            for n in nodes_geometry
        }

        components = get_all_connected_components(graph)
        
        return components
            

    def add_object(
        self,
        asset,
        obj_id=None,
        transform=None,
        translation=None,
        parent_id=None,
        connect_obj_id=None,
        connect_obj_anchor=None,
        connect_parent_id=None,
        connect_parent_anchor=None,
        joint_type="fixed",
        **kwargs,
    ):
        """Add a named object mesh to the scene.

        Args:
            asset (scene.Asset): Asset to be added.
            obj_id (str): Name of the object. If None, automatically generates a string.
            transform (np.ndarray): Homogenous 4x4 matrix describing the objects pose in scene coordinates. If None, is identity. Defaults to None.
            translation (list[float], tuple[float]): 3-vector describing the translation of the object. Cannot be set together with transform. Defaults to None.
            parent_id (str): Name of the parent object/frame in the scene graph. Defaults to base frame of scene.
            connect_obj_id (str): Name of a geometry in the asset to which to which the connect_obj_anchor refers. If this is None, the entire object is considered.
            connect_obj_anchor (tuple(str)): (["center", "com", "centroid", "bottom", "top"])*3 defining the coordinate origin of the object in all three dimensions (x, y, z).
            connect_parent_id (str): Name of an existing object in the scene next to which the new one will be added. If this is base_frame or None, all objects are considered.
            connect_parent_anchor (tuple(str)): (["center", "com", "centroid", "bottom", "top"])*3 defining the coordinate origin of the parent subscene/object in all three dimensions (x, y, z).
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "fixed".
            **use_collision_geometry (bool, optional): Whether to use collision or visual geometry, or both (if None). Defaults to default_use_collision_geometry.

        Returns:
            str: obj_id of added object.
        """
        use_collision_geometry = kwargs.get('use_collision_geometry', self._default_use_collision_geometry)

        if len(kwargs) > 1 or (len(kwargs) == 1 and 'use_collision_geometry' not in kwargs):
            raise ValueError("Unknown keyword arguments passed.")

        if obj_id is None:
            # find an object name that is not yet used
            cnt = 0
            obj_id = str(asset)
            while True:
                if obj_id not in self._scene.metadata["object_nodes"]:
                    break
                obj_id = f"{str(asset)}_{cnt}"
                cnt += 1
        elif obj_id in self._scene.metadata["object_nodes"]:
            raise ValueError(f"Obj_id '{obj_id}' already used in scene!")

        if joint_type not in ["floating", "fixed", None]:
            raise NotImplementedError(
                "Only joint_type 'floating', 'fixed', or None supported for adding objects. Type"
                f" '{joint_type}' not possible."
            )

        if transform is not None and translation is not None:
            raise ValueError(f"Only pass transform or translation, not both.")

        if transform is None:
            transform = np.eye(4)
        if translation is not None:
            transform[:3, 3] = translation

        if parent_id is None:
            parent_id = self._scene.graph.base_frame

        if connect_parent_id is None or len(connect_parent_id) == 0:
            # select entire scene
            connect_parent_id = list(self.metadata["object_nodes"].keys())

        if type(connect_parent_id) not in [list, tuple]:
            connect_parent_id = [connect_parent_id]

        # check that all entries of the connect_parent_id do exist
        for s in connect_parent_id:
            if s != self._scene.graph.base_frame and (s not in self._scene.graph.nodes):
                raise ValueError(f"{s} is not part of the scene. Cannot connect new object.")

        if self._scene.graph.base_frame in connect_parent_id:
            connect_parent_id = list(self.metadata["object_nodes"].keys())

        if connect_obj_anchor is None:
            T_obj = np.eye(4)
        else:
            T_obj = utils.homogeneous_inv(
                asset.get_reference_frame(
                    xyz=connect_obj_anchor,
                    query=connect_obj_id,
                    use_collision_geometry=use_collision_geometry,
                )
            )

        T_scene_frame = parent_id
        if connect_parent_anchor is None:
            if len(connect_parent_id) == 0 or connect_parent_id == list(
                self.metadata["object_nodes"].keys()
            ):
                T_scene = np.eye(4)
            else:
                # if len(connect_parent_id) != 1:
                #     raise ValueError(
                #         "If connect_parent_anchor is None, connect_parent_id must refer to a single"
                #         f" frame in the graph (or entire scene).  Currently: {connect_parent_id}"
                #     )

                T_scene = utils.get_transform(
                    self._scene,
                    frame_from=self._scene.graph.base_frame,
                    frame_to=connect_parent_id[0],
                )
        else:
            # This used to be:
            # frame=parent_id,
            # in case len(connect_parent_id) > 1
            # In the future, we might want to give the user a more explicit
            # choice on this frame since it affects the possible orientations.
            T_scene_frame = connect_parent_id[0]
            T_scene = self.get_reference_frame(
                xyz=connect_parent_anchor,
                query=connect_parent_id,
                frame=connect_parent_id[0],
            )

        if T_scene_frame == parent_id:
            T_parent = np.eye(4)
        else:
            T_parent = utils.homogeneous_inv(
                utils.get_transform(self._scene, frame_from=T_scene_frame, frame_to=parent_id)
            )

        transform = trimesh.util.multi_dot([T_parent, T_scene, transform, T_obj])

        scene_to_add = asset.as_trimesh_scene(
            namespace=obj_id, use_collision_geometry=use_collision_geometry
        )

        # add metadata about new added asset
        keep_metadata = self._scene.metadata.copy()
        keep_metadata["object_nodes"][obj_id] = list(scene_to_add.graph.nodes)
        nodes_geometry_to_add = scene_to_add.graph.nodes_geometry

        # Deal with trimesh issue
        # https://github.com/mikedh/trimesh/issues/2009
        node_data_scene = {
            node_id: self._scene.graph.transforms.node_data[node_id]
            for node_id in self._scene.graph.nodes
        }
        node_data_scene_to_add = {
            node_id: scene_to_add.graph.transforms.node_data[node_id]
            for node_id in scene_to_add.graph.nodes
        }
        for x in node_data_scene:
            node_data_scene[x].pop("geometry", None)
        for x in node_data_scene_to_add:
            node_data_scene_to_add[x].pop("geometry", None)

        self._scene.graph.update(obj_id, frame_from=parent_id, matrix=transform)

        keep_base_frame = self._scene.graph.base_frame
        self._scene = utils.append_scenes([self._scene, scene_to_add], common=[obj_id], share_geometry=self._share_geometry)

        # Check whether new geometry was added
        keep_metadata["object_geometry_nodes"][obj_id] = nodes_geometry_to_add

        self._scene.graph.base_frame = keep_base_frame
        self._scene.metadata = keep_metadata

        # Deal with trimesh issue
        for x in self._scene.graph.transforms.node_data:
            if x in node_data_scene:
                self._scene.graph.transforms.node_data[x].update(node_data_scene[x])
            if x in node_data_scene_to_add:
                self._scene.graph.transforms.node_data[x].update(node_data_scene_to_add[x])

        if joint_type is not None:
            self._scene.graph.transforms.edge_data[
                (self._scene.graph.transforms.parents[obj_id], obj_id)
            ].update(
                {
                    EDGE_KEY_METADATA: {
                        "joint": {
                            "name": f"{obj_id}/{parent_id.replace('/', '_')}_{joint_type}_joint",
                            "type": joint_type,
                        },
                    },
                }
            )

        self.synchronize_collision_manager()

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

        return obj_id

    def add_object_without_collision(
        self,
        obj_id,
        obj_asset,
        obj_pose_iterator,
        max_attempts=300,
        **kwargs,
    ):
        """Add object without colliding with the exisiting scene.

        This function is different from the place_object function since it is not restricted to place on a support surface.

        Args:
            obj_id (str): Name of the object to place.
            obj_asset (scene.Asset): The asset that represents the object.
            obj_pose_iterator: Iterator specifying the pose to place the object
            max_attempts (int, optional): Maximum number of attempts to find a placement pose. Defaults to 300.
            **use_collision_geometry (bool, optional): Whether to use collision or visual geometry or both. Defaults to default_use_collision_geometry.
            **kwargs: Keyword arguments that will be delegated to add_object.

        Returns:
            bool: Success.
        """
        obj_mesh = obj_asset.mesh(use_collision_geometry=kwargs.get('use_collision_geometry', self._default_use_collision_geometry))
        num_attempts = 0

        try:
            while True:
                placement_T = next(obj_pose_iterator)

                colliding = self.in_collision_single(
                    mesh=obj_mesh,
                    transform=placement_T,
                )

                num_attempts += 1

                if not colliding:
                    self.add_object(
                        obj_id=obj_id,
                        asset=obj_asset,
                        transform=placement_T,
                        **kwargs,
                    )
                    break

                if num_attempts > max_attempts:
                    log.warning(f"Couldn't add object {obj_id}!")
                    return False
        except StopIteration:
            log.warning(f"Couldn't add object {obj_id}! Ran out of object pose samples")
            return False

        return True

    def get_geometries(self, query=None):
        """Returns a list of trimesh geometries associated with the query.

        Args:
            query (str): A regular expression that gets matched against geometry names. None will include all geometries. Defaults to None.

        Returns:
            List[trimesh.Trimesh]: Geometries matching the query.
        """
        if query is None:
            geom_names = sorted(list(self.scene.geometry.keys()))
        else:
            geom_names = utils.select_sublist(
                query=query, all_items=sorted(list(self.scene.geometry.keys()))
            )
        
        return [self.geometry[n] for n in geom_names]
    
    def get_geometry_names(self, query=None, obj_id=None):
        """Return all geometry node names associated with that object or all geometry node names in the scene.

        Args:
            query (str or list[str]): A geometry name or list of names or regular expression. None will include all geometries if obj_id is also None. Defaults to None.
            obj_id (str, optional): Object identifier. If None, and query is None entire scene is considered. Defaults to None.

        Returns:
            list(str): List of geometry names.
        """
        if query is None and obj_id is None:
            node_names = set(self.scene.graph.nodes_geometry)
        elif obj_id is None:
            if type(query) is list or type(query) is tuple:
                node_names = []
                for k in query:
                    if k in self.metadata["object_nodes"]:
                        node_names.extend(self.metadata["object_nodes"][k])
                    else:
                        node_names.append(k)
                node_names = set(self.scene.graph.nodes_geometry).intersection(set(node_names))
            else:
                node_names = utils.select_sublist(
                    query=query, all_items=self.scene.graph.nodes_geometry
                )
        else:
            node_names = self._scene.metadata["object_geometry_nodes"][obj_id]

        return node_names
        # return [self.graph[n][1] for n in sorted(node_names)]

    def get_object_name(self, node_id):
        """Return object name given a scene graph node.

        Args:
            node_id (str): Name of the scene graph node.

        Returns:
            str: Name of the object or None if not found.
        """
        for object_name in self._scene.metadata["object_nodes"]:
            if node_id in self._scene.metadata["object_nodes"][object_name]:
                return object_name
        return None

    def get_object_names(self):
        """Return a list of all object names in the scene.

        Returns:
            list[str]: List of object names.
        """
        return list(self.metadata["object_nodes"].keys())

    def get_object_nodes(self, obj_id):
        """Return list of scene graph nodes pertaining to this object.

        Args:
            obj_id (str): Name of the object.

        Returns:
            list[str]: A list of graph nodes that this object consists of.
        """
        return self.metadata["object_nodes"][obj_id]

    def get_object_graph(self):
        """Get a subgraph that only contains nodes of objects and the scene's base frame.

        Returns:
            nx.networkx.Graph: The object graph of the scene.
        """
        object_nodes = list(self.metadata["object_nodes"].keys()) + [self.graph.base_frame]
        object_graph = self.graph.to_networkx()
        object_graph = object_graph.subgraph(object_nodes).copy()
        return object_graph

    def get_links(self, nodes=None):
        """Return a partioning of the scene graph into nodes pertaining to the same link, their root, and list of joints between the links.

        Args:
            nodes (list[str]): Nodes of scene graph to consider. If None, considers entire scene graph. Defaults to None.

        Returns:
            list[set[str]]: List of sets of nodes, each set representing a link.
            list[str]: List of root nodes, one per link.
            list[(str, str, dict]: List of joints.
        """
        # Third Party
        import networkx as nx

        # create scene graph without joint edges
        edges = []
        tmp_joints = []
        for e in self._scene.graph.to_edgelist():
            if nodes is not None and (e[0] not in nodes or e[1] not in nodes):
                # Skip edges that connect a node that is not of interest
                continue

            if EDGE_KEY_METADATA not in e[2] or e[2][EDGE_KEY_METADATA] is None or "joint" not in e[2][EDGE_KEY_METADATA]:
                edges.append(e)
            else:
                tmp_joints.append(e)

        link_graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
        link_graph.add_nodes_from(self.graph.nodes if nodes is None else nodes)

        links = [sorted(list(c)) for c in nx.weakly_connected_components(link_graph)]

        link_graphs = [link_graph.subgraph(l) for l in links]
        link_roots = [[x for x, num in l.in_degree() if num == 0][0] for l in link_graphs]

        root_map = {n: root for l, root in zip(links, link_roots) for n in l}

        joints = {}
        for j in tmp_joints:
            joints[(root_map[j[0]], root_map[j[1]])] = j

        return links, link_roots, joints

    def get_object_transforms(self, frame_from=None):
        """Return a dictionary of all object's transformations.

        Args:
            frame_from (str, optional): The frame in which the transforms are returned. Defaults to None which is the graph's base_frame.

        Returns:
            dict[str, np.ndarray]: A dictionary with obj_id's as keys and homogeneous matrices as values.
        """
        result = {}
        for obj_id in self.metadata["object_nodes"]:
            result[obj_id] = self.graph.get(obj_id, frame_from=frame_from)[0]
        return result

    def get_transform(self, node, frame_from=None):
        """Returns the transform of node node.

        Args:
            node (str): Node identifier.
            frame_from (str): Node identifier.

        Returns:
            np.ndarray: 4x4 homogeneous matrix transformation
        """
        return self.graph.get(node, frame_from=frame_from)[0]

    def set_transform(self, node, frame_from, transform):
        """Set the transform between nodes node and frame_from.
        Only allows setting transforms for existing edges in the scene graph.

        Args:
            node (str): Node identifier.
            frame_from (str): Node identifier.
            transform (np.ndarray): 4x4 homogeneous matrix transformation.

        Raises:
            ValueError: Raise error if there is no edge in the scene graph between the nodes.
        """
        if (frame_from, node) not in self.graph.transforms.edge_data:
            raise ValueError(f"Transform frame_from={frame_from} to node={node} doesn't exist.")
        
        self._scene.graph.update(
            frame_to=node,
            frame_from=frame_from,
            matrix=transform,
        )

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)
    
    def get_transforms(self, nodes=None, frames_from=None):
        """Return a list of all or a subset of all transformations in the scene graph.

        Args:
            nodes (list[str]): List of node identifiers. None means all. Defaults to None.
            frames_from (list[str]): List of node identifiers. None means all. Defaults to None.
        
        Returns:
            list: A list of transformations with tuples of the form (parent [str], child [str], attr [dict]).
        """
        transform_list = []
        for edge, attr in self._scene.graph.transforms.edge_data.items():
            if "matrix" in attr:
                a, b = edge
                if frames_from is not None and a not in frames_from:
                    continue
                if nodes is not None and b not in nodes:
                    continue

                transform_list.append((a, b, {"matrix": attr["matrix"].tolist()}))
        return transform_list

    def set_transforms(self, transforms, only_update_existing=False):
        """Sets the transforms in the scene. Inverse of get_transforms().

        Args:
            only_update_existing (bool): Only update transforms that already exist in the scene. Defaults to False.
            transforms (list): A list of transformations with tuples of the form (parent [str], child [str], attr [dict]).
        """
        for a, b, attr in transforms:
            if only_update_existing:
                if (a, b) in self.graph.transforms.edge_data:
                    new_attr = self.graph.transforms.edge_data[(a, b)]
                    new_attr.update(attr)
                    self._scene.graph.update(
                        frame_from=a,
                        frame_to=b,
                        **new_attr,
                    )
            else:
                new_attr = (
                    self.graph.transforms.edge_data[(a, b)]
                    if (a, b) in self.graph.transforms.edge_data
                    else {}
                )
                new_attr.update(attr)
                self._scene.graph.update(
                    frame_from=a,
                    frame_to=b,
                    **new_attr,
                )

        # invalidate cache of SceneGraph
        utils.invalidate_scenegraph_cache(self._scene)

    def set_joint_configuration(self, configuration, obj_id=None, joint_ids=None):
        """Set configuration of articulated objects, indiviual joints, or for the entire scene at once.
        If obj_id and joint_ids are specified the joint names will be a concatenation of obj_id and joint_ids.

        Note: This is the same as `update_configuration`

        Args:
            configuration (list[float]): New configuration value(s).
            obj_id (str, optional): Object identifier to configure. If None and joint_ids=None, all joints in the scene are expected to be updated. Defaults to None.
            joint_ids (list[str], optional): List of joint names to update. If None, all joints of the object are expected to be updated. Defaults to None.

        Raises:
            ValueError: The obj_id is not part of the scene.
            ValueError: The joint_ids doe not exist.
            ValueError: The len(configuration) does not match the object's number of joints.
            ValueError: The len(configuration) does not match len(joint_ids) if joint_ids is specified.
            ValueError: The len(configuration) does not match the number of joints in the scene if neither obj_id nor joint_ids are specified.

        Examples
        --------
        >>> s = synth.Scene()
        >>> s.add_object(pa.RefrigeratorAsset.random(), 'fridge')
        >>> s.get_joint_names()
        >>> s.set_joint_configuration([1, 0.5])
        >>> s.set_joint_configuration([1, 0.5], obj_id='fridge')
        >>> s.set_joint_configuration([1], joint_ids=['fridge/door_joint'])
        >>> s.set_joint_configuration([1], obj_id='fridge', joint_ids=['door_joint'])
        """
        self.update_configuration(configuration=configuration, obj_id=obj_id, joint_ids=joint_ids)
    
    def update_configuration(self, configuration, obj_id=None, joint_ids=None):
        """Set configuration of articulated objects, indiviual joints, or for the entire scene at once.
        If obj_id and joint_ids are specified the joint names will be a concatenation of obj_id and joint_ids.

        Args:
            configuration (list[float]): New configuration value(s).
            obj_id (str, optional): Object identifier to configure. If None and joint_ids=None, all joints in the scene are expected to be updated. Defaults to None.
            joint_ids (list[str], optional): List of joint names to update. If None, all joints of the object are expected to be updated. Defaults to None.

        Raises:
            ValueError: The obj_id is not part of the scene.
            ValueError: The joint_ids doe not exist.
            ValueError: The len(configuration) does not match the object's number of joints.
            ValueError: The len(configuration) does not match len(joint_ids) if joint_ids is specified.
            ValueError: The len(configuration) does not match the number of joints in the scene if neither obj_id nor joint_ids are specified.

        Examples
        --------
        >>> s = synth.Scene()
        >>> s.add_object(pa.RefrigeratorAsset.random(), 'fridge')
        >>> s.get_joint_names()
        >>> s.update_configuration([1, 0.5])
        >>> s.update_configuration([1, 0.5], obj_id='fridge')
        >>> s.update_configuration([1], joint_ids=['fridge/door_joint'])
        >>> s.update_configuration([1], obj_id='fridge', joint_ids=['door_joint'])
        """
        joint_names = []
        scene_joint_names = self.get_joint_names()
        if obj_id is None and joint_ids is None:
            # set configuration for entire scene
            joint_names = scene_joint_names
        elif joint_ids is None:
            # set configuration for entire object
            if obj_id not in self.metadata["object_nodes"].keys():
                raise ValueError(f"Unknown object_id: {obj_id}")

            joint_names = self.get_joint_names(obj_id=obj_id)
        elif obj_id is None:
            # set configuration for single joint(s)
            for joint_id in joint_ids:
                if joint_id not in scene_joint_names:
                    raise ValueError(f"Unknown joint_id: {joint_id}")
            joint_names = joint_ids
        else:
            # set configuration for single joint(s) but name is split into (obj_id, joint_id)
            if obj_id not in self.metadata["object_nodes"].keys():
                raise ValueError(f"Unknown object_id: {obj_id}")

            joint_names = [obj_id + "/" + joint_id for joint_id in joint_ids]

        # check configuration vector length
        assert len(configuration) == len(joint_names), f"Length of {configuration} != {joint_names}"

        # update scene graph
        utils.forward_kinematics(self._scene, joint_names, configuration)

        self.synchronize_collision_manager()

    @functools.lru_cache(maxsize=None)
    def get_joint_names(
        self, obj_id=None, include_fixed_floating_joints=False, joint_type_query=None
    ):
        """Return list of joint names for an object or the entire scene.
        The order of this list is the same as the values of get_configuration.

        Args:
            obj_id (str, optional): Name of the object. If None will return all joint names in the scene. Defaults to None.
            include_fixed_floating_joints (bool, optional): Whether to include fixed and floating joints. Defaults to False.
            joint_type_query (str/regex or tuple, optional): Filter for specific joint types. Will override include_fixed_floating_joints if not None. Defaults to None.

        Returns:
            list[str]: List of joint names that belong to this articulated object, or of entire scene. Empty if no such object exists or object/scene is not articulated.
        """
        all_types = ["revolute", "prismatic", "continuous", "fixed", "floating"]
        if joint_type_query is not None:
            joint_types = utils.select_sublist(joint_type_query, all_types)
        elif include_fixed_floating_joints:
            joint_types = all_types
        else:
            joint_types = ["revolute", "prismatic", "continuous"]

        scene_edge_data = self._scene.graph.transforms.edge_data
        joint_names = []
        for k in self._scene.graph.transforms.edge_data:
            edge_data = scene_edge_data[k]
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
                and (obj_id is None or edge_data[EDGE_KEY_METADATA]["joint"]["name"].startswith(obj_id + '/'))
            ):
                if edge_data[EDGE_KEY_METADATA]["joint"]["type"] in joint_types:
                    joint_names.append(edge_data[EDGE_KEY_METADATA]["joint"]["name"])

        joint_names = sorted(joint_names)

        return joint_names

    def is_articulated(self, obj_id=None):
        """Check if object or scene is articulated, i.e., has joints.

        Args:
            obj_id (str, optional): Name of the object. If None, will check entire scene. Defaults to None.

        Returns:
            bool: True if object/scene is articulated.
        """
        if obj_id is None:
            return len(self.get_joint_names()) > 0
        return len(self.get_joint_names(obj_id=obj_id)) > 0 or len(self.get_joint_names(obj_id=obj_id, joint_type_query='fixed')) > 1

    @functools.lru_cache(maxsize=None)
    def get_joint_properties(self, obj_id=None, include_fixed_floating_joints=False):
        """Return joint properties of scene or specific object in scene.

        Args:
            obj_id (str, optional): Object identifier. If none, entire scene is considered. Defaults to None.
            include_fixed_floating_joints (bool, optional): Whether to include fixed and floating joints. Defaults to False.

        Returns:
            dict: A dictionary with properties of each joint.
        """
        scene_edge_data = self._scene.graph.transforms.edge_data
        joint_props = {}
        for k in self._scene.graph.transforms.edge_data:
            edge_data = scene_edge_data[k]
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
                and (obj_id is None or edge_data[EDGE_KEY_METADATA]["joint"]["name"].startswith(obj_id))
            ):
                joint_data = edge_data[EDGE_KEY_METADATA]["joint"]
                if include_fixed_floating_joints or joint_data["type"] not in ["fixed", "floating"]:
                    joint_props[joint_data["name"]] = joint_data

        return joint_props

    def update_joint_properties(self, joint_id, **kwargs):
        """Update the properties of a joint.

        Args:
            joint_id (str): Name of the joint to update.
            **q (float, optional): Configuration of the joint.
            **type (str, optional): Type of the joint.
            **origin (np.ndarray, optional): Homogenous matrix representing origin of joint.
            **axis (list[float], optional): Axis of the joint.
            **limit_velocity (float, optional): Joint velocity limit.
            **limit_effort (float, optional): Joint effort limit.
            **limit_lower (float, optional): Lower joint limit.
            **limit_upper (float, optional): Upper joint limit.
            **stiffness (float, optional): Joint stiffness. Will add a drive to the joint during USD export.
            **damping (float, optional): Joint damping. Will add a drive to the joint during USD export.

        Raises:
            ValueError: Error raised if property is unknown.
        """
        for k in kwargs:
            if k not in ('name', 'q', 'origin', 'axis', 'type', 'limit_velocity', 'limit_effort', 'limit_lower', 'limit_upper', 'stiffness', 'damping'):
                raise ValueError(f"Unknown joint property {k}. Can't update.")
        
        parent_node, child_node = self.get_joint_parent_child_node(joint_id)
        
        scene_edge_data = self._scene.graph.transforms.edge_data

        if (parent_node, child_node) not in scene_edge_data:
            raise ValueError(f"Joint {joint_id} not in scene. Can't update its properties.")

        scene_edge_data[(parent_node, child_node)][EDGE_KEY_METADATA]["joint"].update({**kwargs})

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

        # do forward kinematics
        self.invalidate_scenegraph_cache()

    def get_joint_limits(self, obj_id=None, joint_ids=None):
        """Return upper and lower joint limits.

        Args:
            obj_id (str, optional): Only consider a specific object in the scene. Defaults to None.
            joint_ids (list[str], optional): Only consider certain joints. Defaults to None.

        Returns:
            np.ndarray: Nx2 array of lower and upper limits for N joints.
        """
        if joint_ids is not None:
            joint_names = (
                joint_ids if obj_id is None else [obj_id + "/" + joint_id for joint_id in joint_ids]
            )
        else:
            joint_names = self.get_joint_names(obj_id=obj_id, include_fixed_floating_joints=False)

        joint_props = self.get_joint_properties()
        limits = [
            [joint_props[joint_name]["limit_lower"], joint_props[joint_name]["limit_upper"]]
            for joint_name in joint_names
        ]

        return np.array(limits)
    
    def set_joint_zero_configuration(self, obj_id=None, joint_ids=None):
        """This sets the zero configuration of a joint by using its current configuration.
        I.e., the current configuration will be the new zero configuration and joint limits are adapted accordingly.

        Args:
            obj_id (str, optional): Only consider a specific object in the scene. Defaults to None.
            joint_ids (list[str], optional): Only consider certain joints. Defaults to None.
        """
        if joint_ids is not None:
            joint_names = (
                joint_ids if obj_id is None else [obj_id + "/" + joint_id for joint_id in joint_ids]
            )
        else:
            joint_names = self.get_joint_names(obj_id=obj_id, include_fixed_floating_joints=False)
        
        if len(joint_names) == 0:
            return

        joint_props = self.get_joint_properties(include_fixed_floating_joints=True)

        for j in joint_names:
            updated_joint_props = joint_props[j].copy()

            current_configuration = updated_joint_props['q']
            updated_joint_props['limit_lower'] -= current_configuration
            updated_joint_props['limit_upper'] -= current_configuration
            updated_joint_props['q'] = 0.0

            parent, child = self.get_joint_parent_child_node(j)
            T = self.get_transform(child, parent)

            updated_joint_props['origin'] = np.copy(T)

            self.update_joint_properties(joint_id=j, **updated_joint_props)

    def get_joint_types(self, obj_id=None, joint_ids=None, include_fixed_floating_joints=False):
        """Return joint types.

        Args:
            obj_id (str, optional): Only consider a specific object in the scene. Defaults to None.
            joint_ids (list[str], optional): Only consider certain joints. Defaults to None.
            include_fixed_floating_joints (bool, optional): Whether to include fixed and floating joints. Defaults to False.

        Returns:
            list[str]: A list of joint types. In the same order as get_joint_names.
        """
        if joint_ids is not None:
            joint_names = (
                joint_ids if obj_id is None else [obj_id + "/" + joint_id for joint_id in joint_ids]
            )
            include_fixed_floating_joints = True
        else:
            joint_names = self.get_joint_names(
                obj_id=obj_id, include_fixed_floating_joints=include_fixed_floating_joints
            )

        joint_props = self.get_joint_properties(
            include_fixed_floating_joints=include_fixed_floating_joints
        )
        types = [joint_props[joint_name]["type"] for joint_name in joint_names]

        return types

    def set_joint_types(
        self, joint_types, obj_id=None, joint_ids=None, include_fixed_floating_joints=False
    ):
        """Change the type of a joint. Note, this is mostly helpful for changing between `floating`, `fixed`, or no (`None`) joints.
        Since other types of joints (e.g. `prismatic`, `revolute`) also require additional properties (e.g. `axis`, `origin`).
        If a joint type is set to None this will internally call the remove_joint function.

        Args:
            joint_types (list[str]): List of desired joint types.
            obj_id (str, optional): Only consider a specific object in the scene. Defaults to None.
            joint_ids (list[str], optional): Only consider certain joints. Defaults to None.
            include_fixed_floating_joints (bool, optional): Whether to include fixed and floating joints. Defaults to False.

        Raises:
            ValueError: The length of joint_types does not match the number of joints queried through obj_id, joint_ids, and include_fixed_floating_joints.
            ValueError: The joint_types contain an unknown type of joint.
        """
        if joint_ids is not None:
            joint_names = (
                joint_ids if obj_id is None else [obj_id + "/" + joint_id for joint_id in joint_ids]
            )
        else:
            joint_names = self.get_joint_names(
                obj_id=obj_id, include_fixed_floating_joints=include_fixed_floating_joints
            )

        if len(joint_types) != len(joint_names):
            raise ValueError(
                f"List of joint_types should have {len(joint_names)} items but has"
                f" {len(joint_types)}."
            )

        props = self.get_joint_properties(include_fixed_floating_joints=True)

        joints_to_be_removed = []
        for j, joint_type in zip(joint_names, joint_types):
            if joint_type is None:
                joints_to_be_removed.append(j)
            elif joint_type in ("revolute", "prismatic", "floating", "fixed"):
                props[j]["type"] = joint_type
            else:
                raise ValueError(f"Joint type {joint_type} is unknown.")

        self.get_joint_properties.cache_clear()

        if len(joints_to_be_removed) > 0:
            self.remove_joints(joints_to_be_removed)

    def zero_configurations(self):
        """Set all articulations to the zero configuration."""
        self.update_configuration(np.zeros(len(self.get_joint_names())))

    def lower_configurations(self):
        """Set all articulations to their lower limit configuration."""
        limits = self.get_joint_limits()
        if len(limits) > 0:
            self.update_configuration(limits[:, 0])

    def upper_configurations(self):
        """Set all articulations to their upper limit configuration."""
        limits = self.get_joint_limits()
        if len(limits) > 0:
            self.update_configuration(limits[:, 1])

    def random_configurations(self):
        """Randomize all joint configurations, within their limits."""
        limits = self.get_joint_limits()
        if len(limits) > 0:
            cfg = self._rng.uniform(low=limits[:, 0], high=limits[:, 1])
            self.update_configuration(cfg)
        return self

    def get_configuration(self, obj_id=None, joint_ids=None):
        """Return value(s) of joint configuration(s) of object obj_id and joint name joint_ids, or for the entire scene if both are None.

        Args:
            obj_id (str, optional): Object identifier. If None will use full joint_name qualifier. Defaults to None.
            joint_ids (str, List[str], optional): Joint identifier(s). If None will return list of values belonging to this object. Defaults to None.

        Returns:
            np.ndarray: Joint configuration.
        """
        if joint_ids is not None:
            if obj_id is None:
                joint_names = joint_ids
            else:
                joint_names = [obj_id + "/" + joint_id for joint_id in joint_ids]
        else:
            joint_names = self.get_joint_names(obj_id=obj_id)

        joint_props = self.get_joint_properties()
        configuration = [joint_props[joint_name]["q"] for joint_name in joint_names]

        return np.array(configuration, dtype=float)

    def add_joint(self, parent_node, child_node, name, type, **kwargs):
        """Add a new joint to the scene.

        Args:
            parent_node (str): Parent node in the scene graph.
            child_node (str): Child node in the scene graph.
            name (str): Identifier of the joint. Needs to be of the form <obj_id>/<joint_id>.
            type (str): 'revolute', 'prismatic', 'floating', or 'fixed'.
            **q (float, optional): Configuration of the joint. Defaults to 0.0.
            **origin (np.ndarray, optional): Homogenous matrix representing origin of joint. Defaults to self.get_transform(frame_to=child_node, frame_from=parent_node).
            **axis (list[float], optional): Axis of the joint. Defaults to [1, 0, 0].
            **limit_lower (float, optional): Lower joint limit. Defaults to constanst.DEFAULT_LIMIT_LOWER for revolute and prismatic joints.
            **limit_upper (float, optional): Upper joint limit. Defaults to constanst.DEFAULT_LIMIT_UPPER for revolute and prismatic joints.
            **limit_velocity (float, optional): Joint velocity limit. Defaults to None.
            **limit_effort (float, optional): Joint effort limit. Defaults to None.
            **stiffness (float, optional): Joint stiffness. Defaults to None.
            **damping (float, optional): Joint damping. Defaults to None.

        Raises:
            ValueError: If name already exists.
            ValueError: If name is not of the form <obj_id>/<joint_id>.
            ValueError: If type is not one of the predefined ones.
            ValueError: If there's no edge in the scene graph between parent_node and child_node.
            ValueError: If there's already a joint in the scene graph between parent_node and child_node.
        """
        joint_names = self.get_joint_names()
        if name in joint_names:
            raise ValueError(f"Can't add joint with name {name}. It already exists.")

        if type not in ["revolute", "prismatic", "fixed", "floating"]:
            raise ValueError(f"Joint type {type} needs to be one of: revolute, prismatic, floating, fixed.")

        if type != "fixed" and type != "floating":
            if "q" not in kwargs:
                kwargs["q"] = 0.0
            if "origin" not in kwargs:
                kwargs["origin"] = self.get_transform(node=child_node, frame_from=parent_node)
            if "axis" not in kwargs:
                kwargs["axis"] = np.array([1.0, 0, 0])
            if "limit_lower" not in kwargs:
                kwargs["limit_lower"] = DEFAULT_JOINT_LIMIT_LOWER
            if "limit_lower" not in kwargs:
                kwargs["limit_upper"] = DEFAULT_JOINT_LIMIT_UPPER

        scene_edge_data = self._scene.graph.transforms.edge_data
        if (parent_node, child_node) not in scene_edge_data:
            raise ValueError(f"No edge between {parent_node} and {child_node} in the scene graph.")

        parent_child_node_obj_ids = [parent_node.split('/')[0], child_node.split('/')[0]]
        if not '/' in name or not name.split('/')[0] in parent_child_node_obj_ids:
            raise ValueError(f"Joint name {name} must be of the form <obj_id>/<joint_id>. Given parent and child node, <obj_id> can be {parent_child_node_obj_ids}.")

        if (
            EDGE_KEY_METADATA in scene_edge_data[(parent_node, child_node)]
            and scene_edge_data[(parent_node, child_node)][EDGE_KEY_METADATA] is not None
            and "joint" in scene_edge_data[(parent_node, child_node)][EDGE_KEY_METADATA]
        ):
            raise ValueError(
                f"Can't add joint between {parent_node} and {child_node} in the scene graph."
                " There's already one."
            )

        scene_edge_data[(parent_node, child_node)].update(
            {EDGE_KEY_METADATA: {"joint": {"name": name, "type": type, **kwargs}}}
        )

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

    def remove_joints(self, joint_query):
        """Remove one or more joint from the scene.

        Args:
            joint_query (str or list): Identifier of joint(s). Can be a regex.

        Raises:
            ValueError: Joint doesn't exist.
        """
        all_joint_names = self.get_joint_names(include_fixed_floating_joints=True)
        joint_ids = utils.select_sublist(joint_query, all_joint_names)

        scene_edge_data = self._scene.graph.transforms.edge_data

        for k in self._scene.graph.transforms.edge_data:
            edge_data = scene_edge_data[k]
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
                and edge_data[EDGE_KEY_METADATA]["joint"]["name"] in joint_ids
            ):
                del self._scene.graph.transforms.edge_data[k][EDGE_KEY_METADATA]["joint"]

        # clear cache
        self.get_joint_names.cache_clear()
        self.get_joint_properties.cache_clear()

    def find_joint(self, node_name, include_fixed_floating_joints=False):
        """Find first joint in kinematic tree, going from node_name to the root of the tree.

        Args:
            node_name (str): Name of the node in the scene graph / kinematic tree.
            include_fixed_floating_joints (bool, optional): Whether to consider joints of type 'fixed' or 'floating'. Defaults to False.

        Returns:
            str: Name of joint found. None if no joint was found.
        """
        scene_edge_data = self._scene.graph.transforms.edge_data

        n = node_name
        while n != self.graph.base_frame:
            parent = self.graph.transforms.parents[n]

            edge_data = scene_edge_data[(parent, n)]
            # check if articulation between n and parent
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
            ):
                if include_fixed_floating_joints or edge_data[EDGE_KEY_METADATA]["joint"]["type"] not in [
                    "fixed",
                    "floating",
                ]:
                    return edge_data[EDGE_KEY_METADATA]["joint"]["name"]

            n = parent

        return None

    def get_joint_parent_child_node(self, joint_id):
        """Return parent and child node of scene graph with joint attribute.

        Args:
            joint_id (str): Name of the joint.

        Raises:
            ValueError: Joint with this name doesn't exist.

        Returns:
            (str, str): Tuple of parent and child node.
        """
        scene_edge_data = self.graph.transforms.edge_data
        for edge in scene_edge_data:
            edge_data = scene_edge_data[edge]
            if (
                EDGE_KEY_METADATA in edge_data
                and edge_data[EDGE_KEY_METADATA] is not None
                and "joint" in edge_data[EDGE_KEY_METADATA]
                and edge_data[EDGE_KEY_METADATA]["joint"]["name"] == joint_id
            ):
                return edge

        raise ValueError(f"No joint named {joint_id}.")

    def stack_box(
        self,
        obj_id,
        thickness,
        stack_parent_obj_ids=None,
        offset=0.0,
        direction="z",
        joint_type="fixed",
    ):
        """Add a named box mesh to the scene by stacking it next to an existing object.

        Args:
            obj_id (str): Name of the box.
            thickness (float): Box extent along the stacking direction.
            stack_parent_obj_ids (list[str], optional): List of names of existing objects in the scene next to which the box will be placed. Also defines the dimensions of the box. If None all existing objects in the scene will be considered. Defaults to None.
            offset (float, optional): Distance between stacked objects. Defaults to 0.0.
            direction (str, optional): Either 'x', '-x', 'y', '-y', 'z', or '-z'. Defines along which direction to stack the box. Defaults to 'z'.
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "fixed".

        Returns:
            str: obj_id of added box.
        """
        dir_to_ind = {"x": 0, "y": 1, "z": 2, "-x": 0, "-y": 1, "-z": 2}
        if direction not in dir_to_ind:
            raise ValueError(
                "Direction '{direction}' unknown (needs to be x, y, z, -x, -y, or -z). Cannot stack"
                " new object."
            )

        if stack_parent_obj_ids is None:
            stack_parent_obj_ids = list(self.metadata["object_nodes"].keys())
        elif not isinstance(stack_parent_obj_ids, list):
            raise TypeError("stack_parent_obj_ids needs to be None or list.")

        box_extents = self.subscene(obj_ids=stack_parent_obj_ids)._scene.extents
        box_extents[dir_to_ind[direction]] = thickness

        new_box = BoxAsset(extents=box_extents)

        parent_anchor = ["center", "center", "center"]
        obj_anchor = ["center", "center", "center"]
        parent_anchor[dir_to_ind[direction]] = "top"
        obj_anchor[dir_to_ind[direction]] = "bottom"

        if direction.startswith("-"):
            parent_anchor, obj_anchor = obj_anchor, parent_anchor

        offset_transform = tra.identity_matrix()
        offset_transform[dir_to_ind[direction], 3] = (
            -offset if direction.startswith("-") else offset
        )

        return self.add_object(
            obj_id=obj_id,
            asset=new_box,
            connect_parent_id=stack_parent_obj_ids,
            connect_parent_anchor=parent_anchor,
            connect_obj_anchor=obj_anchor,
            transform=offset_transform,
            joint_type=joint_type,
        )

    def _raycasts(self, origins, directions, mesh):
        """Helper function, should be moved to utils?

        Args:
            origins (np.ndarray): Origins of rays.
            directions (np.ndarray): Directions of rays.
            mesh (trimesh.Trimesh): Mesh to use for ray casts.

        Returns:
            np.ndarray: Locations of intersections, where rays hit the mesh.
            np.ndarray: Ray indices, mapping each returned location to a ray.
            np.ndarray: Array of triangle (face) indexes.
        """
        assert len(origins) == len(directions)
        return mesh.ray.intersects_location(origins, directions, multiple_hits=False)

    def _get_support_polyhedra(
        self,
        support_surfaces=None,
        min_volume=0.000001,
        distance_above_support=0.001,
        min_area=0.01,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        ray_cast_count=10,
        max_height=10.0,
        layer="collision",
        **kwargs,
    ):
        """Creates support polyhedra which are volumes created by extruding support polygons until collision.

        Args:
            min_volume (float, optional): Only return polyhedra with volume greater than this minimum. Defaults to 0.000001.
            distance_above_support (float, optional): Support polyhedra are above the support polygon by this amount. Defaults to 0.001.
            min_area (float, optional): See _get_support_polyhedra. Defaults to 0.01.
            gravity (np.ndarray, optional): See _get_support_polyhedra. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): See _get_support_polyhedra. Defaults to 0.02.
            ray_cast_count (int, optional): For testing collisions to extrude support polygons. Defaults to 10.
            max_height (float, optional): Maximum height for container volume (in extrusion direction). Defaults to 10.0.
            layer (str, optional): Name of the layer of the support geometry. Defaults to 'collision'.
            **obj_ids (str): Regular expression of object ids to consider.
            **geom_ids (str): Regular expression of geometry identifiers to use for finding supports.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            list[trimesh.Trimesh]: Support polyhedra in the scene that satisfy the filter criteria.
        """
        if support_surfaces is None:
            support_surfaces = self._get_support_polygons(
                min_area=min_area,
                gravity=gravity,
                gravity_tolerance=gravity_tolerance,
                erosion_distance=erosion_distance,
                layer=layer,
                **kwargs,
            )

        if len(support_surfaces) == 0:
            log.warning("Warning! No support polygons selected.")

        support_polyhedra = []
        support_polyhedra_mask = []

        scene_mesh = self._scene.dump(concatenate=True)

        for support_surface in support_surfaces:
            (is_support_polyhedra, inscribing_polyhedra,) = self._compute_support_polyhedra(
                support_surface=support_surface,
                mesh=scene_mesh,
                gravity=gravity,
                ray_cast_count=ray_cast_count,
                min_volume=min_volume,
                distance_above_support=distance_above_support,
                max_height=max_height,
                erosion_distance=erosion_distance,
            )
            if is_support_polyhedra:
                support_polyhedra.append(
                    Container(
                        geometry=inscribing_polyhedra,
                        node_name=support_surface.node_name,
                        transform=support_surface.transform,
                        support_surface=support_surface,
                    )
                )

            support_polyhedra_mask.append(is_support_polyhedra)

        return (
            support_polyhedra_mask,
            support_polyhedra,
        )

    def _raycast_surface(
        self,
        support_surface,
        ray_cast_count,
        mesh=None,
        gravity=None,
        distance_above_support=1e-3,
        debug=False,
    ):
        """
        Extrudes a support polygon until collision.

        Args:
            support_surface (SupportSurface): The support surface.
            ray_cast_count (int, optional): For testing collisions to extrude support polygons.
            mesh (trimesh.Trimesh, optional): Defaults to the scene's mesh.
            gravity (np.ndarray, optional): Defaults to np.array([0, 0, -1]) in the surface's coordinate frame.
            distance_above_support (float, optional): Support polyhedra are above the support polygon by this amount.
            debug (bool, optional): Whether to visualize the raycasting results.

        Returns:
            list[np.ndarray]: List of ray origins on the surface.
            list[np.ndarray]: List of ray intersections on the mesh.
        """
        # for each support polygon, sample raycasts to determine maximum height of extrusion in direction of gravity
        pts = utils.sample_polygon(support_surface.polygon, count=ray_cast_count, seed=self._rng)
        # pts = np.array(support_surface.polygon.exterior.coords)

        if len(pts) == 0:
            return [], []

        pts3d_local = np.column_stack([pts, distance_above_support * np.ones(len(pts))])
        T = self._scene.graph.get(support_surface.node_name)[0] @ support_surface.transform
        pts3d = trimesh.transform_points(points=pts3d_local, matrix=T)

        if mesh is None:
            mesh = self._scene.dump(concatenate=True)
        if gravity is None:
            gravity = T[:3, :3] @ np.array([0, 0, -1])

        intersections, ray_ids, _ = self._raycasts(
            origins=pts3d,
            directions=np.array(len(pts) * [list(-tra.unit_vector(gravity))]),
            mesh=mesh,
        )
        if len(intersections) == 0:
            return [], []

        origins = pts3d[ray_ids]
        if debug:
            surface_color = utils.random_color(seed=self._rng)
            surface_path = trimesh.load_path(support_surface.polygon).to_3D().apply_transform(T)
            surface_path.colors = len(surface_path.entities) * [surface_color]
            ray_path = trimesh.load_path(
                np.swapaxes(np.stack([origins, intersections], axis=2), 1, 2)
            )
            trimesh.Scene([mesh, surface_path, ray_path]).show()

        return origins, intersections

    def _estimate_surface_coverage(
        self,
        support_surface,
        rays_per_area=1 / 1e-1**2,
        covered_distance=0.25,
        **kwargs,
    ):
        """
        Estimates the fraction of a surface that is covered above.

        Args:
            support_surface (SupportSurface): The support surface.
            rays_per_area (float, optional): The number of rays per square meter.
            covered_distance (float, optional): The maximum ray distance that is considered covered.
            **kwargs: Keyword arguments that will be delegated to _raycast_surface.

        Returns:
            float: An estimate of the fraction of the surface that is covered.
        """

        area = support_surface.polygon.area
        num_rays = int(rays_per_area * area) + 1
        origins, intersections = self._raycast_surface(support_surface, num_rays, **kwargs)
        if len(intersections) == 0:
            return 0.0
        distances = np.linalg.norm(intersections - origins, axis=1)
        num_hits = np.sum(distances <= covered_distance)
        coverage = num_hits / num_rays
        return coverage

    def create_surface_coverage_test(self, max_surface_coverage=0.9, **kwargs):
        """Creates a test for whether a surface is covered.

        Args:
            max_surface_coverage (float, optional): The maximum fraction of the surface that can be covered.
            **kwargs: Keyword arguments that will be delegated to _estimate_surface_coverage.

        Returns:
            function: Boolean function that tests whether a surface is covered.
        """
        scene_mesh = self._scene.dump(concatenate=True)

        def _test(surface):
            if max_surface_coverage is None:
                return True
            surface_coverage = self._estimate_surface_coverage(surface, mesh=scene_mesh, **kwargs)
            return surface_coverage <= max_surface_coverage

        return _test

    def _compute_support_polyhedra(
        self,
        support_surface,
        mesh,
        gravity,
        ray_cast_count,
        min_volume,
        distance_above_support,
        max_height,
        erosion_distance,
        **kwargs,
    ):
        """
        See documentation for _get_support_polyhedra. Computes support polyhedra for a single polygon

        Returns:
            bool: If support_polygon is a support polyhedra
            trimesh.Trimesh: Corresponding support polyhedra
        """
        origins, intersections = self._raycast_surface(
            support_surface=support_surface,
            ray_cast_count=ray_cast_count,
            mesh=mesh,
            gravity=gravity,
            distance_above_support=distance_above_support,
            **kwargs,
        )

        # if no intersection occurs we don't deem this a support polyhedra (e.g. top of shelf or table)
        if len(intersections) > 0:
            distances = np.linalg.norm((intersections - origins), axis=1)
            min_distance = np.min(distances)
            assert min_distance >= 0

            if min_distance >= trimesh.constants.tol.merge and min_distance <= max_height:
                if support_surface.polygon.geom_type == "MultiPolygon":
                    # This is probably due to the erosion operation when creating supports
                    return False, None

                if (min_distance - erosion_distance) > 0:
                    inscribing_polyhedra = trimesh.creation.extrude_polygon(
                        support_surface.polygon, min_distance - erosion_distance, engine="triangle"
                    )
                else:
                    return False, None

                if inscribing_polyhedra.volume >= min_volume:
                    return True, inscribing_polyhedra

        return False, None

    def _compute_support_polygons(
        self,
        geometry_node_name,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        min_area=0.01,
        **kwargs,
    ):
        """Extract support facets by comparing normals with gravity vector and checking area.

        Args:
            geometry_node_name (str): Name of the geometry node.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            gravity_tolerance (float, optional): Tolerance for comparsion between surface normals and gravity vector (dot product). Defaults to 0.5.
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.
            layer (str, optional): Layer name to search for support geometries. Defaults to 'collision'.
            **min_area (float): Minimum area of support facets [m^2]. Defaults to 0.01.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            list[trimesh.path.polygons.Polygon]: List of support polygons.
            list[np.ndarray]: List of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
            list[str]: List of node names that represent the reference frames for the transformations.
            list[int]: List of facet indices of the mesh that form the support polygon.
        """
        mesh_transform, geometry_name = self.graph[geometry_node_name]
        obj_mesh = self.geometry[geometry_name]

        # rotate gravity vector into mesh coordinates
        local_gravity = mesh_transform[:3, :3].T @ gravity

        # get all facets that are aligned with -local_gravity and bigger than min_area
        support_facet_indices = np.argsort(obj_mesh.facets_area)
        support_facet_indices = [
            idx
            for idx in support_facet_indices
            if np.isclose(
                tra.unit_vector(obj_mesh.facets_normal[idx]).dot(-tra.unit_vector(local_gravity)),
                1.0,
                atol=gravity_tolerance,
            )
            and obj_mesh.facets_area[idx] > min_area
        ]

        support_surfaces = []
        for index in support_facet_indices:
            normal = obj_mesh.facets_normal[index]
            origin = obj_mesh.facets_origin[index]

            facet_T = trimesh.geometry.plane_transform(origin, normal)
            facet_T_inv = trimesh.transformations.inverse_matrix(facet_T)

            facet_T_world = mesh_transform @ facet_T_inv

            if "min_x" in kwargs and facet_T_world[0, 3] < kwargs["min_x"]:
                continue
            if "min_y" in kwargs and facet_T_world[1, 3] < kwargs["min_y"]:
                continue
            if "min_z" in kwargs and facet_T_world[2, 3] < kwargs["min_z"]:
                continue
            if "max_x" in kwargs and facet_T_world[0, 3] > kwargs["max_x"]:
                continue
            if "max_y" in kwargs and facet_T_world[1, 3] > kwargs["max_y"]:
                continue
            if "max_z" in kwargs and facet_T_world[2, 3] > kwargs["max_z"]:
                continue

            vertices = trimesh.transform_points(obj_mesh.vertices, facet_T)[:, :2]

            # find boundary edges for the facet
            edges = obj_mesh.edges_sorted.reshape((-1, 6))[obj_mesh.facets[index]].reshape((-1, 2))
            group = trimesh.grouping.group_rows(edges, require_count=1)

            # run the polygon conversion
            polygons = trimesh.path.polygons.edges_to_polygons(
                edges=edges[group], vertices=vertices
            )

            for polygon in polygons:
                if polygon.geom_type == "MultiPolygon":
                    # This can be recursive!!
                    polys = list(polygon.geoms)
                else:
                    polys = [polygon]

                for uneroded_poly in polys:
                    # erode to avoid object on edges
                    eroded_poly = uneroded_poly.buffer(-erosion_distance)

                    if eroded_poly.geom_type == "MultiPolygon":
                        eroded_polys = list(eroded_poly.geoms)
                    else:
                        eroded_polys = [eroded_poly]

                    for poly in eroded_polys:
                        if not poly.is_empty and poly.area > min_area:
                            support_surfaces.append(
                                SupportSurface(
                                    polygon=poly,
                                    facet_index=index,
                                    node_name=geometry_node_name,
                                    transform=facet_T_inv,
                                )
                            )
        return support_surfaces

    def _get_support_polygons(
        self,
        layer="collision",
        surface_test=lambda surface: True,
        **kwargs,
    ):
        """Extract support facets by comparing normals with gravity vector and checking area.

        Args:
            layer (str, optional): Layer name to search for support geometries. Defaults to 'collision'.
            surface_test (function, optional): Function that tests whether a surface is valid.
            **obj_ids (str): Regular expression of object identifiers to use for finding supports.
            **geom_ids (str): Regular expression of geometry node names to use for finding supports.
            **kwargs: Keyword arguments that will be delegated to _compute_support_polygons.

        Returns:
            list[trimesh.path.polygons.Polygon]: List of support polygons.
            list[np.ndarray]: List of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
            list[str]: List of node names that represent the reference frames for the transformations.
            list[int]: List of facet indices of the mesh that form the support polygon.
        """
        if "obj_ids" in kwargs:
            obj_ids = utils.select_sublist(
                query=kwargs["obj_ids"], all_items=list(self.metadata["object_nodes"].keys())
            )
        else:
            obj_ids = list(self._scene.metadata["object_nodes"].keys())

        geometry_names = [item for name in obj_ids for item in self.get_geometry_names(obj_id=name)]
        if "geom_ids" in kwargs:
            x = re.compile(kwargs["geom_ids"])
            geometry_names = list(filter(x.search, geometry_names))

        # filter geometries based on their layer property
        geometry_names = [
            geom_name
            for geom_name in geometry_names
            if not "layer" in self._scene.geometry[self.graph[geom_name][1]].metadata
            or self._scene.geometry[self.graph[geom_name][1]].metadata["layer"] == layer
        ]
        if len(geometry_names) == 0:
            log.warning("Warning! No support meshes selected.")

        support_surfaces = []
        for geometry_name in geometry_names:
            support_surfaces.extend(self._compute_support_polygons(geometry_name, **kwargs))
        if surface_test is not None:
            support_surfaces = list(filter(surface_test, support_surfaces))
        return support_surfaces

    def label_part(self, label, geometry_name_regex):
        """Label specific geometries in the scene based on a regular expression of the name in the scene graph.

        Args:
            label (str): A label for a part.
            geometry_name_regex (str): A regular expression. All matched scene geometries will be referenced to this labeled part.

        Returns:
            list[str]: Returns all geometry names in the scene that match the regular expression.
        """
        x = re.compile(geometry_name_regex)

        # could use match() or findall() ?
        # search: only find first appearance
        # match: starts from beginning of string
        # geometry_names = list(filter(x.search, self._scene.geometry.keys()))
        geometry_names = list(filter(x.search, self._scene.graph.nodes_geometry))

        self._scene.metadata["parts"][label] = geometry_names

        return geometry_names

    def add_containment(self, label, parent_node_name, transform, extents):
        """Add box containment. Can be used for sampling poses.

        Args:
            label (str): Identifier of the containment.
            parent_node_name (str): Name of the scene graph node this containment will be attached to.
            transform (np.ndarray): 4x4 homogeneous matrix.
            extents (np.ndarray, list): 3-dimensional vector representing the box size.
        """
        self._scene.metadata["containers"][label] = [
            Container(
                geometry=trimesh.creation.box(extents=extents),
                node_name=parent_node_name,
                transform=transform,
            )
        ]

    def label_containment(
        self,
        label,
        min_area=0.01,
        min_volume=0.00001,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        distance_above_support=0.001,
        **kwargs,
    ):
        """Label containers in the scene. This is done by selecting support surfaces of the scene geometry and extruding volumes along the gravity direction until collision. Collision checking is done via sampling ray casts and might be imperfect depending on parameter settings. All parameters of the ``label_support`` function apply.

        Args:
            label (str): Identifier of the container volume.
            min_area (float, optional): Minimum support area of the container. Defaults to 0.01.
            min_volume (float, optional): Minimum volume of the container. Defaults to 0.00001.
            gravity (np.ndarray, optional): Gravity direction of support surface and extrusion direction of container. Defaults to np.array([0, 0, -1.0]).
            gravity_tolerance (float, optional): Tolerance for gravity vector when selecting support surfaces. Defaults to 0.1.
            erosion_distance (float, optional): Erosion distance of support surface. Defaults to 0.02.
            distance_above_support (float, optional): Offset of container above support surface. Defaults to 0.001.
            **obj_ids (str): Regular expression of object identifiers to use for finding supports.
            **geom_ids (str): Regular expression of geometry identifiers to use for finding supports.
            **min_area (float): Minimum area of support facets [m^2]. Defaults to 0.01.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            scene.Container: Container volume description.
        """
        _, support_data = self._get_support_polyhedra(
            support_surfaces=None,
            min_area=min_area,
            min_volume=min_volume,
            gravity=gravity,
            gravity_tolerance=gravity_tolerance,
            erosion_distance=erosion_distance,
            distance_above_support=distance_above_support,
            **kwargs,
        )

        if len(support_data) == 0:
            log.warning(f"No containers found for label '{label}'.")
        else:
            self._scene.metadata["containers"][label] = support_data

        return support_data

    def label_support(
        self,
        label,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        layer="collision",
        **kwargs,
    ):
        """Gives one or multiple support areas in the scene a string identifier which can be used for e.g. placement.

        Args:
            label (str): String identifier.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            gravity_tolerance (float, optional): Tolerance for comparsion between surface normals and gravity vector (dot product). Defaults to 0.5.
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.
            layer (str, optional): Layer name to search for support geometries. Defaults to 'collision'.
            **obj_ids (str): Regular expression of object identifiers to use for finding supports.
            **geom_ids (str): Regular expression of geometry identifiers to use for finding supports.
            **min_area (float): Minimum area of support facets [m^2]. Defaults to 0.01.
            **consider_support_polyhedra (bool): If set to True, will sample raycasts to ensure support surface has a "roof". Can be used to exclude top surfaces in shelves.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            list[trimesh.path.polygons.Polygon]: List of support polygons.
            list[np.ndarray]: List of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
            list[str]: List of node names that represent the reference frames for the transformations.
            list[int]: List of facet indices of the mesh that form the support polygon.
        """
        support_data = self._get_support_polygons(
            gravity=gravity,
            gravity_tolerance=gravity_tolerance,
            erosion_distance=erosion_distance,
            layer=layer,
            **kwargs,
        )

        if kwargs.get("consider_support_polyhedra", False):
            (
                is_support_polyhedra,
                _,
            ) = self._get_support_polyhedra(support_surfaces=support_data)
            log.info(
                f"Only {np.sum(is_support_polyhedra)}/{len(is_support_polyhedra)} support surfaces"
                " used for placing objects"
            )

            support_data = [s for (s, b) in zip(support_data, is_support_polyhedra) if b]

        if len(support_data) == 0:
            log.warning(f"No supports found for label '{label}'.")
        else:
            self._scene.metadata["support_polygons"][label] = support_data

        return support_data

    def in_collision(self, return_names=False, return_data=False, ignore_object_self_collisions=False, ignore_penetration_depth=None):
        """Check if any pair of geometries in the scene collide with one another.

        Args:
            return_names (bool, optional): If true, a set is returned containing the names of all pairs of geometries in collision. Defaults to False.
            return_data (bool, optional): If true, a list of ContactData is returned as well. Defaults to False.
            ignore_object_self_collisions (bool, optional): If true, any pair of colliding geometries that belongs to the same object is ignored. Defaults to False.
            ignore_penetration_depth (float, optional): Any collisions whose penetration depth is lower than this number is ignored. If None, nothing is ignored. Defaults to None.

        Returns:
            bool: True if a collision occurred between any pair of geometry in the scene and False otherwise.
            list[tuple[str, str]]: The set of pairwise collisions. Each tuple contains two names in alphabetical order indicating that the two corresponding geometries are in collision. Only returned if return_names = True.
            list[fcl.Contact]: List of all contacts detected. Only returned if return_data = True.
        """
        if ignore_object_self_collisions or ignore_penetration_depth is not None:
            _, names, contacts = self.collision_manager().in_collision_internal(return_names=True, return_data=True)
            
            collision = False
            for c in contacts:
                if ignore_penetration_depth is not None and c.depth < ignore_penetration_depth:
                    continue
                if ignore_object_self_collisions:
                    names_in_collision = list(c.names)
                    if names_in_collision[0].split('/')[0] == names_in_collision[1].split('/')[0]:
                        continue
                
                collision = True
            
            if return_names and return_data:
                return collision, names, contacts
            elif return_names:
                return collision, names
            elif return_data:
                return collision, contacts
            return collision

        return self.collision_manager().in_collision_internal(return_names=return_names, return_data=return_data)

    def in_collision_single(
        self,
        mesh,
        transform,
        min_distance=0.0,
        epsilon=1.0 / 1e3,
    ):
        """Check whether the scene is in collision with mesh. Optional: Define a minimum distance.

        Args:
            mesh (trimesh.Scene): Object Trimesh model to test with scene.
            transform (np.ndarray): Pose of the object mesh as a 4x4 homogenous matrix.
            min_distance (float, optional): Minimum distance that is considered in collision. Defaults to 0.0.
            epsilon (float, optional): Epsilon for minimum distance check. Defaults to 1.0/1e3.

        Returns:
            bool: Whether the object mesh is colliding with anything in the scene.
        """
        coll_mgr = self.collision_manager()
        colliding = coll_mgr.in_collision_single(mesh=mesh, transform=transform)

        if not colliding and min_distance > 0.0:
            distance = coll_mgr.min_distance_single(mesh=mesh, transform=transform)

            if distance < min_distance - epsilon:
                colliding = True

        return colliding

    def in_collision_other(
        self,
        other_manager,
        min_distance=0.0,
        epsilon=1.0 / 1e3,
        return_names=False,
        return_data=False,
    ):
        """Check whether the scene is in collision with another collision manager. Optional: Define a minimum distance.

        Args:
            other_manager (trimesh.collision.CollisionManager): Collision manager to test against.
            min_distance (float, optional): Minimum distance that is considered in collision. Defaults to 0.0.
            epsilon (float, optional): Epsilon for minimum distance check. Defaults to 1.0/1e3.
            return_names (bool, optional): If true, a set is returned containing the names of all pairs of objects in collision. Defaults to false
            return_data (bool, optional): If true, a list of ContactData is returned as well. Defaults to False.

        Returns:
            bool: Whether the object mesh is colliding with anything in the scene.
            set[2-tuples]: Set of 2-tuples of pairwaise collisions.
            list[ContactData]: List of contacts.
        """
        coll_mgr = self.collision_manager()
        colliding = coll_mgr.in_collision_other(
            other_manager=other_manager, return_names=return_names, return_data=return_data
        )

        if not colliding and min_distance > 0.0:
            distance = coll_mgr.min_distance_other(
                other_manager=other_manager, return_names=return_names, return_data=return_data
            )

            if distance < min_distance - epsilon:
                if return_names and return_data:
                    return True, distance[1], distance[2]
                elif return_names or return_data:
                    return True, distance[1]
                else:
                    return True

        if return_names and return_data:
            return colliding[0], colliding[1], colliding[2]
        elif return_names or return_data:
            return colliding[0], colliding[1]
        else:
            return colliding

    def container_generator(self, container_ids=None, sampling_fn=itertools.cycle):
        """A generator for iterating through the scene's labelled containers.

        Args:
            container_ids (list[str] or str, optional): A list or single container identifier. None will use all labelled containers. Defaults to None.
            sampling_fn (fn, optional): An iterator to be applied to the list of containers. Defaults to itertools.cycle.

        Returns:
            fn[scene.Container]: The resulting generator over a list of scene.Container.
        """
        if container_ids is None:
            support_data = sum(self._scene.metadata["containers"].values(), [])
        elif isinstance(container_ids, str):
            support_data = self._scene.metadata["containers"][container_ids]
        else:
            support_data = sum(list(map(self._scene.metadata["containers"].get, container_ids)), [])

        if len(support_data) == 0:
            raise ValueError(
                f"No container data to iterate over. Check container_ids={container_ids}"
            )

        def _generator(data):
            while True:
                yield sampling_fn(data)

        if isinstance(sampling_fn(support_data), Iterable):
            return sampling_fn(support_data)
        else:
            return _generator(support_data)

    def support_generator(self, support_ids=None, sampling_fn=itertools.cycle):
        """A generator for iterating through the scene's labelled support surfaces.

        Args:
            support_ids (list[str] or str, optional): A list or single support surface identifier. None will use all labelled surfaces. Defaults to None.
            sampling_fn (fn, optional): An iterator to be applied to the list of support surfaces. Defaults to itertools.cycle.

        Returns:
            fn[scene.SupportSurface]: The resulting generator over a list of scene.SupportSurface.
        """
        if support_ids is None:
            support_data = sum(self._scene.metadata["support_polygons"].values(), [])
        elif isinstance(support_ids, str):
            support_data = self._scene.metadata["support_polygons"][support_ids]
        else:
            support_data = sum(
                list(map(self._scene.metadata["support_polygons"].get, support_ids)), []
            )

        if len(support_data) == 0:
            raise ValueError(f"No support data to iterate over. Check support_ids={support_ids}")

        def _generator(data):
            while True:
                yield sampling_fn(data)

        if isinstance(sampling_fn(support_data), Iterable):
            return sampling_fn(support_data)
        else:
            return _generator(support_data)

    def place_object(
        self,
        obj_id,
        obj_asset,
        support_id=None,
        parent_id=None,
        obj_position_iterator=None,
        obj_orientation_iterator=None,
        obj_support_id_iterator=None,
        max_iter=100,
        distance_above_support=0.001,
        joint_type="floating",
        valid_placement_fn=lambda obj_asset, support, placement_T: True,
        **kwargs,
    ):
        """Add object by placing it in a non-colliding pose on top of a support surface or inside a container.

        Args:
            obj_id (str): Name of the object to place.
            obj_asset (scene.Asset): The asset that represents the object to be placed.
            support_id (str, optional): Defines the support that will be used for placing. Defaults to None.
            parent_id (str): Name of the object in the scene on which to place the object. Or None if any support surface works. Defaults to None.
            obj_position_iterator (iterator, optional): Iterator for sampling object positions in the support frame. Defaults to PositionIteratorUniform.
            obj_orientation_iterator (iterator, optional): Iterator for sampling object orientation in the object asset frame. Defaults to utils.orientation_generator_uniform_around_z.
            obj_support_id_iterator (iterator, optional): Iterator for sampling ids for support surface or container. Defaults to None.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "floating".
            valid_placement_fn (function, optional): Function for testing valid placements. Defaults to returning True.
            **use_collision_geometry (bool, optional): Defaults to default_use_collision_geometry.
            **kwargs: Keyword arguments that will be delegated to add_object.

        Raises:
            RuntimeError: In case the support_id does not exist.

        Returns:
            bool: Success.
        """
        if obj_position_iterator is None:
            obj_position_iterator = utils.PositionIteratorUniform(seed=self._rng)
        if obj_orientation_iterator is None:
            obj_orientation_iterator = utils.orientation_generator_uniform_around_z(seed=self._rng)

        if obj_support_id_iterator is None:
            if support_id is not None:
                if support_id not in self._scene.metadata["support_polygons"]:
                    raise RuntimeError(f"Support id '{support_id}' does not exist.")

                obj_support_id_iterator = utils.cycle_list(
                    self._scene.metadata["support_polygons"][support_id],
                    self._rng.permutation(
                        len(self._scene.metadata["support_polygons"][support_id])
                    ),
                )
            else:
                raise ValueError("Please pass in either support_id or obj_support_id_iterator")

        return self.place_objects(
            obj_id_iterator=itertools.repeat(obj_id),
            obj_asset_iterator=itertools.repeat(obj_asset, 1),
            obj_support_id_iterator=obj_support_id_iterator,
            obj_position_iterator=obj_position_iterator,
            obj_orientation_iterator=obj_orientation_iterator,
            parent_id=parent_id,
            max_iter=max_iter,
            distance_above_support=distance_above_support,
            joint_type=joint_type,
            valid_placement_fn=valid_placement_fn,
            **kwargs,
        )

    @property
    def keep_collision_manager_synchronized(self):
        """Whether collision manager synchronization (after each object add or removal) is off or on.

        Returns:
            bool: Whether collision manager synchronization is off or on.
        """
        return self._keep_collision_manager_synchronized

    @keep_collision_manager_synchronized.setter
    def keep_collision_manager_synchronized(self, value):
        """Turn collision manager synchronization (after each object add or removal) off/on.

        Args:
            value (bool): Whether to turn collision manager synchronization off or on.
        """
        self._keep_collision_manager_synchronized = bool(value)

    def synchronize_collision_manager(self, reset=False):
        """Synchronize the collision manager with the scene.

        Args:
            reset (bool, optional): Wheter to create BvH's from scratch. Defaults to False.
        """
        if not self._keep_collision_manager_synchronized:
            return

        if reset or self._collision_manager is None:
            self._collision_manager, _ = trimesh.collision.scene_to_collision(self._scene)
        else:
            names_coll_mgr = set(self._collision_manager._objs.keys())
            names_scene = set(self._scene.graph.nodes_geometry)

            # Add nodes not present in collision manager
            for node in names_scene.difference(names_coll_mgr):
                T, geometry_name = self._scene.graph[node]
                self._collision_manager.add_object(
                    name=node, mesh=self._scene.geometry[geometry_name], transform=T
                )

            # Remove nodes not present in scene
            for node in names_coll_mgr.difference(names_scene):
                self._collision_manager.remove_object(name=node)

            # Set changing transforms
            transforms = {node: self._scene.graph[node][0] for node in names_scene}
            utils.collision_manager_set_transforms(self._collision_manager, transforms)

        self._collision_manager_hash = hash(self._scene)

    def collision_manager(self):
        """Return a collision manager for this scene.

        Returns:
            trimesh.collision.CollisionManager: A collision manager of the scene.
        """
        if self._collision_manager_hash != hash(self._scene):
            self.synchronize_collision_manager(reset=True)
        return self._collision_manager

    def place_objects(
        self,
        obj_id_iterator,
        obj_asset_iterator,
        obj_support_id_iterator,
        obj_position_iterator,
        obj_orientation_iterator,
        parent_id=None,
        max_iter=100,
        max_iter_per_support=10,
        distance_above_support=0.002,
        joint_type="floating",
        valid_placement_fn=lambda obj_asset, support, placement_T: True,
        debug=False,
        **kwargs,
    ):
        """Add objects and place them in a non-colliding pose on top of a support surface or inside a container.

        Args:
            obj_id_iterator (iterator): Iterator for sampling name of the object to place.
            obj_asset_iterator (iterator): Iterator for sampling asset to be placed.
            obj_support_id_iterator (iterator, optional): Iterator for sampling ids for support surface or container. Defaults to None.
            obj_position_iterator (iterator, optional): Iterator for sampling object positions in the support frame.
            obj_orientation_iterator (iterator, optional): Iterator for sampling object orientation in the object asset frame.
            parent_id (str): Name of the node in the scene graph at which to attach the object. Or None if same as support node. Defaults to None.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            max_iter_per_support (int, optional): Maximum number of attempts per support surface. Defaults to 10.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.002.
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "floating".
            valid_placement_fn (function, optional): Function for testing valid placements. Defaults to returning True.
            debug (bool, optional): In case the placement is not valide, this will show a window with a scene with the attempted object placement. Defaults to False.

            **use_collision_geometry (bool, optional): Defaults to default_use_collision_geometry.
            **kwargs: Keyword arguments that will be delegated to add_object.

        Returns:
            bool: Success.
        """
        use_collision_geometry = kwargs.pop('use_collision_geometry', self._default_use_collision_geometry)

        success = True
        for obj_id, obj_asset in zip(obj_id_iterator, obj_asset_iterator):
            obj_coll_mgr, _ = trimesh.collision.scene_to_collision(
                obj_asset.as_trimesh_scene(use_collision_geometry=use_collision_geometry)
            )
            obj_coll_mgr_transforms = utils.collision_manager_get_transforms(obj_coll_mgr)

            try:
                # get support surface or volume
                if isinstance(obj_support_id_iterator, dict):
                    support_id_iterator = obj_support_id_iterator[obj_id]
                else:
                    support_id_iterator = obj_support_id_iterator

                if isinstance(obj_position_iterator, dict):
                    position_iterator = obj_position_iterator[obj_id]
                else:
                    position_iterator = obj_position_iterator

                if isinstance(obj_orientation_iterator, dict):
                    orientation_iterator = obj_orientation_iterator[obj_id]
                else:
                    orientation_iterator = obj_orientation_iterator

                iter = 0
                iter_per_support = 0
                support = next(support_id_iterator)
                position_iterator(support)
                while True:
                    pts = next(position_iterator)
                    pose = next(orientation_iterator)

                    # To avoid collisions with the support surface
                    pts3d = np.append(pts, distance_above_support) if pts.size == 2 else pts

                    # Transform plane coordinates into scene coordinates
                    placement_T = support.transform @ trimesh.transformations.translation_matrix(
                        pts3d
                    )
                    node_name_T = support.node_name

                    placement_T = placement_T @ pose

                    world_T = self._scene.graph.get(node_name_T)[0] @ placement_T

                    # Check custom validity function
                    is_valid = valid_placement_fn(obj_asset, support, world_T)

                    # Check collisions
                    utils.collision_manager_transform(obj_coll_mgr, world_T, premultiply=True)
                    is_valid &= not self.in_collision_other(
                        other_manager=obj_coll_mgr,
                        min_distance=distance_above_support,
                    )
                    utils.collision_manager_set_transforms(obj_coll_mgr, obj_coll_mgr_transforms)

                    iter += 1
                    iter_per_support += 1

                    if is_valid:
                        parent_id_name = node_name_T
                        if parent_id is not None:
                            parent_id_name = parent_id
                            parent_T = self.get_transform(node_name_T, frame_from=parent_id)
                            placement_T = parent_T @ placement_T

                        log.debug(f"Adding {obj_id} to {parent_id_name}")
                        self.add_object(
                            obj_id=obj_id,
                            asset=obj_asset,
                            parent_id=parent_id_name,
                            use_collision_geometry=use_collision_geometry,
                            transform=placement_T,
                            joint_type=joint_type,
                            **kwargs,
                        )
                        position_iterator.update(pts)
                        break
                    elif debug:
                        nmesh = obj_asset.mesh(use_collision_geometry=use_collision_geometry)
                        nmesh.apply_transform(world_T)
                        trimesh.Scene([self.scene, nmesh]).show()

                    if iter_per_support > max_iter_per_support:
                        # go to next support surface
                        iter_per_support = 0
                        support = next(support_id_iterator)
                        position_iterator(support)

                    if iter > max_iter:
                        success = False
                        log.warning(f"Couldn't place object {obj_id}!")
                        break
            except StopIteration:
                pass

        return success

    def asset(self, obj_id=None, try_to_remove_base_frame=True, **kwargs):
        """Create an asset of this scene or an object in the scene.
        Works well in conjuction with subscene.

        Args:
            obj_id (str, optional): If None, takes the entire scene. Defaults to None.
            try_to_remove_base_frame (bool, optional): Remove root node if possible. Defaults to True.

        Returns:
            TrimeshAsset or TrimeshSceneAsset: The scene as an asset.
        """
        if obj_id is None:
            asset_scene = self.copy()
        else:
            asset_scene = self.subscene(obj_ids=[obj_id], base_frame=obj_id)

        base_frame_removed = False
        if try_to_remove_base_frame:
            T_origin = tra.inverse_matrix(
                asset_scene.get_transform(
                    asset_scene.graph.transforms.children[asset_scene.graph.base_frame][0]
                )
            )

            if (
                len(asset_scene.graph.transforms.children[asset_scene.graph.base_frame]) == 1
                and len(asset_scene.graph.nodes) > 2
            ):
                asset_scene.remove_base_frame(keep_transform=False)
                base_frame_removed = True
        else:
            T_origin = np.eye(4)

        # remove namespaces
        node_mappings = [
            (old_name, "/".join(old_name.split("/")[1:]))
            for old_name in asset_scene.graph.nodes
            if "/" in old_name
        ]
        joint_mappings = [
            (old_name, "/".join(old_name.split("/")[1:]))
            for old_name in asset_scene.get_joint_names(include_fixed_floating_joints=True)
            if "/" in old_name
        ]
        geometry_mappings = [
            (old_name, "/".join(old_name.split("/")[1:]))
            for old_name in asset_scene.scene.geometry.keys()
            if "/" in old_name
        ]

        # check correctness of mappings -- if not use different mapping, preserving namespace
        if len(set([x for _, x in node_mappings])) != len(node_mappings):
            node_mappings = [
                (old_name, "_".join(old_name.split("/")))
                for old_name in asset_scene.graph.nodes
                if "/" in old_name
            ]
        if len(set([x for _, x in joint_mappings])) != len(joint_mappings):
            joint_mappings = [
                (old_name, "_".join(old_name.split("/")))
                for old_name in asset_scene.get_joint_names(include_fixed_floating_joints=True)
                if "/" in old_name
            ]
        if len(set([x for _, x in geometry_mappings])) != len(geometry_mappings):
            geometry_mappings = [
                (old_name, "_".join(old_name.split("/")))
                for old_name in asset_scene.scene.geometry.keys()
                if "/" in old_name
            ]

        asset_scene.rename_nodes(mappings=node_mappings)
        asset_scene.rename_joints(mappings=joint_mappings)
        asset_scene.rename_geometries(mappings=geometry_mappings)

        if try_to_remove_base_frame and not base_frame_removed:
            return TrimeshAsset(mesh=asset_scene.scene.dump(concatenate=True), **kwargs)

        return TrimeshSceneAsset(scene=asset_scene._scene, transform=T_origin, **kwargs)

    def move_object(self, obj_id, support_id=None, obj_support_id_iterator=None, **kwargs):
        """Move existing object in scene by placing it in a non-colliding pose on top of a support surface.
        Internally, creates another asset from the existing object, removes the existing object, and places the asset again.

        Args:
            obj_id (str): Name of the object to move.
            support_id (str, optional): Defines the support that will be used for placing. Defaults to None.
            obj_support_id_iterator (iterator, optional): Iterator for sampling ids for support surface or container. Defaults to None.
            **kwargs (optional): Arguments delegated to place_object(..).

        Raises:
            ValueError: The obj_id is not part of the scene.
        """
        if obj_id not in self.metadata["object_nodes"].keys():
            raise ValueError(f"Unknown obj_id: {obj_id}")

        if obj_support_id_iterator is None:
            if support_id is not None:
                obj_support_id_iterator = utils.cycle_list(
                    self._scene.metadata["support_polygons"][support_id],
                    self._rng.permutation(
                        len(self._scene.metadata["support_polygons"][support_id])
                    ),
                )
            else:
                raise ValueError("Please pass in either support_id or obj_support_id_iterator")

        obj_asset = self.subscene([obj_id], base_frame=obj_id).asset()

        self.remove_object(obj_id)
        self.place_object(
            obj_id=obj_id,
            obj_asset=obj_asset,
            obj_support_id_iterator=obj_support_id_iterator,
            **kwargs,
        )

    def _get_subtree(self, root_node, edges=None):
        if not edges:
            edges = self._scene.graph.to_edgelist()

        children = list(filter(lambda elements: elements[0] == root_node, edges))
        res = []
        for c in children:
            res.extend(self._get_subtree(root_node=c[1], edges=edges))

        return res + children

    def subscene(self, obj_ids, base_frame=None):
        """Return a part of the scene, made from a sub-graph of the original scene graph.

        Args:
            obj_ids (list[str]): Objects to be included in the subscene.
            base_frame (str, optional): The base frame of the resulting subscene. If None, it's the same as the base frame of self.

        Raises:
            ValueError: The base_frame doesn't exist in the scene.
            ValueError: The subgraphs overlap each other.

        Returns:
            trimesh.Scene: A new scene that contains a subset of the original objects.
        """
        subscene_graph = trimesh.scene.transforms.SceneGraph(
            base_frame=self._scene.graph.base_frame if base_frame is None else base_frame
        )

        if subscene_graph.base_frame not in self.graph.nodes:
            raise ValueError(
                f"Desired base_frame for subscene {subscene_graph.base_frame} doesn't exist in"
                " current scene."
            )

        # get all nodes that are part of desired objects
        nodes = []
        for k in obj_ids:
            if k in self.metadata["object_nodes"]:
                nodes.extend(self.metadata["object_nodes"][k])
            else:
                nodes.append(k)

        # add base_frame
        nodes = set(nodes + [subscene_graph.base_frame])

        # get every edge that has both included nodes
        edges = [e for e in self.graph.to_edgelist() if e[0] in nodes and e[1] in nodes]

        # check all obj_ids that don't have a predecessor
        for obj_id in obj_ids:
            if not any(obj_id == e[1] for e in edges):
                edge_data = {
                    "matrix": utils.get_transform(self.scene, obj_id, subscene_graph.base_frame)
                }
                if obj_id in self.scene.graph.nodes_geometry:
                    edge_data["geometry"] = self.scene.graph[obj_id][1]

                edges.append(
                    (
                        subscene_graph.base_frame,
                        obj_id,
                        edge_data,
                    )
                )

        # create a scene graph when
        subscene_graph.from_edgelist(edges)

        geometry_nodes = set(self.scene.graph.nodes_geometry).intersection(set(nodes))
        geometry_names = [
            self.scene.graph[g][1] for g in geometry_nodes if self.scene.graph[g][1] is not None
        ]
        geometry = {k: self.scene.geometry[k] for k in geometry_names}

        # Create metadata for subscene
        metadata = {"object_nodes": {}, "object_geometry_nodes": {}}
        for key in self.metadata["object_nodes"]:
            if key in obj_ids:
                metadata["object_nodes"][key] = self.metadata["object_nodes"][key].copy()
        for key in self.metadata["object_geometry_nodes"]:
            if key in obj_ids:
                metadata["object_geometry_nodes"][key] = self.metadata["object_geometry_nodes"][key].copy()

        return Scene(
            trimesh_scene=trimesh.Scene(geometry=geometry, graph=subscene_graph, metadata=metadata)
        )

    def colorize(
        self, specific_objects={}, specific_geometries={}, color=None, reset_visuals=True, **kwargs
    ):
        """Colorize meshes.

        Args:
            specific_objects (dict, optional): A dictionary of object id's to be colored. Defaults to {}.
            color (list[float]): The RGB color to be used. If None, a random color will be chosen. Defaults to None.
            reset_visuals (bool, optional): Whether to overwrite existing visuals of a geometry such as textures. Defaults to True.
            **brightness: Brightness of color. Defaults to 1.0.
            **transparency: Transparency of color. Defaults to 1.0.

        Returns:
            scene.Scene: The colorized scene.
        """
        if not specific_objects and not specific_geometries:
            for _, obj_mesh in self._scene.geometry.items():
                full_color = utils.adjust_color(color=color, seed=self._rng, **kwargs)
                log.debug(f"Colorize everything. Using color: {full_color}")
                # Some meshes don't have the face_color property.
                # In this case the method would crash.
                # We're just ignoring those meshes.
                # There's probably a nicer way to do this.
                try:
                    if not obj_mesh.visual.defined or reset_visuals:
                        obj_mesh.visual = trimesh.visual.create_visual(
                            mesh=obj_mesh,
                            face_colors=full_color,
                        )
                    elif obj_mesh.visual.kind == "face":
                        obj_mesh.visual.face_colors[:] = full_color
                except Exception as e:
                    log.warning(e)

        for obj_id, color in specific_objects.items():
            full_color = utils.adjust_color(color=color, seed=self._rng, **kwargs)
            log.debug(f"Colorize specific objects. Using color: {full_color}")
            try:
                geom_names = [self.graph[gn][1] for gn in self._scene.metadata["object_geometry_nodes"][obj_id]]
                for geometry_name in geom_names:
                    if not self._scene.geometry[geometry_name].visual.defined or reset_visuals:
                        self._scene.geometry[geometry_name].visual = trimesh.visual.create_visual(
                            mesh=self._scene.geometry[geometry_name],
                            face_colors=full_color,
                        )
                    elif self._scene.geometry[geometry_name].visual.kind == "face":
                        self._scene.geometry[geometry_name].visual.face_colors[:] = full_color
            except Exception as e:
                log.warning(e)

        for geometry_name, color in specific_geometries.items():
            full_color = utils.adjust_color(color=color, seed=self._rng, **kwargs)
            log.debug(f"Colorize specific geometries. Using color: {full_color}")
            try:
                if not self._scene.geometry[geometry_name].visual.defined or reset_visuals:
                    self._scene.geometry[geometry_name].visual = trimesh.visual.create_visual(
                        mesh=self._scene.geometry[geometry_name],
                        face_colors=full_color,
                    )
                elif self._scene.geometry[geometry_name].visual.kind == "face":
                    self._scene.geometry[geometry_name].visual.face_colors[:] = full_color
            except Exception as e:
                log.warning(e)

        return self

    def colorize_support(self, color=[0, 255, 0, 255]):
        """Colorize faces of meshes that belong to support areas.

        Args:
            color (list[float], optional): Color. Defaults to [0, 255, 0, 255].

        Returns:
            scene.Scene: The colorized scene.
        """
        for k in self._scene.metadata["support_polygons"]:
            for support_data in self._scene.metadata["support_polygons"][k]:
                geom_name = support_data.node_name
                geom = self._scene.geometry[geom_name]
                geom.visual = trimesh.visual.create_visual()
                geom.visual.face_colors[geom.facets[support_data.facet_index]] = color
        return self

    def colorize_parts(self, color=[0, 255, 0, 255]):
        """Colorize geometries that belong to parts.

        Args:
            color (list[float], optional): Color. Defaults to [0, 255, 0, 255].

        Returns:
            scene.Scene: The colorized scene.
        """
        for _, geom_names in self._scene.metadata["parts"].items():
            for geom_name in geom_names:
                self._scene.geometry[geom_name].visual = trimesh.visual.create_visual()
                self._scene.geometry[geom_name].visual.face_colors[:] = color
        return self

    def remove_visuals(self):
        """Remove all texture and color visuals. Objects in scene will be gray."""
        for _, g in self._scene.geometry.items():
            g.visual = trimesh.visual.create_visual()

    def get_colors(self):
        """Returns a dictionary of geometry identifiers and their the main color.
        Can be used in combination with colorize(specific_geometries={}).

        Returns:
            dict[str, np.ndarray]: Dictionary of geometry names and associated main color.
        """
        colors = {}
        for key, geom in self._scene.geometry.items():
            try:
                colors[key] = np.copy(geom.visual.main_color)
            except:
                # geometry does not have a main color
                pass
        return colors

    def get_visuals(self):
        """Returns the visual elements (color, texture) of all scene geometries.
        This is handy to store a color scheme for later reuse.

        Note:
            Do not use this across different scenes! The visuals contain
            face/vertex-specific information and are tied to the original
            geometry.

        Returns:
            dict[str, trimesh.visual.color.Visuals]: Dictionary of geometry names and associated visuals.
        """
        visuals = {}
        for key, geom in self._scene.geometry.items():
            visuals[key] = geom.visual.copy()
        return visuals

    def set_visuals(self, visuals):
        """Sets the visual elements (color, texture) of geometries in the scene.
        This is handy to reproduce a previous color scheme.

        Note:
            Do not apply visuals from a different scene! The visuals contain
            face/vertex-specific information and are tied to the original
            geometry.

        Args:
            visuals (dict[str, trimesh.visual.color.Visuals]): Dictionary of geometry names and associated visuals.
        """
        for key, visual in visuals.items():
            if key not in self._scene.geometry:
                log.warning(f"Couldn't find geometry '{key}'.")
            else:
                self._scene.geometry[key].visual = visual

    def _support_scene(
        self,
        scene,
        support_id_query=None,
        layer="support",
        color=None,
        add_to_base_frame=True,
        use_path_geometry=False,
        extruded_polygon_height=1e-3,
        covered=None,
    ):
        """Internal method to add support surfaces as geometry to a scene.

        Args:
            scene (trimesh.Scene): A trimesh scene to add geometry to.
            support_id_query (str or list or regex, optional): A string, list, or regular expression that refers to support IDs. None means all supports. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "support".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the supports to the scene's base frame or their original nodes. Defaults to True.
            use_path_geometry (bool, optional): Whether to use a trimesh.path.Path3D or an extruded polygon (trimesh.Trimesh) as geometry. Defaults to False (trimesh.Trimesh).
            extruded_polygon_height (bool, optional): If `use_path_geometry=False` this defines the height of the extrusion. Defaults to 1e-3.
            covered (bool, optional): Use only surfaces that are either covered or not covered.
        """
        if support_id_query is None:
            support_ids = list(self._scene.metadata["support_polygons"].keys())
        else:
            support_ids = utils.select_sublist(
                query=support_id_query,
                all_items=list(self._scene.metadata["support_polygons"].keys()),
            )

        for support_id in support_ids:
            support_color = utils.random_color(seed=self._rng) if color is None else color
            for i, support in enumerate(self._scene.metadata["support_polygons"][support_id]):
                if (covered is not None) and (support.covered != covered):
                    continue
                if use_path_geometry:
                    geometry = trimesh.load_path(support.polygon).to_3D()
                    geometry.colors = len(geometry.entities) * [support_color]
                else:
                    geometry = trimesh.creation.extrude_polygon(
                        support.polygon, height=extruded_polygon_height, engine="triangle"
                    )
                    geometry.visual.face_colors[:] = support_color

                geometry.metadata["layer"] = layer

                if add_to_base_frame:
                    scene.add_geometry(
                        node_name=f"{support_id}/support{i}",
                        geom_name=f"{support_id}/support{i}",
                        geometry=geometry,
                        transform=self.get_transform(support.node_name) @ support.transform,
                    )
                else:
                    scene.add_geometry(
                        node_name=f"{support_id}/support{i}",
                        geom_name=f"{support_id}/support{i}",
                        geometry=geometry,
                        transform=support.transform,
                        parent_node_name=support.node_name,
                    )

    def support_scene(
        self,
        support_id_query=None,
        layer="support",
        color=None,
        use_path_geometry=False,
        extruded_polygon_height=1e-3,
        **kwargs,
    ):
        """Return a trimesh.Scene that only contains the scene's labelled supports.

        Args:
            support_id_query (str or list or regex, optional): A string, list, or regular expression that refers to support IDs. None means all supports. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "support".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            use_path_geometry (bool, optional): Whether to use a trimesh.path.Path3D or an extruded polygon (trimesh.Trimesh) as geometry. Defaults to False (trimesh.Trimesh).
            extruded_polygon_height (bool, optional): If `use_path_geometry=False` this defines the height of the extrusion. Defaults to 1e-3.
            **kwargs: Additional keyword arguments that will be piped to _support_scene.

        Returns:
            trimesh.Scene: A trimesh scene with labelled supports.
        """
        s = trimesh.Scene(base_frame=self.graph.base_frame)
        self._support_scene(
            scene=s,
            support_id_query=support_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=True,
            use_path_geometry=use_path_geometry,
            extruded_polygon_height=extruded_polygon_height,
            **kwargs,
        )
        return s

    def add_supports_as_layer(
        self,
        support_id_query=None,
        layer="support",
        color=None,
        add_to_base_frame=True,
        use_path_geometry=False,
        extruded_polygon_height=1e-3,
        **kwargs,
    ):
        """Add all supports as geometries to the scene and put them into a layer.

        Args:
            support_id_query (str or list or regex, optional): A string, list, or regular expression that refers to support IDs. None means all supports. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "support".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the supports to the scene's base frame or their original nodes. Defaults to True.
            use_path_geometry (bool, optional): Whether to use a trimesh.path.Path3D or an extruded polygon (trimesh.Trimesh) as geometry. Defaults to False (trimesh.Trimesh).
            extruded_polygon_height (bool, optional): If `use_path_geometry=False` this defines the height of the extrusion. Defaults to 1e-3.
            **kwargs: Additional keyword arguments that will be piped to _support_scene.
        """
        self._support_scene(
            scene=self._scene,
            support_id_query=support_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=add_to_base_frame,
            use_path_geometry=use_path_geometry,
            extruded_polygon_height=extruded_polygon_height,
            **kwargs,
        )

    def _container_scene(
        self,
        scene,
        container_id_query=None,
        layer="container",
        color=None,
        add_to_base_frame=True,
    ):
        """Internal method to add containers as geometry to a scene.

        Args:
            scene (trimesh.Scene): A trimesh scene to add geometry to.
            container_id_query (str or list or regex, optional): A string, list, or regular expression that refers to container IDs. None means all containers. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "container".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the containers to the scene's base frame or their original nodes. Defaults to True.
        """
        if container_id_query is None:
            container_ids = list(self._scene.metadata["containers"].keys())
        else:
            container_ids = utils.select_sublist(
                query=container_id_query, all_items=list(self._scene.metadata["containers"].keys())
            )

        for container_id in container_ids:
            container_color = utils.random_color(seed=self._rng) if color is None else color
            for i, container in enumerate(self._scene.metadata["containers"][container_id]):
                container.geometry.visual.face_colors[:] = container_color
                container.geometry.metadata["layer"] = layer

                if add_to_base_frame:
                    scene.add_geometry(
                        node_name=f"{container_id}/container{i}",
                        geom_name=f"{container_id}/container{i}",
                        geometry=container.geometry,
                        transform=self.get_transform(container.node_name) @ container.transform,
                    )
                else:
                    scene.add_geometry(
                        node_name=f"{container_id}/container{i}",
                        geom_name=f"{container_id}/container{i}",
                        geometry=container.geometry,
                        transform=container.transform,
                        parent_node_name=container.node_name,
                    )

    def container_scene(
        self,
        container_id_query=None,
        layer="container",
        color=None,
    ):
        """Return a trimesh.Scene that only contains the scene's labelled containers.

        Args:
            container_id_query (str or list or regex, optional): A string, list, or regular expression that refers to container IDs. None means all containers. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "container".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.

        Returns:
            trimesh.Scene: A trimesh scene with labelled containers.
        """
        s = trimesh.Scene(base_frame=self.graph.base_frame)
        self._container_scene(
            scene=s,
            container_id_query=container_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=True,
        )
        return s

    def add_containers_as_layer(
        self,
        container_id_query=None,
        layer="container",
        color=None,
        add_to_base_frame=True,
    ):
        """Add all containers as geometries to the scene and put them into a layer.

        Args:
            container_id_query (str or list or regex, optional): A string, list, or regular expression that refers to container IDs. None means all containers. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "container".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the containers to the scene's base frame or their original nodes. Defaults to True.
        """
        self._container_scene(
            scene=self._scene,
            container_id_query=container_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=add_to_base_frame,
        )

    def _part_scene(
        self,
        scene,
        part_id_query=None,
        layer="part",
        color=None,
        add_to_base_frame=True,
    ):
        """Internal method to add parts as geometry to a scene.

        Args:
            scene (trimesh.Scene): A trimesh scene to add geometry to.
            part_id_query (str or list or regex, optional): A string, list, or regular expression that refers to part IDs. None means all parts. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "part".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the parts to the scene's base frame or their original nodes. Defaults to True.
        """
        if part_id_query is None:
            part_ids = list(self._scene.metadata["parts"].keys())
        else:
            part_ids = utils.select_sublist(
                query=part_id_query, all_items=list(self._scene.metadata["parts"].keys())
            )

        for part_id in part_ids:
            part_color = utils.random_color(seed=self._rng) if color is None else color
            for i, geom_node_name in enumerate(self._scene.metadata["parts"][part_id]):
                T, geom_name = self.graph[geom_node_name]

                part = self.geometry[geom_name].copy()
                part.visual = trimesh.visual.create_visual()
                part.visual.face_colors[:] = part_color
                part.metadata["layer"] = layer

                if add_to_base_frame:
                    scene.add_geometry(
                        node_name=f"{part_id}/part{i}",
                        geom_name=f"{part_id}/part{i}",
                        geometry=part,
                        # transform=self.get_transform(geom_name),
                        transform=T,
                    )
                else:
                    # Here we're using the first occurence of the geometry in the scene.
                    # In theory a geometry could appear multiple times.
                    # But to keep it consistent with the upper branch let's do this for now.
                    scene.add_geometry(
                        node_name=f"{part_id}/part{i}",
                        geom_name=f"{part_id}/part{i}",
                        geometry=part,
                        parent_node_name=geom_node_name,
                    )

    def part_scene(
        self,
        part_id_query=None,
        layer="part",
        color=None,
    ):
        """Return a trimesh.Scene that only contains the scene's labelled parts.

        Args:
            part_id_query (str or list or regex, optional): A string, list, or regular expression that refers to part IDs. None means all parts. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "part".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.

        Returns:
            trimesh.Scene: The scene with parts.
        """
        s = trimesh.Scene(base_frame=self.graph.base_frame)
        self._part_scene(
            scene=s,
            part_id_query=part_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=True,
        )
        return s

    def add_parts_as_layer(
        self,
        part_id_query=None,
        layer="part",
        color=None,
        add_to_base_frame=True,
    ):
        """Add all parts as geometries to the scene and put them into a layer.

        Args:
            part_id_query (str or list or regex, optional): A string, list, or regular expression that refers to part IDs. None means all parts. Defaults to None.
            layer (str, optional): The new geometry will be part of a new layer of this name. Defaults to "part".
            color (list[int], optional): An RGBA value defining the geometry's color. If none, random colors will be chosen. Defaults to None.
            add_to_base_frame (bool, optional): Whether to add the parts to the scene's base frame or their original nodes. Defaults to True.
        """
        self._part_scene(
            scene=self._scene,
            part_id_query=part_id_query,
            layer=layer,
            color=color,
            add_to_base_frame=add_to_base_frame,
        )

    def show_graph(
        self,
        layers=None,
        with_labels=True,
        edge_color="lightgrey",
        edge_color_joint="cyan",
        edge_color_joint_fixed="orange",
        edge_color_joint_floating="blue",
        edge_color_joint_prismatic="green",
        edge_color_joint_revolute="red",
        edge_color_joint_continuous="brown",
        node_color="lightgreen",
        font_size=10,
        node_size=80,
        **kwargs,
    ):
        """Plots the scene graph. Nicer layout than scene.graph.show().

        But requires networkx and pygraphviz:
        sudo apt-get install graphviz graphviz-dev
        python -m pip install pygraphviz

        Args:
            layers (list[str], optional): Which layers to show. None means every layer. Defaults to None.
            with_labels (bool, optional): Whether to display node names. Defaults to True.
            edge_color (str, optional): Color of graph edges. Defaults to "lightgrey".
            edge_color_joint (str, optional): Color of graph edges that constitute a joint. Defaults to "cyan".
            edge_color_joint_fixed (str, optional): Color of graph edges that constitute a fixed joint. Defaults to "orange".
            edge_color_joint_floating (str, optional): Color of graph edges that constitute a floating joint. Defaults to "blue".
            edge_color_joint_prismatic (str, optional): Color of graph edges that constitute a prismatic joint. Defaults to "green".
            edge_color_joint_revolute (str, optional): Color of graph edges that constitute a revolute joint. Defaults to "red".
            edge_color_joint_continuous (str, optional): Color of graph edges that constitute a continuous joint. Defaults to "brown".
            node_color (str, optional): Color of graph nodes. Defaults to "lightgreen".
            font_size (int, optional): Font size of node names. Defaults to 10.
            node_size (int, optional): Size of graph nodes. Defaults to 40.
            **kwargs: Keyword arguments that will be delegated to nx.draw().
        """
        # Third Party
        import matplotlib.pyplot as plt
        import networkx as nx

        edges = self._scene.graph.to_edgelist()
        if layers is not None:
            edges = [
                e
                for e in edges
                if "geometry" not in e[2]
                or self._scene.geometry[e[2]["geometry"]].metadata.get("layer", layers[0]) in layers
            ]

        G = nx.from_edgelist(edges, create_using=nx.DiGraph)
        edge_colors = []
        edge_color_joints = {
            "prismatic": edge_color_joint_prismatic if edge_color_joint_prismatic is not None else edge_color_joint,
            "revolute": edge_color_joint_revolute if edge_color_joint_revolute is not None else edge_color_joint,
            "continuous": edge_color_joint_continuous if edge_color_joint_continuous is not None else edge_color_joint,
            "floating": edge_color_joint_floating if edge_color_joint_floating is not None else edge_color_joint,
            "fixed": edge_color_joint_fixed if edge_color_joint_fixed is not None else edge_color_joint,
        }
        for edge in G.edges:
            scene_edge = [e for e in edges if e[0] == edge[0] and e[1] == edge[1]][0]
            if (
                EDGE_KEY_METADATA in scene_edge[2]
                and scene_edge[2][EDGE_KEY_METADATA] is not None
                and "joint" in scene_edge[2][EDGE_KEY_METADATA]
            ):
                color = edge_color_joints.get(scene_edge[2][EDGE_KEY_METADATA]["joint"]["type"], edge_color_joint)
                edge_colors.append(color)
            else:
                edge_colors.append(edge_color)

        try:
            pos = nx.nx_agraph.graphviz_layout(G=G, prog="twopi", root=self._scene.graph.base_frame)
        except ImportError as error:
            raise ImportError("This method requires pygraphviz (http://pygraphviz.github.io/).\nInstall with:\nsudo apt-get install graphviz graphviz-dev\npython -m pip install pygraphviz\n") from error
        
        nx.draw(
            G=G,
            pos=pos,
            with_labels=with_labels,
            node_size=node_size,
            font_size=font_size,
            edge_color=edge_colors,
            node_color=node_color,
            **kwargs,
        )
        from matplotlib.lines import Line2D
        handles_dict = {Line2D([], [], color=color, label=f"{joint_type} joint") for joint_type, color in edge_color_joints.items()}
        plt.legend(handles=handles_dict)
        plt.show()

    def show_supports(
        self,
        support_id_query=None,
        layers=None,
        color=None,
        use_path_geometry=False,
        extruded_polygon_height=1e-3,
    ):
        """Show labelled supports in trimesh viewer, optionally on top of scene.

        Note: If only supports need to be shown, use argument `layers=['support']`.

        Args:
            support_id_query (str or list or regex, optional): A string, list, or regular expression that refers to support IDs. None means all supports. Defaults to None.
            layers (list[str], optional): A list of visible layer names. If None everything will be visible. Defaults to None.
            color (list[int], optional): An RGBA value used to color the supports. If None, random color is chosen. Defaults to None.
            use_path_geometry (bool, optional): Whether to use a trimesh.path.Path3D or an extruded polygon (trimesh.Trimesh) as geometry. Defaults to False (trimesh.Trimesh).
            extruded_polygon_height (bool, optional): If `use_path_geometry=False` this defines the height of the extrusion. Defaults to 1e-3.
        """
        s = self.support_scene(
            support_id_query=support_id_query,
            color=color,
            use_path_geometry=use_path_geometry,
            extruded_polygon_height=extruded_polygon_height,
        )

        if layers == ["support"]:
            # This is significantly faster than hiding everything
            s.show()
        else:
            self.show(layers=layers, other_scene=s)

    def show_containers(self, container_id_query=None, layers=None, color=None):
        """Show labelled containers in trimesh viewer, optionally on top of scene.

        Note: If only containers need to be shown, use argument `layers=['container']`.

        Args:
            container_id_query (str or list or regex, optional): A string, list, or regular expression that refers to container IDs. None means all containers. Defaults to None.
            layers (list[str], optional): A list of visible layer names. If None everything will be visible. Defaults to None.
            color (list[int], optional): An RGBA value used to color the parts. If None, random color is chosen. Defaults to None.
        """
        s = self.container_scene(
            container_id_query=container_id_query,
            color=color,
        )

        if layers == ["container"]:
            # This is significantly faster than hiding everything
            s.show()
        else:
            self.show(layers=layers, other_scene=s)

    def show_parts(self, part_id_query=None, layers=None, color=None):
        """Show labelled parts in trimesh viewer, optionally on top of scene.

        Note: If only parts need to be shown, use argument `layers=['part']`.

        Args:
            part_id_query (str or list or regex, optional): A string, list, or regular expression that refers to part IDs. None means all parts. Defaults to None.
            layers (list[str], optional): A list of visible layer names. If None everything will be visible. Defaults to None.
            color (list[int], optional): An RGBA value used to color the parts. If None, random color is chosen. Defaults to None.
        """
        s = self.part_scene(
            part_id_query=part_id_query,
            color=color,
        )

        if layers == ["part"]:
            # This is significantly faster than hiding everything
            s.show()
        else:
            self.show(layers=layers, other_scene=s)

    def show(self, layers=None, other_scene=None):
        """Show scene using the trimesh viewer.

        Args:
            layers (list[str], optional): Filter to show only certain layers, e.g. 'visual' or 'collision'. Defaults to None, showing everything.
            other_scene (trimesh.Scene, optional): Another trimesh scene that will be appended to the scene itself. Defaults to None.

        Returns:
            trimesh.viewer.windowed.SceneViewer: The viewer.
        """
        scene_to_show = self._scene if other_scene is None else self._scene + other_scene
        viewer = trimesh.viewer.SceneViewer(
            scene=scene_to_show,
            start_loop=False,
        )

        if layers is not None and len(layers) > 0:
            for k, v in scene_to_show.geometry.items():
                if ("layer" in v.metadata and not v.metadata["layer"] in layers) or (
                    "layer" not in v.metadata and None not in layers
                ):
                    viewer.hide_geometry(node=k)

        _pyglet_app_run()

        return viewer

    def save_image(
        self,
        fname,
        resolution=[1920, 1080],
        visible=True,
        camera_transform=None,
        camera_rotation=None,
        **kwargs,
    ):
        """Save image of the current scene.

        Args:
            fname (str): Filename, including ending. If fname == 'bytes' the raw png bytes will be returned.
            resolution (list, optional): Resolution of the stored image. Defaults to [1920, 1080].
            visible (bool, optional): Whether to open the viewer during saving. This can avoid problems sometimes. Defaults to True.
            camera_transform (np.ndarray, optional): The 4x4 homogeneous matrix that represents the pose of the virtual camera which takes the image. This can be used to create images from the exact same viewpoint. Defaults to None.
            camera_rotation (np.ndarray, optional): Just the 3x3 rotation part of the homogeneous matrix. Defaults to None.

        Returns:
            image_data (bytes): The raw png bytes are returned if fname == 'bytes'. They can be turned into a PIL Image object via:
                                image = PIL.Image.open(io.BytesIO(image_data))
        """
        if camera_transform is not None:
            self._scene.camera_transform = camera_transform
        elif camera_rotation is not None:
            self._scene.camera_transform = self._scene.camera.look_at(
                self._scene.convex_hull.vertices, rotation=camera_rotation
            )

        png = self._scene.save_image(resolution=resolution, visible=visible, *kwargs)

        if fname == "bytes":
            return png

        with open(fname, "wb") as f:
            f.write(png)
            f.close()

    def get_layer_names(self):
        """Return the names of all layers that are currently present in the graph.
        Also returns None as an element, if a geometry is missing the layer metadata.

        Returns:
            set: A set of all layer names in the graph.
        """
        return {self._scene.geometry[k].metadata.get("layer", None) for k in self._scene.geometry}

    def set_layer_name(self, name, existing_layer=None):
        """Sets the layer name of all geometries. Or renames an existing layer.

        Args:
            name (str): New layer name.
            existing_layer (str, optional): If set only geometries belonging to this layer will be changed. Effectively renames an existing layer. Defaults to None.
        """
        for k in self._scene.geometry:
            geom = self._scene.geometry[k]
            if existing_layer is not None and geom.metadata["layer"] != existing_layer:
                continue

            geom.metadata["layer"] = name

    def remove_layer(self, name):
        """Remove all geometries that belong to a specific layer.

        Args:
            name (str): Name of the layer to be removed.
        """
        layer_found = False
        geometry_names = list(self._scene.geometry.keys())
        for geometry_name in geometry_names:
            if self._scene.geometry[geometry_name].metadata.get("layer", None) == name:
                layer_found = True
                self._scene.delete_geometry(geometry_name)
                self._scene.graph.transforms.remove_node(geometry_name)

                # update metadata
                obj_id = geometry_name.split("/")[0]

                if obj_id in self.metadata["object_nodes"]:
                    self.metadata["object_nodes"][obj_id].remove(geometry_name)
                    self.metadata["object_geometry_nodes"][obj_id].remove(geometry_name)

        if not layer_found:
            log.warning(f"No geometry belonging to layer '{name}' found. Nothing removed.")

    def add_convex_decomposition(self, input_layer=None, output_layer="collision"):
        """Add convex geometries for all geometries in a particular layer. Uses trimesh's VHACD interface.

        Args:
            input_layer (str, optional): Name of layer whose geometries will be decomposed. None considers all geometries.
            output_layer (str, optional): Name of layer convex geometries will belong to. Defaults to "collision".
        """
        # Otherwise will complain about mutation of scene graph during iteration
        geometry_names = list(self._scene.geometry.keys())
        for k in geometry_names:
            geom = self._scene.geometry[k]
            if geom.metadata.get("layer") != input_layer:
                continue

            convex_geometries = geom.convex_decomposition()
            if not isinstance(convex_geometries, Iterable):
                convex_geometries = [convex_geometries]

            for convex_geom in convex_geometries:
                convex_geom.metadata["layer"] = output_layer

            for n in self._scene.graph.geometry_nodes[k]:
                obj_id = n.split("/")[0]
                for convex_geom in convex_geometries:
                    geom_id = f"{obj_id}/{output_layer}_geometry_{str(uuid.uuid4())[:10]}"
                    self._scene.add_geometry(
                        geometry=convex_geom,
                        node_name=geom_id,
                        geom_name=geom_id,
                        parent_node_name=n,
                    )

                    # update metadata
                    self.metadata["object_nodes"][obj_id].append(geom_id)
                    self.metadata["object_geometry_nodes"][obj_id].append(geom_id)

    def _add_bounding_volumes(self, input_layer, output_layer, primitive_type):
        """Internal helper function used for all bounding approximations.

        Args:
            input_layer (str): Name of layer of considered geometries. None considers all geometries.
            output_layer (str): Name of layer of generated geometries.
            primitive_type (str): Type of bounding geometries: bounding_primitive, bounding_box, bounding_box_oriented, bounding_cylinder, bounding_sphere, convex_hull

        Raises:
            ValueError: Raised if primitive_type is unknown.
        """
        geometry_names = list(self._scene.geometry.keys())
        for k in geometry_names:
            geom = self._scene.geometry[k]
            if geom.metadata.get("layer") != input_layer:
                continue

            if primitive_type == "bounding_primitive":
                bounding_volume = geom.bounding_primitive.copy()
            elif primitive_type == "bounding_box":
                bounding_volume = geom.bounding_box.copy()
            elif primitive_type == "bounding_box_oriented":
                bounding_volume = geom.bounding_box_oriented.copy()
            elif primitive_type == "bounding_cylinder":
                bounding_volume = geom.bounding_cylinder.copy()
            elif primitive_type == "bounding_sphere":
                bounding_volume = geom.bounding_sphere.copy()
            elif primitive_type == "convex_hull":
                bounding_volume = geom.convex_hull.copy()
            else:
                raise ValueError(
                    f"Unknown value for primitive_type='{primitive_type}'. Needs to be one of:"
                    " bounding_primitive, bounding_box, bounding_box_oriented, bounding_cylinder,"
                    " bounding_sphere, convex_hull"
                )

            bounding_volume.metadata["layer"] = output_layer

            for n in self._scene.graph.geometry_nodes[k]:
                obj_id = n.split("/")[0]
                geom_id = f"{obj_id}/{output_layer}_geometry_{str(uuid.uuid4())[:10]}"
                self._scene.add_geometry(
                    geometry=bounding_volume,
                    node_name=geom_id,
                    geom_name=geom_id,
                    parent_node_name=n,
                )

                # update metadata
                self.metadata["object_nodes"][obj_id].append(geom_id)
                self.metadata["object_geometry_nodes"][obj_id].append(geom_id)

    def add_voxel_decomposition(self, input_layer, output_layer="voxels", pitch=0.01):
        """Add voxels for all geometries in a particular layer.
        Note, that these voxels will not be aligned to a common grid, but with each geometry's coordinate system.

        Args:
            input_layer (str): Name of layer whose geometries will be voxelized. None considers all geometries.
            output_layer (list, optional): Name of layer voxels will belong to. Defaults to "voxels".
            pitch (float, optional): Edge length of a single voxel. Defaults to 0.01.
        """
        geometry_names = list(self._scene.geometry.keys())
        for k in geometry_names:
            geom = self._scene.geometry[k]
            if geom.metadata.get("layer") != input_layer:
                continue

            voxel_grid = geom.voxelized(pitch=pitch)

            voxel_geom = voxel_grid.as_boxes()
            voxel_geom.metadata["layer"] = output_layer

            for n in self._scene.graph.geometry_nodes[k]:
                obj_id = n.split("/")[0]
                geom_id = f"{obj_id}/{output_layer}_geometry_{str(uuid.uuid4())[:10]}"
                self._scene.add_geometry(
                    geometry=voxel_geom,
                    node_name=geom_id,
                    geom_name=geom_id,
                    parent_node_name=n,
                )

                # update metadata
                self.metadata["object_nodes"][obj_id].append(geom_id)
                self.metadata["object_geometry_nodes"][obj_id].append(geom_id)

    def add_bounding_boxes(self, input_layer, output_layer="bboxes"):
        """Add aligned bounding boxes for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate bounding boxes. None considers all geometries.
            output_layer (str, optional): Name of layer bounding boxes will belong to. Defaults to "bboxes".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="bounding_box",
        )

    def add_bounding_boxes_oriented(self, input_layer, output_layer="obb"):
        """Add oriented bounding boxes for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate bounding boxes. None considers all geometries.
            output_layer (str, optional): Name of layer bounding boxes will belong to. Defaults to "obb".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="bounding_box_oriented",
        )

    def add_bounding_cylinders(self, input_layer, output_layer="bcylinders"):
        """Add bounding cylinders for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate bounding cylinders. None considers all geometries.
            output_layer (str, optional): Name of layer bounding cylinders will belong to. Defaults to "bcylinders".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="bounding_cylinder",
        )

    def add_bounding_spheres(self, input_layer, output_layer="bspheres"):
        """Add bounding spheres for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate bounding spheres. None considers all geometries.
            output_layer (str, optional): Name of layer bounding spheres will belong to. Defaults to "bspheres".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="bounding_sphere",
        )

    def add_bounding_primitives(self, input_layer, output_layer="bprimitive"):
        """Add bounding primitives (cylinder, sphere, box - chose the one with smallest volume) for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate bounding primitives. None considers all geometries.
            output_layer (str, optional): Name of layer bounding primitives will belong to. Defaults to "bprimitive".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="bounding_primitive",
        )

    def add_convex_hulls(self, input_layer, output_layer="chull"):
        """Add convex hulls for all geometries in a particular layer.

        Args:
            input_layer (str): Name of layer whose geometries will be used to calculate convex hulls. None considers all geometries.
            output_layer (str, optional): Name of layer convex hulls will belong to. Defaults to "chull".
        """
        self._add_bounding_volumes(
            input_layer=input_layer,
            output_layer=output_layer,
            primitive_type="convex_hull",
        )

    def explode(self, distance, direction=None, origin=None) -> None:
        """
        Explode the current scene in-place around a point and vector.

        Parameters
        -----------
        vector : (3,) float or float
           Explode radially around a direction vector or spherically
        origin : (3,) float
          Point to explode around
        """
        if origin is None:
            origin = self._scene.centroid
        # if vector is None:
        #     vector = self._scene.scale / 25.0
        vector = np.asanyarray(distance, dtype=np.float64)
        origin = np.asanyarray(origin, dtype=np.float64)

        for obj_id in self.metadata["object_nodes"]:
            transform, _ = self.graph.get(obj_id, frame_from=None)

            # transform, geometry_name = self._scene.graph[node_name]
            # centroid = self._scene.geometry[geometry_name].centroid
            centroid = np.zeros(3)
            # transform centroid into nodes location
            centroid = np.dot(transform, np.append(centroid, 1))[:3]

            if vector.shape == ():
                # case where our vector is a single number
                offset = (centroid - origin) * vector
            elif np.shape(vector) == (3,):
                projected = np.dot(vector, (centroid - origin))
                offset = vector * projected
            else:
                raise ValueError("explode vector wrong shape!")

            # original transform is read-only
            T_new = transform.copy()
            T_new[:3, 3] += offset
            self._scene.graph[obj_id] = T_new
        
        # update stuff

    @classmethod
    def load(cls, fname):
        """Loads scene from file.

        Args:
            fname (str): File name of scene. Currently only json is supported. (generate via `export(...)`)

        Returns:
            scene.Scene: The loaded scene object.
        """
        if fname.endswith(".json"):
            scene_data = json.load(open(fname, "r"))

            trimesh_scene = trimesh.exchange.load.load(scene_data)

            return cls(trimesh_scene=trimesh_scene)
        else:
            raise NotImplementedError("Unknown scene file format.")

    def remove_lights(self):
        """Remove all lights in the scene. Note, that lights might even exist without explicitly adding them (added automatically)."""
        while self._scene.lights:
            self._scene.lights.pop()

    def unwrap_geometries(self, query=None):
        """Adds uv textures to geometries. Primitives will be converted to meshes.
        the vertices have been assigned uv texture coordinates. Vertices may be
        duplicated by the unwrapping algorithm.
        This uses xatlas internally (requires `pip install xatlas`).

        Args:
            query (str or list[str]): A geometry name or list of names or regular expression. None will include all geometries. Defaults to None.
        """
        if query is None:
            node_names = set(self.scene.graph.nodes_geometry)
        elif type(query) is list or type(query) is tuple:
            node_names = []
            for k in query:
                if k in self.metadata["object_nodes"]:
                    node_names.extend(self.metadata["object_nodes"][k])
                else:
                    node_names.append(k)
            node_names = set(self.scene.graph.nodes_geometry).intersection(set(node_names))
        else:
            node_names = utils.select_sublist(
                query=query, all_items=self.scene.graph.nodes_geometry
            )
        
        for n in node_names:
            self.geometry[n] = self.geometry[n].unwrap()



    def export(self, fname=None, file_type=None, **kwargs):
        """Export scene as either: glb, gltf, obj, stl, ply, dict, json, urdf, usd, or usda.

        Args:
            fname (str, optional): Filename. If None, data is returned instead of written to disk. Defaults to None.
            file_type (str, optional): File format. If file_type is None, the type is deducted from the filename. If fname is None, file_type needs to be specified.
            **kwargs (optional): The export is highly configurable, each format has different parameters and flags. For details, see the format-specific docstrings in exchange.export_*.

        Returns:
            str or bytes or dict (optional): If fname is None, data will be returned in the specified file_type.
        """
        # (This assignment creates the camera)
        # There is a slightly better solution, see trimesh issue in github.
        foo = self._scene.camera
        foo = self._scene.lights

        return export.export_scene(
            scene=self,
            file_obj=fname,
            file_type=file_type,
            **kwargs,
        )
