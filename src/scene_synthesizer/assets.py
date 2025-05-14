# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Standard Library
import os
import subprocess

# Third Party
import numpy as np
import trimesh
import trimesh.transformations as tra
import trimesh.viewer
import yourdfpy

# Local Folder
from . import constants
from . import utils
from .utils import log
from .trimesh_utils import compute_stable_poses

try:
    # Third Party
    from pyglet.app import run as _pyglet_app_run
except BaseException as E:
    _pyglet_app_run = utils.late_bind_exception(E)


def asset_generator(fnames, **kwargs):
    """A simple generator that runs through a list of filenames and returns an Asset for each one.

    Args:
        fnames (list[str]): A list of file names.

    Yields:
        assets.Asset: An asset.
    """
    for fname in fnames:
        yield Asset(fname=fname, **kwargs)

class Asset(object):
    """The asset base class."""

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str) or "fname" in kwargs:
            fname = kwargs["fname"] if "fname" in kwargs else args[0]
            fname = fname.lower()
            if fname.endswith(".urdf"):
                return super(Asset, cls).__new__(URDFAsset)
            elif (
                fname.endswith(".usd")
                or fname.endswith(".usda")
                or fname.endswith(".usdc")
                or fname.endswith(".usdz")
            ):
                return super(Asset, cls).__new__(USDAsset)
            elif fname.endswith(".xml"):
                return super(Asset, cls).__new__(MJCFAsset)
            elif fname.endswith(".scad"):
                return super(Asset, cls).__new__(OpenSCADAsset)
            else:
                return super(Asset, cls).__new__(MeshAsset)
        return super(Asset, cls).__new__(cls)

    def scene(self, obj_id="object", **kwargs):
        """Return a scene consisting of only this asset.

        Args:
            obj_id (str, optional): Name of object in scene. Defaults to 'object'.
            **kwargs: Additional keyword arguments that will be piped to the add_object method.

        Returns:
            scene.Scene: A scene
        """
        # SRL
        from scene_synthesizer.scene import Scene

        # Pass the asset seed to make scene construction deterministic
        seed = getattr(self, "_rng", None)

        return Scene.single_object_scene(asset=self, obj_id=obj_id, seed=seed, **kwargs)

    def mesh(self, use_collision_geometry=False):
        """Return a trimesh.Trimesh object of the asset.

        Args:
            use_collision_geometry (bool, optional): Whether to use the collision or visual geometry or both. Defaults to False.

        Returns:
            trimesh.Trimesh: A trimesh mesh.
        """
        return self.as_trimesh_scene(use_collision_geometry=use_collision_geometry).dump(
            concatenate=True
        )

    def compute_stable_poses(
        self,
        convexify=False,
        center_mass=None,
        sigma=0.0,
        n_samples=1,
        threshold=0.0,
        tolerance_zero_extent=1e-6,
        use_collision_geometry=True,
    ):
        """Wrapper for trimesh.poses.compute_stable_poses function.

        Args:
            convexify (bool, optional): Whether to use the convex hull of the object.
            center_mass ((3,) float, optional): The object center of mass. If None, this method assumes uniform density and watertightness and computes a center of mass explicitly. Defaults to None.
            sigma (float, optional): The covariance for the multivariate gaussian used to sample center of mass locations. Defaults to 0.0.
            n_samples (int, optional): The number of samples of the center of mass location. Defaults to 1.
            threshold (float, optional): The probability value at which to threshold returned stable poses. Defaults to 0.0.
            tolerance_zero_extent (float, optional): The threshold for considering a dimension to have zero length. In this case, trimesh.poses.compute_stable_poses gets caught in an infinite loop. We avoid this by specifying the stable poses to be along zero-length dimensions. Defaults to 1e-6.
            use_collision_geometry (bool, optional): Whether to use the collision geometry or visual geometry to calculate stable poses. Defaults to True.

        Returns:
            transforms ((n, 4, 4) float): The homogeneous matrices that transform the object to rest in a stable pose.
            probs ((n,) float): Probability in (0, 1) for each pose
        """
        if not hasattr(self, "_stable_poses") or self._stable_poses is None:
            mesh = self.as_trimesh_scene(use_collision_geometry=use_collision_geometry).dump(
                concatenate=True
            )

            if convexify:
                # Qhull fails if one of the extents is (close to) zero.
                # In this case we're dealing with a surface.
                zero_extents = mesh.extents < tolerance_zero_extent
                if any(zero_extents):
                    mesh = mesh.bounding_box
                elif not mesh.is_convex:
                    mesh = mesh.convex_hull

            # trimesh.poses.compute_stable_poses has trouble with meshes
            # whose extents are close to zero (in any dimension)
            # we will check for those and return stable poses accordingly
            zero_extents = mesh.extents < tolerance_zero_extent
            if any(zero_extents):
                if sum(zero_extents) == 1:
                    vector_a = np.eye(3)[zero_extents][0]
                    stable_transforms = np.array(
                        [
                            trimesh.geometry.align_vectors(vector_a, [0.0, 0.0, 1.0]),
                            trimesh.geometry.align_vectors(vector_a, [0.0, 0.0, -1.0]),
                        ]
                    )
                    stable_probs = np.array([0.5, 0.5])
                elif sum(zero_extents) == 2:
                    num_orientations = len(mesh.faces)
                    not_zero_extent = [not elem for elem in zero_extents]
                    vector_a = np.eye(3)[not_zero_extent][0]
                    theta = np.linspace(0.0, 2.0 * np.pi, num=num_orientations)
                    circle_points = np.column_stack(
                        (np.sin(theta), -np.cos(theta), np.zeros(num_orientations))
                    )
                    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector_a)
                    transformed_points = trimesh.transform_points(
                        matrix=transform, points=circle_points
                    )

                    new_X = np.cross(transformed_points, np.tile(vector_a, (num_orientations, 1)))
                    new_Y = np.tile(vector_a, (num_orientations, 1))
                    new_Z = transformed_points

                    stable_transforms = np.tile(np.eye(4), (num_orientations, 1, 1))
                    stable_transforms[:, 0, :3] = new_X
                    stable_transforms[:, 1, :3] = new_Y
                    stable_transforms[:, 2, :3] = new_Z

                    stable_probs = np.array([1.0 / len(theta)] * len(theta))
                else:
                    # all three dims are close to zero
                    num_orientations = len(mesh.faces)
                    stable_transforms = trimesh.transformations.random_rotation_matrix(
                        num=num_orientations
                    )

                    stable_probs = np.array([1.0 / num_orientations] * num_orientations)
                self._stable_poses = (stable_transforms, stable_probs)
                return self._stable_poses

            if center_mass is None:
                center_mass = utils.get_mass_properties(mesh)[1]

            # We're using a slightly modified version of trimesh's
            # compute_stable_poses which avoids infinite loops in case
            # the CoM is outside the convex hull
            self._stable_poses = compute_stable_poses(
                mesh=mesh,
                center_mass=center_mass,
                sigma=sigma,
                n_samples=n_samples,
                threshold=threshold,
            )

        return self._stable_poses

    def sample_stable_pose(self, seed=None, **kwargs):
        """Return a stable pose according to their likelihood.

        Returns:
            np.ndarray: homogeneous 4x4 matrix
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        """
        rng = np.random.default_rng(seed)

        if not hasattr(self, "_stable_poses") or self._stable_poses is None:
            self.compute_stable_poses(**kwargs)

        poses, probabilities = self._stable_poses
        if len(poses) == 0:
            raise RuntimeError(f"Unable to detect any stable poses for {self}.")
        probabilities = np.array(probabilities) / sum(
            probabilities
        )

        index = rng.choice(len(poses), p=probabilities)
        inplane_rot = tra.rotation_matrix(angle=rng.uniform(0, 2.0 * np.pi), direction=[0, 0, 1])
        return inplane_rot.dot(poses[index])

    def __str__(self):
        """Return a readable string for this asset.
        It is constructed by taking the class name, removing any occurrence of 'Asset', and converting to snake_case.

        Returns:
            str: Readable string of this asset type.
        """
        if hasattr(self, "_fname") and len(self._fname) > 0:
            return os.path.basename(self._fname)
        if hasattr(self, "_name") and len(self._name) > 0:
            return self._name
        return utils.snake_case("".join(type(self).__name__.split("Asset")))

    def show(self, use_collision_geometry=False, layers=None):
        """Display the asset via the trimesh scene viewer.

        Args:
            use_collision_geometry (bool, optional): Which geometry to show: visual or collision geometry. Defaults to False.
            layers (list[str], optional): Filter to show only certain layers, e.g. 'visual' or 'collision'. Defaults to None, showing everything.
        """
        scene = self.as_trimesh_scene(use_collision_geometry=use_collision_geometry)

        viewer = trimesh.viewer.SceneViewer(
            scene=scene,
            start_loop=False,
        )

        if layers is not None and len(layers) > 0:
            for k, v in scene.geometry.items():
                if "layer" in v.metadata and not v.metadata["layer"] in layers:
                    viewer.hide_geometry(node=k)

        _pyglet_app_run()

        return viewer

    def _apply_arguments(self, **kwargs):
        """Apply keyword arguments provided through constructor."""
        return

    def _get_scale(self, raw_extents):
        scale = 1.0

        # calculate extents according to desired orientation
        # make sure this is equivalent to the operations in _get_origin_transform
        rotation = np.eye(3)
        if (
            "front" in self._attributes
            and "up" in self._attributes
            and len(self._attributes["front"]) == 3
            and len(self._attributes["up"]) == 3
        ):
            rotation[1, :3] = self._attributes["front"]
            rotation[2, :3] = self._attributes["up"]
            if abs(rotation[1, :3].dot(rotation[2, :3])) > self._attributes.get(
                "tolerance_up_front_orthogonality",
                constants.DEFAULT_TOLERANCE_UP_FRONT_ORTHOGONALITY,
            ):
                raise ValueError("Vector 'up' and 'front' are not orthogonal!")
            rotation[0, :3] = np.cross(rotation[1, :3], rotation[2, :3])
        elif "front" in self._attributes and len(self._attributes["front"]) == 3:
            rotation = trimesh.geometry.align_vectors(self._attributes["front"], [0, 1, 0])[:3, :3]
        elif "up" in self._attributes and len(self._attributes["up"]) == 3:
            rotation = trimesh.geometry.align_vectors(self._attributes["up"], [0, 0, 1])[:3, :3]

        extents = np.abs(rotation.dot(raw_extents))

        if (
            ("width" in self._attributes and "width_scale" in self._attributes)
            or ("depth" in self._attributes and "depth_scale" in self._attributes)
            or ("height" in self._attributes and "height_scale" in self._attributes)
        ):
            raise ValueError(
                "Asset can only have either {width,depth,height} or {width,depth,height}_scale not"
                " both for the same dimension."
            )

        if "scale" in self._attributes:
            if (
                "width_scale" in self._attributes
                or "depth_scale" in self._attributes
                or "height_scale" in self._attributes
            ):
                raise ValueError("Can't use {width,depth,scale}_scale if scale is used.")

            try:
                scale = float(self._attributes["scale"])
            except TypeError:
                scale = [float(x) for x in self._attributes["scale"]]
                assert len(scale) == 3
        elif "size" in self._attributes:
            if (
                "width_scale" in self._attributes
                or "depth_scale" in self._attributes
                or "height_scale" in self._attributes
            ):
                raise ValueError("Can't use {width,depth,scale}_scale if size is used.")

            assert len(self._attributes["size"]) == 3
            scale = self._attributes["size"] / extents
        elif "extents" in self._attributes:
            if (
                "width_scale" in self._attributes
                or "depth_scale" in self._attributes
                or "height_scale" in self._attributes
            ):
                raise ValueError("Can't use {width,depth,scale}_scale if extents is used.")

            assert len(self._attributes["extents"]) == 3
            scale = self._attributes["extents"] / extents
        elif (
            "height" in self._attributes
            and "depth" in self._attributes
            and "width" in self._attributes
        ):
            scale = (
                np.array(
                    [
                        float(self._attributes["width"]),
                        float(self._attributes["depth"]),
                        float(self._attributes["height"]),
                    ]
                )
                / extents
            )
        elif "height" in self._attributes and "depth" in self._attributes:
            scale_height = float(self._attributes["height"]) / extents[2]
            scale_depth = float(self._attributes["depth"]) / extents[1]
            scale = np.array(
                [
                    max(scale_depth, scale_height),
                    scale_depth,
                    scale_height,
                ]
            )
        elif "height" in self._attributes and "width" in self._attributes:
            scale_width = float(self._attributes["width"]) / extents[0]
            scale_height = float(self._attributes["height"]) / extents[2]
            scale = np.array(
                [
                    scale_width,
                    max(scale_width, scale_height),
                    scale_height,
                ]
            )
        elif "depth" in self._attributes and "width" in self._attributes:
            scale_width = float(self._attributes["width"]) / extents[0]
            scale_depth = float(self._attributes["depth"]) / extents[1]
            scale = np.array(
                [
                    scale_width,
                    scale_depth,
                    max(scale_width, scale_depth),
                ]
            )
        elif "height" in self._attributes:
            scale = float(self._attributes["height"]) / extents[2]
        elif "depth" in self._attributes:
            scale = float(self._attributes["depth"]) / extents[1]
        elif "width" in self._attributes:
            scale = float(self._attributes["width"]) / extents[0]
        elif "max_length" in self._attributes:
            scale = float(self._attributes["max_length"]) / extents.max()
        elif "min_length" in self._attributes:
            scale = float(self._attributes["min_length"]) / extents.min()
        elif "max_width_depth" in self._attributes:
            scale = float(self._attributes["max_width_depth"]) / extents[:2].max()

        if "width_scale" in self._attributes:
            scale[0] = self._attributes["width_scale"]
        if "depth_scale" in self._attributes:
            scale[1] = self._attributes["depth_scale"]
        if "height_scale" in self._attributes:
            scale[2] = self._attributes["height_scale"]

        # rotate scale vector if needed
        if hasattr(scale, "__len__") and ("front" in self._attributes or "up" in self._attributes):
            scale = np.abs(rotation.dot(scale))

        return scale

    def _get_origin_transform(self, bounds, center_mass, centroid):
        origin = np.eye(4)

        if self._attributes.get("origin", None) is not None:
            origin = tra.inverse_matrix(
                utils.get_reference_frame(
                    bounds=bounds,
                    center_mass=center_mass,
                    centroid=centroid,
                    x=self._attributes["origin"][0],
                    y=self._attributes["origin"][1],
                    z=self._attributes["origin"][2],
                )
            )

        if (
            "front" in self._attributes
            and "up" in self._attributes
            and len(self._attributes["front"]) == 3
            and len(self._attributes["up"]) == 3
        ):
            T_target = np.eye(4)
            T_target[1, :3] = self._attributes["front"]
            T_target[2, :3] = self._attributes["up"]
            if abs(T_target[1, :3].dot(T_target[2, :3])) > self._attributes.get(
                "tolerance_up_front_orthogonality",
                constants.DEFAULT_TOLERANCE_UP_FRONT_ORTHOGONALITY,
            ):
                raise ValueError("Vector 'up' and 'front' are not orthogonal!")
            T_target[0, :3] = np.cross(T_target[1, :3], T_target[2, :3])

            origin = T_target @ origin

        elif "front" in self._attributes and len(self._attributes["front"]) == 3:
            origin = trimesh.geometry.align_vectors(self._attributes["front"], [0, 1, 0]) @ origin
        elif "up" in self._attributes and len(self._attributes["up"]) == 3:
            origin = trimesh.geometry.align_vectors(self._attributes["up"], [0, 0, 1]) @ origin

        if "transform" in self._attributes:
            origin = origin @ np.array(self._attributes["transform"])

        return origin

    def _calculate_stable_poses(self):
        if "stable_poses" in self._attributes:
            poses = self._attributes["stable_poses"]
            if "stable_pose_probs" in self._attributes:
                probs = self._attributes["stable_pose_probs"]
            else:
                probs = np.ones(len(poses)) / len(poses)

            self._stable_poses = (poses, probs)

    def _get_mass_properties(self):
        if "mass" in self._attributes and "density" in self._attributes:
            raise ValueError(
                "Can't define 'mass' and 'density' of an asset. Define either one or the other."
            )

        mass = self._attributes.get("mass", None)
        density = self._attributes.get("density", None)
        center_mass = self._attributes.get("center_mass", None)

        return mass, density, center_mass

    def get_bounds(self, query=None, frame=None, use_collision_geometry=None):
        """Return bounds of asset defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this asset. None means entire asset. Defaults to None.
            frame (str, optional): The reference frame to use. None means asset's base frame is used. Defaults to None.
            use_collision_geometry (bool, optional): Whether to use collision geometry, visual geometry or both (if None). Defaults to None.

        Returns:
            np.ndarray: A 2x3 matrix of minimum and maximum coordinates for each dimension.
        """
        trimesh_scene = self.as_trimesh_scene(
            namespace="", use_collision_geometry=use_collision_geometry
        )

        # select geometry according to query
        if query is None:
            node_names = set(trimesh_scene.graph.nodes_geometry)
        else:
            node_names = utils.select_sublist(
                query=query, all_items=trimesh_scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        all_bounds = []
        for n in node_names:
            T, geomn = trimesh_scene.graph.get(n)
            bounds_w = trimesh.transform_points(trimesh_scene.geometry[geomn].bounds, T)
            all_bounds.append(bounds_w)
        all_bounds = np.vstack(all_bounds)

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            all_bounds = trimesh.transform_points(all_bounds, T)

        return np.array([np.min(all_bounds, axis=0), np.max(all_bounds, axis=0)])

    def get_extents(self, query=None, frame=None, use_collision_geometry=None):
        """Return extents of asset defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this asset. None means entire asset. Defaults to None.
            frame (str, optional): The reference frame to use. None means asset's base frame is used. Defaults to None.
            use_collision_geometry (bool, optional): Whether to use collision geometry, visual geometry or both (if None). Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the extents of each dimension.
        """
        return np.diff(
            self.get_bounds(
                query=query, frame=frame, use_collision_geometry=use_collision_geometry
            ),
            axis=0,
        )[0]

    def get_center_mass(self, query=None, frame=None, use_collision_geometry=None):
        """Return center of mass for subscene defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this asset. None means entire asset. Defaults to None.
            frame (str, optional): The reference frame to use. None means asset's base frame is used. Defaults to None.
            use_collision_geometry (bool, optional): Whether to use collision geometry, visual geometry or both (if None). Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the center of mass of the queried subscene.
        """
        trimesh_scene = self.as_trimesh_scene(
            namespace="", use_collision_geometry=use_collision_geometry
        )

        if query is None:
            node_names = set(trimesh_scene.graph.nodes_geometry)
        else:
            node_names = utils.select_sublist(
                query=query, all_items=trimesh_scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        result = utils.center_mass(trimesh_scene=trimesh_scene, node_names=node_names)

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            result = tra.translation_from_matrix(T @ tra.translation_matrix(result))

        return result

    def get_centroid(self, query=None, frame=None, use_collision_geometry=None):
        """Return centroid for asset defined through nodes selected by query.

        Args:
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this asset. None means entire asset. Defaults to None.
            frame (str, optional): The reference frame to use. None means asset's base frame is used. Defaults to None.
            use_collision_geometry (bool, optional): Whether to use collision geometry, visual geometry or both (if None). Defaults to None.

        Returns:
            np.ndarray: A 3-vector describing the centroid of the queried subscene.
        """
        trimesh_scene = self.as_trimesh_scene(
            namespace="", use_collision_geometry=use_collision_geometry
        )

        if query is None:
            node_names = set(trimesh_scene.graph.nodes_geometry)
        else:
            node_names = utils.select_sublist(
                query=query, all_items=trimesh_scene.graph.nodes_geometry
            )

        if len(node_names) == 0:
            raise ValueError("No geometry selected. Check your 'query' argument.")

        total_area = 0.0
        total_centroid = np.zeros(3)
        for n in node_names:
            T, geomn = trimesh_scene.graph.get(n)
            area = trimesh_scene.geometry[geomn].area
            total_centroid += (
                area * (T @ np.append(trimesh_scene.geometry[geomn].centroid, 1.0))[:3]
            )
            total_area += area
        result = total_centroid / total_area

        if frame is not None:
            T = utils.homogeneous_inv(self.get_transform(frame))
            result = tra.translation_from_matrix(T @ tra.translation_matrix(result))

        return result

    def get_reference_frame(self, xyz, query=None, frame=None, use_collision_geometry=None):
        """Return reference frame for subscene defined through nodes selected by query.

        Args:
            xyz (tuple[str]): A 3-tuple/list of ['top', 'center', 'bottom', 'com', 'centroid']
            query (list[str] or str): A list, string, or regular expression referring to a subset of all geometry of this asset. None means entire asset. Defaults to None.
            frame (str, optional): The reference frame to use. None means scene's base frame is used. Defaults to None.
            use_collision_geometry (bool, optional): Whether to use collision geometry, visual geometry or both (if None). Defaults to None.

        Raises:
            ValueError: Unknown reference string.

        Returns:
            np.ndarray: A 4x4 homogenous matrix.
        """
        translation = np.zeros(3)

        # calculate only necessary stuff
        if any(keyword in xyz for keyword in ("top", "center", "bottom", "left", "right", "front", "back")):
            bounds = self.get_bounds(
                query=query, frame=frame, use_collision_geometry=use_collision_geometry
            )
        if "com" in xyz:
            center_mass = self.get_center_mass(
                query=query, frame=frame, use_collision_geometry=use_collision_geometry
            )
        if "centroid" in xyz:
            centroid = self.get_centroid(
                query=query, frame=frame, use_collision_geometry=use_collision_geometry
            )

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

    def as_trimesh_scene(self, namespace="object", use_collision_geometry=True):
        if self._attributes.get("reference_only", False):
            s = trimesh.Scene(base_frame=namespace)
            dummy_geometry = self._attributes.get("reference_only_geometry", trimesh.creation.box())
            s.add_geometry(node_name=f"{namespace}/dummy", geom_name=f'{namespace}/dummy', parent_node_name=namespace, geometry=dummy_geometry)
            
            utils.add_filename_to_trimesh_metadata(
                mesh_or_scene=s,
                fname=self._fname,
            )
            return s
    
        trimesh_scene = self._as_trimesh_scene(
            namespace=namespace, use_collision_geometry=use_collision_geometry
        )

        trimesh_scene = utils.normalize_and_bake_scale(trimesh_scene)

        # scale asset
        scale = self._get_scale(raw_extents=trimesh_scene.extents)
        scaled_scene = utils.scaled_trimesh_scene(trimesh_scene, scale=scale)

        # add origin transform to root node
        if "center_mass" in self._attributes:
            center_mass = self._attributes["center_mass"]
        else:
            center_mass = utils.center_mass(trimesh_scene=scaled_scene, node_names=scaled_scene.graph.nodes_geometry)
        origin = self._get_origin_transform(
            bounds=scaled_scene.bounds,
            center_mass= center_mass,
            centroid=scaled_scene.centroid,
        )

        for child in scaled_scene.graph.transforms.children[scaled_scene.graph.base_frame]:
            T_child = scaled_scene.graph.get(child)[0]
            attrib = scaled_scene.graph.transforms.edge_data[(scaled_scene.graph.base_frame, child)]
            attrib["matrix"] = origin @ T_child
            scaled_scene.graph.update(
                frame_from=scaled_scene.graph.base_frame,
                frame_to=child,
                **attrib,
            )
        utils.invalidate_scenegraph_cache(scaled_scene)

        # apply mass / density / inertial properties
        mass, density, center_mass = self._get_mass_properties()
        if mass is not None:
            volume = sum(
                utils.get_mass_properties(geom)[3] for geom in scaled_scene.geometry.values()
            )
            for geom in scaled_scene.geometry.values():
                geom.density = mass / volume
        elif density is not None:
            for geom in scaled_scene.geometry.values():
                geom.density = density
        if center_mass is not None:
            coms = utils.distribute_center_mass(
                center_mass=center_mass, geoms=scaled_scene.geometry.values()
            )

            # convert coms to individual reference frames
            for geom_name, com in zip(scaled_scene.geometry, coms):
                T, geomn = scaled_scene.graph.get(geom_name)
                geom = scaled_scene.geometry[geomn]
                geom.metadata["center_mass"] = (utils.homogeneous_inv(T) @ np.append(com, 1.0))[:3]

        return scaled_scene


class MeshAsset(Asset):
    def __init__(self, fname, **kwargs):
        """A MeshAsset is a loadable file that describes geometry/ies and make up scenes. These are pure triangular mesh assets.

        Args:
            fname (str): File name to load the asset.
            **scale (float): Scale of the asset. Defaults to 1.0.
            **size (list[float]): 3D size of asset.
            **height (float): Height of the asset (length in z-dimension).
            **width (float): Width of the asset (length in x-dimension).
            **depth (float): Depth of the asset (length in y-dimension).
            **max_length (float): No dimension of the asset will exceed a max value. At least one dimension will be of max_length.
            **min_length (float): No dimension of the asset will be less than a min value. At least one dimension will be of min_length.
            **max_width_depth (float): The x/y dimensions of the asset will not exceed the max value. Either x, y, or both will be of max_width_depth.
            **origin (list[str]): Reference point. 3-dimensional vector with each element of ['top', 'bottom', 'left', 'right', 'front', 'back', 'center', 'com', 'centroid'].
            **front (list[float]): Affects -y direction of orientation. 3-dimensional unit vector describing the main interactive direction of an object.
            **up (list[float]): Affects z direction of orientation. 3-dimensional unit vector describing the up direction of an object.
            **tolerance_up_front_orthogonality (float, optional): Tolerance for checking whether front and up are orthogonal. Defaults to constants.TOLERANCE_UP_FRONT_ORTHOGONALITY (=1e-7).
            **transform (np.ndarray): Homogeneous transform which shifts the origin. Defaults to the identity matrix.
            **force_mesh (bool): In case of URDF files: force each loaded geometry to be concatenated into a single one (will result in one node in the resulting scene graph - but might lose texture information). Caution: This will also lose the metadata that points to the source of the original geometry file - export without writing out new mesh files won't work. Defaults to False.
            **split_obj (bool): Split meshes at each `o` declared in obj file.
        """
        self._fname = fname
        self._origin = np.eye(4)
        self._attributes = kwargs

        if 'name' in kwargs:
            self._name = kwargs['name']

        self._stable_poses = None

        self._model = self._load(
            fname,
            force_mesh=kwargs.get("force_mesh", False),
            split_obj=kwargs.get("split_obj", False),
        )

    def _load(self, fname, force_mesh=False, split_obj=False):
        log.info(f"Loading asset {fname}")

        mesh = trimesh.load(
            fname,
            force="scene",
            split_object=split_obj,  # split_object doesn't exist anymore since trimesh 12/2022
            group_material=not split_obj,
        )

        utils.add_filename_to_trimesh_metadata(mesh_or_scene=mesh, fname=fname)
        utils.add_extents_to_trimesh_metadata(mesh_or_scene=mesh)

        return mesh

    def _as_trimesh_scene(self, namespace="object", use_collision_geometry=True):
        if not isinstance(self._model, trimesh.Scene):
            # should never happen
            raise ValueError("This asset's _model is not of type trimesh.Scene.")

        s = trimesh.Scene(base_frame=namespace)

        cnt = 0
        for name in self._model.graph.nodes_geometry:
            T, geom_name = self._model.graph.get(name)
            mesh = self._model.geometry[geom_name]

            new_name = name
            # This creates problems when exporting as USD
            # scene paths can't start with a number
            if name[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                new_name = f"geometry_{cnt}"
                cnt += 1

            copied_mesh = mesh.copy()
            copied_mesh.visual = mesh.visual.copy()

            s.add_geometry(
                node_name=f"{namespace}/{new_name}",
                geom_name=f"{namespace}/{new_name}",
                parent_node_name=s.graph.base_frame,
                geometry=copied_mesh,
                transform=T,
            )

        return s


class URDFAsset(Asset):
    def __init__(self, fname, **kwargs):
        """A file-based asset, loaded from a URDF file.

        Args:
            fname (str): File name.

        Raises:
            ValueError: Raises exception if file doesn't exist.
        """
        self._fname = fname
        self._stable_poses = None

        self._init_default_attributes(**kwargs)

        self._model = self._load(
            fname,
            force_mesh=kwargs.get("force_mesh", False),
            filename_handler=kwargs.get("filename_handler", None),
        )

        if kwargs.get("ignore_articulation", False):
            raise NotImplementedError(f"ignore_articulation=True is not implemented for URDF assets.")

        if "configuration" in kwargs:
            if isinstance(kwargs["configuration"], str):
                cfg_str = kwargs["configuration"].lower()
                if cfg_str == "upper":
                    cfg = np.array(
                        [
                            self._joint_limit_upper(self._model.joint_map[joint_name])
                            for joint_name in self._model.actuated_joint_names
                        ]
                    )
                elif cfg_str == "lower":
                    cfg = np.array(
                        [
                            self._joint_limit_lower(self._model.joint_map[joint_name])
                            for joint_name in self._model.actuated_joint_names
                        ]
                    )
                else:
                    raise ValueError(
                        "If argument `configuration` is a string, it can only be 'upper' or"
                        f" 'lower' (currently: {cfg_str})."
                    )
            else:
                cfg = np.array(kwargs["configuration"])

            if len(self._configuration) != len(cfg):
                raise ValueError(
                    f"Length of argument `configuration` {len(cfg)} does not match DoFs of asset"
                    f" {(len(self._configuration))}."
                )

            self._configuration = cfg

    def _init_default_attributes(self, **kwargs):
        self._origin = np.eye(4)
        self._attributes = kwargs
        
        self._default_joint_limit_lower = kwargs.get(
            "default_joint_limit_lower", constants.DEFAULT_JOINT_LIMIT_LOWER
        )
        self._default_joint_limit_upper = kwargs.get(
            "default_joint_limit_upper", constants.DEFAULT_JOINT_LIMIT_UPPER
        )
        self._default_joint_limit_velocity = kwargs.get(
            "default_joint_limit_velocity", constants.DEFAULT_JOINT_LIMIT_VELOCITY
        )
        self._default_joint_limit_effort = kwargs.get(
            "default_joint_limit_effort", constants.DEFAULT_JOINT_LIMIT_EFFORT
        )
        self._default_joint_stiffness = kwargs.get(
            "default_joint_stiffness", constants.DEFAULT_JOINT_STIFFNESS
        )
        self._default_joint_damping = kwargs.get(
            "default_joint_damping", constants.DEFAULT_JOINT_DAMPING
        )

    def _load(self, fname, force_mesh=False, filename_handler=None):
        log.debug(f"Loading asset {fname}")

        urdf_model = yourdfpy.URDF.load(
            fname,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=force_mesh,
            force_collision_mesh=force_mesh,
            filename_handler=filename_handler,
        )
        self._configuration = np.zeros(len(urdf_model.actuated_joint_names))

        return urdf_model

    def _joint_limit_upper(self, joint):
        return (
            self._default_joint_limit_upper
            if joint.limit is None or joint.limit.upper is None
            else joint.limit.upper
        )

    def _joint_limit_lower(self, joint):
        return (
            self._default_joint_limit_lower
            if joint.limit is None or joint.limit.lower is None
            else joint.limit.lower
        )

    def _as_trimesh_scene(self, namespace="object", use_collision_geometry=True):
        s = trimesh.Scene(base_frame=namespace)

        self._model.update_cfg(self._configuration)

        urdf_scene = utils.create_yourdfpy_scene(
            self._model, use_collision_geometry=use_collision_geometry
        )

        # copy nodes and edges from original scene graph
        # and change identifiers by prepending a namespace
        edges = []
        for a, b, attr in urdf_scene.graph.to_edgelist():
            if "geometry" in attr:
                attr["geometry"] = f"{namespace}/{attr['geometry']}"

            # rename nodes with additional namespace
            edges.append((f"{namespace}/{a}", f"{namespace}/{b}", attr))

        # add base link
        edges.append(
            (
                s.graph.base_frame,
                f"{namespace}/{urdf_scene.graph.base_frame}",
                {
                    "matrix": self._origin,
                    constants.EDGE_KEY_METADATA: {
                        "joint": {
                            "name": f"{namespace}/origin_joint",
                            "type": "fixed",
                        }
                    },
                },
            )
        )

        # copy geometries
        geometry = {}
        for k, v in urdf_scene.geometry.items():
            if isinstance(v, trimesh.primitives.Primitive):
                geometry[f"{namespace}/{k}"] = v.copy()
            else:
                geometry[f"{namespace}/{k}"] = v.copy(include_cache=True)
            # This is required in the latest trimesh version, see
            # https://github.com/mikedh/trimesh/issues/1989
            geometry[f"{namespace}/{k}"].density = v.density
            geometry[f"{namespace}/{k}"].metadata = v.metadata.copy()
        
        s.graph.from_edgelist(edges, strict=True)
        s.geometry.update(geometry)

        # extract articulation information
        for joint in self._model.robot.joints:
            parent_node_name = f"{namespace}/{joint.parent}"
            node_name = f"{namespace}/{joint.child}"

            assert (
                parent_node_name,
                node_name,
            ) in s.graph.transforms.edge_data, f"{parent_node_name} -> {node_name} not in graph"

            # set joint props that are not read by default
            limit_velocity = (
                self._default_joint_limit_velocity
                if joint.limit is None or joint.limit.velocity is None
                else joint.limit.velocity
            )
            limit_effort = (
                self._default_joint_limit_effort
                if joint.limit is None or joint.limit.effort is None
                else joint.limit.effort
            )
            limit_upper = self._joint_limit_upper(joint)
            limit_lower = self._joint_limit_lower(joint)

            damping = self._default_joint_damping if joint.dynamics is None or joint.dynamics.damping is None else joint.dynamics.damping

            joint_property_dict = {
                "name": f"{namespace}/{joint.name}",
                "type": joint.type,
                "q": self._configuration[self._model.actuated_joints.index(joint)]
                if joint in self._model.actuated_joints
                else 0.0,
                "axis": joint.axis.tolist()
                if not joint.type == "fixed"
                else [1.0, 0, 0],
                "origin": joint.origin.tolist(),
                "limit_velocity": limit_velocity,
                "limit_effort": limit_effort,
                "limit_lower": limit_lower,
                "limit_upper": limit_upper,
            }

            if damping is not None:
                joint_property_dict['damping'] = damping

            # add articulation data as edge attributes
            s.graph.transforms.edge_data[(parent_node_name, node_name)].update(
                {
                    constants.EDGE_KEY_METADATA: {
                        "joint": joint_property_dict
                    }
                }
            )

        return s


class USDAsset(Asset):
    def __init__(self, fname, **kwargs):
        """A file-based asset, loaded from a USD file.

        Args:
            fname (str): USD file name.
            **ignore_articulation (bool, optional): Will ignore any joints. Defaults to False.
            **default_joint_limit_upper (float, optional): Defaults to scene_synthesizer.constants.DEFAULT_JOINT_LIMIT_UPPER.
            **default_joint_limit_lower (float, optional): Defaults to scene_synthesizer.constants.DEFAULT_JOINT_LIMIT_LOWER.
            **default_joint_limit_velocity (float, optional): Defaults to scene_synthesizer.constants.DEFAULT_JOINT_LIMIT_VELOCITY.
            **default_joint_limit_effort (float, optional): Defaults to scene_synthesizer.constants.DEFAULT_JOINT_LIMIT_EFFORT.
            **reference_only (bool, optional): Do not load actual file, only keep a reference to it. Defaults to False.
            **reference_only_geometry (trimesh.Trimesh, optional): Dummy geometry to represent the referenced file. Only used if `reference_only==True`. Defaults to trimesh.creation.box().
            **configuration (list[float] or str, optional): Configuration of articulation. Can be specified as a list of floats or as 'upper'/'lower' which will use the joint limits.

        Raises:
            ValueError: If configuration argument is a string that is not 'upper' or 'lower'.
        """
        # Local Folder
        from . import usd_import

        # load USD file
        self._fname = fname
        self._attributes = kwargs
        self._stage = usd_import.Usd.Stage.Open(fname)

        self._ignore_prim_paths = ["/groundPlane", "/groundPlane/CollisionMesh"]

        self._default_joint_limit_lower = kwargs.get(
            "default_joint_limit_lower", constants.DEFAULT_JOINT_LIMIT_LOWER
        )
        self._default_joint_limit_upper = kwargs.get(
            "default_joint_limit_upper", constants.DEFAULT_JOINT_LIMIT_UPPER
        )
        self._default_joint_limit_velocity = kwargs.get(
            "default_joint_limit_velocity", constants.DEFAULT_JOINT_LIMIT_VELOCITY
        )
        self._default_joint_limit_effort = kwargs.get(
            "default_joint_limit_effort", constants.DEFAULT_JOINT_LIMIT_EFFORT
        )
        self._default_joint_stiffness = kwargs.get(
            "default_joint_stiffness", constants.DEFAULT_JOINT_STIFFNESS
        )
        self._default_joint_damping = kwargs.get(
            "default_joint_damping", constants.DEFAULT_JOINT_DAMPING
        )

        self._configuration = None

        self._ignore_articulation = kwargs.get("ignore_articulation", False)

        if "configuration" in kwargs:
            if isinstance(kwargs["configuration"], str):
                cfg_str = kwargs["configuration"].lower()
                if cfg_str not in ["lower", "upper"]:
                    raise ValueError(
                        "If argument `configuration` is a string, it can only be 'upper' or"
                        f" 'lower' (currently: {cfg_str})."
                    )
                self._configuration = cfg_str
            else:
                cfg = np.array(kwargs["configuration"])
                self._configuration = cfg

    def _as_trimesh_scene(self, namespace="object", use_collision_geometry=True):
        s = trimesh.Scene(base_frame=namespace)

        ignore_world_joints = True

        def _usd_prim_path_to_node_name(path):
            if len(path.pathString) == 0 or path.pathString == "/":
                return s.graph.base_frame
            tmp = path.pathString.replace("/", "_")
            if tmp.startswith("_"):
                tmp = tmp[1:]
            return f"{namespace}/{tmp}"

        # Local Folder
        from . import usd_import

        stage = usd_import.get_stage(self._fname)

        # collect joint information
        joints = {}
        num_actuated_joints = 0
        if not self._ignore_articulation:
            for joint_path in usd_import.get_scene_paths(
                stage=stage,
                prim_types=["PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint"],
            ):
                joint_prim = self._stage.GetPrimAtPath(joint_path)

                body_0_targets = joint_prim.GetRelationship("physics:body0").GetTargets()
                body_1_targets = joint_prim.GetRelationship("physics:body1").GetTargets()

                if len(body_0_targets) == 0 and len(body_1_targets) == 0:
                    raise ValueError(f"Joint '{joint_path}' has two undefined body targets.")

                if ignore_world_joints and (len(body_0_targets) == 0 or len(body_1_targets) == 0):
                    continue

                body_0 = (
                    body_0_targets[0]
                    if len(body_0_targets) > 0
                    else self._stage.GetPseudoRoot().GetPath()
                )
                body_1 = (
                    body_1_targets[0]
                    if len(body_1_targets) > 0
                    else self._stage.GetPseudoRoot().GetPath()
                )

                joints[(body_0, body_1)] = joint_prim

                joints[(body_1, body_0)] = "reversed"

            num_actuated_joints = len(
                usd_import.get_scene_paths(
                    stage=stage,
                    prim_types=["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"],
                )
            )

        if self._configuration is not None and not isinstance(self._configuration, str):
            if num_actuated_joints != len(self._configuration):
                raise ValueError(
                    f"Length of argument `configuration` {len(self._configuration)} does not match"
                    f" DoFs of asset ({num_actuated_joints})."
                )

        # collect xforms / transformations
        added_joint_names = []
        xform_paths = sorted(usd_import.get_scene_paths(stage=stage, prim_types=["Xform", "Scope"]))
        log.debug("All xpaths found:\n" + "\n".join(map(str, xform_paths)))

        failure_cnt = 0
        q_index = 0
        q_final = []
        while len(xform_paths) > 0:
            xform_path = xform_paths.pop(0)
            xform_prim = self._stage.GetPrimAtPath(xform_path)

            log.debug(f"Working on {xform_path}")

            # check if connection to parent is defined as joint
            # Note: Joint constraints are undirected, and some USDs give them in different order
            parent_path = xform_prim.GetParent().GetPath()
            log.debug(
                f"Parent: {parent_path}  (node name: {_usd_prim_path_to_node_name(parent_path)})"
            )

            # Child of root path can be added immediately
            if parent_path.pathString != "/":
                for x, y in joints:
                    if y == xform_path:
                        # since they could be connected via a joint it, we will first check whether the other body is part of the scene
                        log.debug(f"Found relevant joint constraint: {x} <-> {y}")
                        parent_path = x

                        if _usd_prim_path_to_node_name(x) in list(s.graph.nodes) + [
                            s.graph.base_frame
                        ]:
                            log.debug(f"Parent (via joint constraint) is in the graph.")
                            break

                graph_node_list = list(s.graph.nodes) + [s.graph.base_frame]
                if _usd_prim_path_to_node_name(parent_path) not in graph_node_list:
                    if (
                        _usd_prim_path_to_node_name(xform_prim.GetParent().GetPath())
                        in graph_node_list
                        and "PhysicsArticulationRootAPI"
                        not in xform_prim.GetParent().GetAppliedSchemas()
                    ):
                        parent_path = xform_prim.GetParent().GetPath()
                        log.debug(
                            f"Will add {xform_path} since {parent_path} is in scenegraph and"
                            " PhysicsArticulationRootAPI is not applied to it."
                        )
                    else:
                        xform_paths.append(xform_path)
                        log.debug(
                            f"Parent {parent_path} (node name:"
                            f" {_usd_prim_path_to_node_name(parent_path)}) not in scene - next"
                            " one..."
                        )
                        log.debug(f"Nodes in graph: {s.graph.nodes}")

                        failure_cnt += 1
                        if failure_cnt > 1000:
                            raise ValueError("Can't parse USD. Tree structure incorrect.")
                        continue

            # reset failure counter
            failure_cnt = 0

            node_name = _usd_prim_path_to_node_name(xform_path)
            parent_node_name = _usd_prim_path_to_node_name(parent_path)

            matrix = usd_import.get_pose(xform_prim)
            extras = None

            if (parent_path, xform_path) in joints:
                # if it's a joint, overwrite transformation
                joint_prim = joints[parent_path, xform_path]
                if joint_prim == "reversed":
                    joint_prim = joints[xform_path, parent_path]
                    transform_B, transform_A = usd_import.get_joint_transform(joint_prim)
                    transform_A = utils.homogeneous_inv(transform_A)
                    transform_B = utils.homogeneous_inv(transform_B)
                    joint_axis_map = {"X": [-1.0, 0, 0], "Y": [0, -1.0, 0], "Z": [0, 0, -1.0]}
                else:
                    transform_A, transform_B = usd_import.get_joint_transform(joint_prim)
                    joint_axis_map = {"X": [1.0, 0, 0], "Y": [0, 1.0, 0], "Z": [0, 0, 1.0]}

                # add articulation metadata
                if joint_prim.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]:
                    joint_name = f"{namespace}/{joint_prim.GetName()}"

                    # make sure joint name is unique
                    if joint_name in added_joint_names:
                        cnt = 1
                        while joint_name + f"_{cnt}" in added_joint_names:
                            cnt += 1
                        joint_name += f"_{cnt}"

                    joint_type_map = {
                        "PhysicsRevoluteJoint": "revolute",
                        "PhysicsPrismaticJoint": "prismatic",
                    }
                    joint_type = joint_type_map[joint_prim.GetTypeName()]
                    meters_per_unit = usd_import.get_meters_per_unit(joint_prim.GetStage())

                    if joint_prim.GetAttribute("physics:maxJointVelocity").Get() is None:
                        limit_velocity = self._default_joint_limit_velocity
                    else:
                        if joint_type == "prismatic":
                            limit_velocity = (
                                joint_prim.GetAttribute("physics:maxJointVelocity").Get()
                                * meters_per_unit
                            )
                        else:
                            limit_velocity = joint_prim.GetAttribute(
                                "physics:maxJointVelocity"
                            ).Get()

                    if joint_prim.GetAttribute("drive:angular:physics:maxForce").Get() is None:
                        limit_effort = self._default_joint_limit_effort
                    else:
                        if joint_type == "prismatic":
                            limit_effort = (
                                joint_prim.GetAttribute("drive:angular:physics:maxForce").Get()
                                * meters_per_unit
                            )
                        else:
                            limit_effort = joint_prim.GetAttribute(
                                "drive:angular:physics:maxForce"
                            ).Get()

                    if joint_prim.GetAttribute("physics:lowerLimit").Get() is None:
                        limit_lower = self._default_joint_limit_lower
                    else:
                        if joint_type == "prismatic":
                            limit_lower = (
                                joint_prim.GetAttribute("physics:lowerLimit").Get()
                                * meters_per_unit
                            )
                        else:
                            limit_lower = np.deg2rad(
                                joint_prim.GetAttribute("physics:lowerLimit").Get()
                            )

                    if joint_prim.GetAttribute("physics:upperLimit").Get() is None:
                        limit_upper = self._default_joint_limit_upper
                    else:
                        if joint_type == "prismatic":
                            limit_upper = (
                                joint_prim.GetAttribute("physics:upperLimit").Get()
                                * meters_per_unit
                            )
                        else:
                            limit_upper = np.deg2rad(
                                joint_prim.GetAttribute("physics:upperLimit").Get()
                            )

                    q = usd_import.get_joint_position(joint_prim=joint_prim)

                    if isinstance(self._configuration, str):
                        if self._configuration == "upper":
                            q = limit_upper
                        elif self._configuration == "lower":
                            q = limit_lower
                        else:
                            raise ValueError(
                                "If argument `configuration` is a string, it can only be 'upper' or"
                                f" 'lower' (currently: {self._configuration})."
                            )
                    elif self._configuration is not None:
                        q = self._configuration[q_index]
                        q_index += 1
                    q_final.append(q)

                    extras = {
                        "joint": {
                            "name": joint_name,
                            "type": joint_type,
                            "q": q,
                            "axis": joint_axis_map[joint_prim.GetAttribute("physics:axis").Get()],
                            "origin": transform_A.tolist(),
                            "limit_velocity": limit_velocity,
                            "limit_effort": limit_effort,
                            "limit_lower": limit_lower,
                            "limit_upper": limit_upper,
                        }
                    }
                    added_joint_names.append(joint_name)

                    # Add an additional node since USD joints have relative transforms w.r.t. body0 and body1
                    joint_node_frame = joint_name + "_frame"
                    log.debug(f"Adding joint {joint_name} with nodes {parent_node_name} --> {joint_node_frame}")
                    utils.add_node_to_scene(
                        scene=s,
                        node_name=joint_node_frame,
                        parent_node_name=parent_node_name,
                        transform=np.eye(4),
                        geom_name=None,
                        geometry=None,
                        **{constants.EDGE_KEY_METADATA: extras},
                    )

                    parent_node_name = joint_node_frame
                    matrix = transform_B
                    extras = None
                else:
                    matrix = transform_A @ transform_B

            log.debug(f"Adding {parent_path} ({parent_node_name})--> {xform_path} ({node_name}) with transform={matrix}")
            utils.add_node_to_scene(
                scene=s,
                node_name=node_name,
                parent_node_name=parent_node_name,
                transform=matrix,
                geom_name=None,
                geometry=None,
                node_data={"prim_path": xform_path.pathString},
                **{constants.EDGE_KEY_METADATA: extras},
            )

        # add meshes
        mesh_paths = usd_import.get_scene_paths(
            stage=stage,
            prim_types=["Mesh"],
            # scene_path_regex=".*collisions" if use_collision_geometry else "^(?!.*(collisions))",
        )
        for mesh_path in mesh_paths:
            if mesh_path.pathString in self._ignore_prim_paths:
                continue

            if not use_collision_geometry and mesh_path.pathString.endswith("collisions"):
                log.info(
                    f"{mesh_path.pathString} will be skipped since it contains `collisions` and"
                    f" use_collision_geometry={use_collision_geometry}."
                )
                continue

            mesh_prim = self._stage.GetPrimAtPath(mesh_path)
            usd_data = usd_import.import_mesh(
                stage=stage,
                scene_path=mesh_prim.GetPath(),
                heterogeneous_mesh_handler=usd_import.heterogeneous_mesh_handler_naive_homogenize,
            )

            if usd_data is None:
                continue

            geometry = trimesh.Trimesh(
                vertices=usd_data.vertices, faces=usd_data.faces, face_colors=usd_data.display_color
            )
            utils.add_filename_to_trimesh_metadata(
                mesh_or_scene=geometry,
                fname=self._fname,
                file_element=mesh_prim.GetPath().pathString,
            )
            # Store the original mesh exents, note that this does *not* include the file-specific
            # scaling factor. During reference-based export we will use this information
            # to infer any additional necessary scaling factor.
            # Note: There might be a use case in which storing the extents *after* scaling
            # the geometry is helpful.
            utils.add_extents_to_trimesh_metadata(mesh_or_scene=geometry)

            scale = usd_import.get_scale(mesh_prim)
            if not np.allclose(scale, [1.0, 1.0, 1.0]):
                geometry.apply_scale(scale)

            node_name = _usd_prim_path_to_node_name(mesh_path)
            parent_node_name = _usd_prim_path_to_node_name(mesh_prim.GetParent().GetPath())
            if _usd_prim_path_to_node_name(mesh_prim.GetParent().GetPath()) not in s.graph.nodes:
                # Skip empty parent(s)
                tmp_mesh_parent = mesh_prim.GetParent()
                while (
                    _usd_prim_path_to_node_name(tmp_mesh_parent.GetPath()) not in s.graph.nodes
                    and tmp_mesh_parent.GetTypeName() == ""
                    and tmp_mesh_parent.GetParent().IsValid()
                ):
                    tmp_mesh_parent = tmp_mesh_parent.GetParent()

                if _usd_prim_path_to_node_name(tmp_mesh_parent.GetPath()) in s.graph.nodes:
                    parent_node_name = _usd_prim_path_to_node_name(tmp_mesh_parent.GetPath())
                else:
                    log.warning(
                        f"Warning: Can't add mesh {node_name} to graph since its parent"
                        f" {parent_node_name} is not part of the graph."
                    )
                    continue

            matrix = usd_import.get_pose(mesh_prim)

            utils.add_node_to_scene(
                scene=s,
                node_name=node_name,
                parent_node_name=parent_node_name,
                transform=matrix,
                geometry=geometry,
                geom_name=f"{namespace}/{mesh_path.pathString.split('/')[-1]}",
                node_data={"prim_path": mesh_path.pathString},
            )

        # add primitives
        primitive_paths = usd_import.get_scene_paths(
            stage=stage,
            prim_types=["Capsule", "Cube", "Cylinder", "Sphere"],
            # scene_path_regex=".*collisions" if use_collision_geometry else "^(?!.*(collisions))",
        )
        for primitive_path in primitive_paths:
            if primitive_path.pathString in self._ignore_prim_paths:
                continue

            primitive_prim = self._stage.GetPrimAtPath(primitive_path)
            usd_data = usd_import.import_primitive(stage=stage, scene_path=primitive_prim.GetPath())

            if usd_data["type"] == "Cube":
                geometry = trimesh.primitives.Box(extents=usd_data["extents"])
            elif usd_data["type"] == "Sphere":
                geometry = trimesh.primitives.Sphere(radius=usd_data["radius"])
            elif usd_data["type"] == "Cylinder":
                geometry = trimesh.primitives.Cylinder(
                    radius=usd_data["radius"], height=usd_data["height"]
                )
            elif usd_data["type"] == "Capsule":
                geometry = trimesh.primitives.Capsule(
                    radius=usd_data["radius"],
                    height=usd_data["height"],
                    transform=tra.translation_matrix([0, 0, -usd_data["height"] * 0.5]),
                )
            else:
                print(f"Ignoring unknown primitive type '{usd_data['type']}'.")
                continue

            matrix = usd_import.get_pose(primitive_prim)

            utils.add_filename_to_trimesh_metadata(
                mesh_or_scene=geometry,
                fname=self._fname,
                file_element=primitive_prim.GetPath().pathString,
            )
            utils.add_extents_to_trimesh_metadata(mesh_or_scene=geometry)

            parent_node_name = _usd_prim_path_to_node_name(primitive_prim.GetParent().GetPath())
            primitive_node_name = _usd_prim_path_to_node_name(primitive_path)
            utils.add_node_to_scene(
                scene=s,
                node_name=primitive_node_name,
                parent_node_name=parent_node_name,
                transform=matrix,
                geometry=geometry,
                geom_name=primitive_node_name,
                node_data={"prim_path": primitive_path.pathString},
            )

        utils.forward_kinematics(
            s,
            joint_names=added_joint_names,
            configuration=q_final,
        )

        return s

class MJCFAsset(Asset):
    def __init__(self, fname, **kwargs):
        """A file-based asset, loaded from a MJCF file.

        Args:
            fname (str): File name.
            **geom_class_visual (str): Class string in geom element that indicates whether this is visual geometry. Defaults to 'visual'.
            **geom_class_collision (str): Class string in geom element that indicates whether this is collision geometry. Defaults to 'collision'.
            **geom_groups_visual (list[int]): List of body/geom/group numbers that will be considered a visual geometry. Defaults to all.
            **geom_groups_collision (list[int]): List of body/geom/group numbers that will be considered a collision geometry. Defaults to all.

        Raises:
            ValueError: Raises exception if file doesn't exist.
        """
        self._fname = fname
        self._model_dir = kwargs.get("model_dir", os.path.dirname(fname))
        self._attributes = kwargs

        self._default_joint_limit_lower = kwargs.get(
            "default_joint_limit_lower", constants.DEFAULT_JOINT_LIMIT_LOWER
        )
        self._default_joint_limit_upper = kwargs.get(
            "default_joint_limit_upper", constants.DEFAULT_JOINT_LIMIT_UPPER
        )
        self._default_joint_limit_velocity = kwargs.get(
            "default_joint_limit_velocity", constants.DEFAULT_JOINT_LIMIT_VELOCITY
        )
        self._default_joint_limit_effort = kwargs.get(
            "default_joint_limit_effort", constants.DEFAULT_JOINT_LIMIT_EFFORT
        )
        self._default_joint_stiffness = kwargs.get(
            "default_joint_stiffness", constants.DEFAULT_JOINT_STIFFNESS
        )
        self._default_joint_damping = kwargs.get(
            "default_joint_damping", constants.DEFAULT_JOINT_DAMPING
        )

        self._configuration = None

        if "configuration" in kwargs:
            if isinstance(kwargs["configuration"], str):
                cfg_str = kwargs["configuration"].lower()
                if cfg_str not in ["lower", "upper"]:
                    raise ValueError(
                        "If argument `configuration` is a string, it can only be 'upper' or"
                        f" 'lower' (currently: {cfg_str})."
                    )
                self._configuration = cfg_str
            else:
                cfg = np.array(kwargs["configuration"])
                self._configuration = cfg
    
    def _mjcf_id(self, namespace, worldbody):
        def create_identifier(elem):
            if elem == worldbody:
                return namespace
            log.debug(f"Create identifier for MJCF element: {elem} (full: {elem.full_identifier})")
            if elem.full_identifier is None:
                return None
            return namespace + '/' + elem.full_identifier.replace("/", "")
        
        return create_identifier

    def _get_transform(self, elem, rad_conversion_fn):
        pos = elem.pos if elem.pos is not None else [0, 0, 0]
        T = tra.translation_matrix(pos)
        if elem.quat is not None:
            return T @ tra.quaternion_matrix(elem.quat)
        if elem.axisangle is not None:
            return T @ tra.rotation_matrix(
                angle=rad_conversion_fn(elem.axisangle[-1]),
                direction=elem.axisangle[:3],
            )
        if elem.euler is not None:
            # This is also defined as eulerseq in the compiler
            # This is the default sequence
            return T @ tra.euler_matrix(*elem.euler, axes='rxyz')
        return T
    
    def _traverse_xml_tree(self, elem, node_name, scene, identifier_fn, rad_conversion_fn, use_collision_geometry):
        from PIL import Image

        if elem.tag == "body":
            log.debug(f"Adding body {identifier_fn(elem)} to {identifier_fn(elem.parent)}")

            body_T = self._get_transform(elem, rad_conversion_fn)

            node_name = utils.add_node_to_scene(
                scene=scene,
                parent_node_name=identifier_fn(elem.parent),
                node_name=identifier_fn(elem),
                transform=body_T,
            )
        elif elem.tag == "geom":
            log.debug(f"Adding geom {identifier_fn(elem)} to {identifier_fn(elem.parent)}")
            
            unknown_geometry = False
            if elem.type == "mesh" or (hasattr(elem, 'mesh') and elem.mesh is not None):
                geometry = trimesh.load(
                    trimesh.util.wrap_as_stream(elem.mesh.file.contents),
                    file_type=elem.mesh.file.extension[1:],
                )
                
                if hasattr(elem.mesh, 'scale') and elem.mesh.scale is not None:
                    geometry.apply_scale(elem.mesh.scale)

                geometry_T = self._get_transform(elem, rad_conversion_fn)
                geometry.apply_transform(geometry_T)

                # Set mesh material
                if elem.material is not None:
                    if elem.material.texture is not None:
                        material = trimesh.visual.material.SimpleMaterial(
                            image=Image.open(trimesh.util.wrap_as_stream(elem.material.texture.file.contents))
                        )
                        texture = trimesh.visual.TextureVisuals(uv=geometry.visual.uv, material=material)
                        geometry.visual = texture
                    else:
                        specular = getattr(elem.material, 'specular', None)
                        diffuse = getattr(elem.material, 'rgba', None)

                        num_faces = len(geometry.faces)
                        face_colors = np.tile(diffuse, (num_faces, 1))
                        # 'emission'
                        # 'reflectance'
                        # 'metallic'
                        # 'roughness'
                        # 'rgba'

                        geometry.visual = trimesh.visual.ColorVisuals(mesh=geometry, face_colors=face_colors)
            elif elem.type == "sphere":
                # The sphere type defines a sphere.
                # Only one size parameter is used, specifying the radius of the sphere.
                geometry = trimesh.primitives.Sphere(
                    radius=elem.size,
                    transform=self._get_transform(elem, rad_conversion_fn)
                )
            elif elem.type == "box":
                # The box type defines a box.
                # Three size parameters are required, corresponding to the half-sizes of the box along the X, Y and Z axes of the geom’s frame.
                geometry = trimesh.primitives.Box(
                    extents=2.0 * elem.size,
                    transform=self._get_transform(elem, rad_conversion_fn)
                )
            elif elem.type == "cylinder":
                # The cylinder type defines a cylinder.
                # It requires two size parameters: the radius and half-height of the cylinder
                geometry = trimesh.primitives.Cylinder(
                    radius=elem.size[0],
                    height=2 * elem.size[1],
                    transform=self._get_transform(elem, rad_conversion_fn)
                )
            elif elem.type == "capsule":
                # The capsule type defines a capsule, which is a cylinder capped with two half-spheres.
                # It is oriented along the Z axis of the geom’s frame.
                # When the geom frame is specified in the usual way, two size parameters are required: the radius of the capsule followed by the half-height of the cylinder part.
                geometry = trimesh.primitives.Capsule(
                    radius=elem.size[0],
                    height=2 * elem.size[1],
                    transform=self._get_transform(elem, rad_conversion_fn),
                )
            else:
                unknown_geometry = True

            elem_dclass_visual_label = self._attributes.get("geom_class_visual", 'visual')
            elem_dclass_collision_label = self._attributes.get("geom_class_collision", 'collision')
            elem_dclass = None if elem.dclass is None else elem.dclass.dclass

            if not unknown_geometry and ((use_collision_geometry and elem_dclass == elem_dclass_visual_label) or (use_collision_geometry == False and elem_dclass == elem_dclass_collision_label)):
                log.debug(f"Ignore geometry {elem.type} since it is of class '{elem_dclass}'.")
            else:
                if not unknown_geometry:
                    if elem.mass is not None:
                        if geometry.is_volume:
                            volume = geometry.volume
                            geometry.density = elem.mass / volume
                        else:
                            log.debug(f"Can't set mass {elem.mass} for {elem.type} because it has no volume.")

                    if elem.density is not None:
                        if geometry.is_volume:
                            geometry.density = elem.density
                        else:
                            log.debug(f"Can't set density {elem.mass} for {elem.type} because it has no volume.")
                    
                    if elem.friction is not None:
                        # 3D array with sliding, torsional, and rolling friction coefficients
                        pass
                    
                    ignore_geometry = False
                    if elem_dclass is not None:
                        if elem_dclass == elem_dclass_visual_label:
                            geometry.metadata["layer"] = 'visual'
                        elif elem_dclass == elem_dclass_collision_label:
                            geometry.metadata["layer"] = 'collision'
                        else:
                            geometry.metadata["layer"] = elem_dclass
                    else:
                        if elem.group is not None:
                            if "geom_groups_collision" in self._attributes:
                                if elem.group in self._attributes['geom_groups_collision']:
                                    geometry.metadata["layer"] = 'collision'
                            if "geom_groups_visual" in self._attributes:
                                if elem.group in self._attributes['geom_groups_visual']:
                                    geometry.metadata["layer"] = 'visual'
                            if "geom_groups_visual" in self._attributes and "geom_groups_collision" in self._attributes and elem.group not in self._attributes['geom_groups_collision'] and elem.group not in self._attributes['geom_groups_visual']:
                                ignore_geometry = True

                    if not ignore_geometry:
                        utils.add_node_to_scene(
                            scene=scene,
                            geometry=geometry,
                            node_name=identifier_fn(elem),
                            geom_name=identifier_fn(elem),
                            parent_node_name=identifier_fn(elem.parent),
                        )
        else:
            log.debug(f"Not used: {elem.tag}, {identifier_fn(elem)}")

        for child in elem.all_children():
            self._traverse_xml_tree(elem=child, node_name=node_name, scene=scene, identifier_fn=identifier_fn, rad_conversion_fn=rad_conversion_fn, use_collision_geometry=use_collision_geometry)
    
    def _get_default_joint_attribute(self, attribute, joint, my_default):
        joint_value = getattr(joint, attribute, None)
        if joint_value is None:
            joint_default = joint.dclass
            while joint_default is not None and joint_value is None and joint_default != joint_default.root:
                joint_value = getattr(joint_default.joint, attribute)
                joint_default = joint_default.parent

        if joint_value is None:
            return my_default

        return joint_value

    def _add_mjcf_joint(self, scene, scene_edge_data, parent_node, child_node, joint, identifier_fn, rad_conversion_fn):
        new_parent_node = scene.graph.transforms.parents[child_node]

        # delete old edge
        old_data = scene_edge_data[(new_parent_node, child_node)]
        T_old_data = old_data.get('matrix', np.eye(4))
        del scene_edge_data[(new_parent_node, child_node)]
        
        # add node
        new_child_node = identifier_fn(joint) + "_frame"
        scene.graph.transforms.node_data[new_child_node].update({})

        if joint.range is None:
            limit_lower = self._default_joint_limit_lower
            limit_upper = self._default_joint_limit_upper
        else:
            limit_lower = rad_conversion_fn(joint.range[0])
            limit_upper = rad_conversion_fn(joint.range[1])

        # Check default values
        joint_range = self._get_default_joint_attribute('range', joint, None)
        if joint_range is None:
            limit_lower, limit_upper = self._default_joint_limit_lower, self._default_joint_limit_upper
        else:
            limit_lower, limit_upper = rad_conversion_fn(joint_range[0]), rad_conversion_fn(joint_range[1])
        joint_pos = self._get_default_joint_attribute('pos', joint, [0., 0., 0.])
        joint_axis = self._get_default_joint_attribute('axis', joint, [0., 0., 1.])
        joint_type = self._get_default_joint_attribute('type', joint, 'revolute')    
        if joint_type == "slide":
            joint_type = "prismatic"

        log.debug(f"{joint_pos}, {joint_type}, {joint_axis}, {limit_lower}, {limit_upper}")
        
        # add edges
        scene_edge_data[(new_parent_node, new_child_node)].update(
            {
                constants.EDGE_KEY_METADATA: {
                    "joint": {
                        "name": identifier_fn(joint),
                        "type": joint_type,
                        "q": 0.0,
                        "origin": T_old_data @ tra.translation_matrix(joint_pos),
                        "limit_lower": limit_lower,
                        "limit_upper": limit_upper,
                        "limit_effort": self._default_joint_limit_effort,
                        "limit_velocity": self._default_joint_limit_velocity,
                        "stiffness": self._default_joint_stiffness,
                        "damping": self._default_joint_damping,
                        "axis": joint_axis,
                    }
                }
            }
        )
        
        scene_edge_data[(new_child_node, child_node)].update({
            'matrix': utils.homogeneous_inv(tra.translation_matrix(joint_pos))
        })

        # update graph
        scene.graph.transforms.parents[new_child_node] = new_parent_node
        scene.graph.transforms.parents[child_node] = new_child_node
        
    def _add_joints(self, root, scene, identifier_fn, rad_conversion_fn):
        joints = root.find_all("joint")

        scene_edge_data = scene.graph.transforms.edge_data

        for joint in joints:
            parent_node = identifier_fn(joint.parent.parent)
            child_node = identifier_fn(joint.parent)
            log.debug(f"Adding joint {identifier_fn(joint)}: {parent_node}->{child_node}")
            
            self._add_mjcf_joint(scene, scene_edge_data, parent_node, child_node, joint, identifier_fn, rad_conversion_fn)
    
    def _load_mjcf(self, fname, model_dir, namespace, use_collision_geometry=None):
        from dm_control import mjcf
        root = mjcf.from_file(fname, model_dir=model_dir)
        assets = root.get_assets()

        rad_conversion_fn = np.deg2rad
        if root.compiler.angle == 'radian':
            rad_conversion_fn = lambda x: x
            
        scene = trimesh.Scene(base_frame=namespace)
        identifier_fn = self._mjcf_id(namespace=namespace, worldbody=root.worldbody)
        
        self._traverse_xml_tree(
            elem=root.worldbody,
            node_name=scene.graph.base_frame,
            scene=scene,
            identifier_fn=identifier_fn,
            rad_conversion_fn=rad_conversion_fn,
            use_collision_geometry=use_collision_geometry
        )

        self._add_joints(root, scene, identifier_fn=identifier_fn, rad_conversion_fn=rad_conversion_fn)

        # Clear cache
        scene.graph.transforms._cache = {}

        return scene
        
    def _as_trimesh_scene(self, namespace="object", use_collision_geometry=True):

        self._model = self._load_mjcf(
            fname=self._fname,
            model_dir=self._model_dir,
            namespace=namespace,
            use_collision_geometry=use_collision_geometry,
        )

        utils.forward_kinematics(self._model)

        return self._model




class TrimeshAsset(MeshAsset):
    def __init__(self, mesh, **kwargs):
        """An asset based on a trimesh.Trimesh

        Args:
            mesh (trimesh.Trimesh): The mesh.
            **kwargs: See Asset() constructor.
        """
        self._origin = np.eye(4)
        self._attributes = kwargs

        if 'name' in kwargs:
            self._name = kwargs['name']

        self._model = mesh.scene()


class BoxAsset(TrimeshAsset):
    def __init__(self, extents, transform=None, **kwargs):
        """A box primitive.

        Args:
            extents (list[float]): 3D extents of the box.
            transform (np.ndarray, optional):  4x4 homogeneous transformation matrix for box center. Defaults to None.
        """
        super().__init__(mesh=trimesh.primitives.Box(extents=extents, transform=transform), **kwargs)

class BoxMeshAsset(TrimeshAsset):
    def __init__(self, extents, transform=None, **kwargs):
        """A triangular mesh in the shape of a box.

        Args:
            extents (list[float]): 3D extents of the box.
            transform (np.ndarray, optional):  4x4 homogeneous transformation matrix for box center. Defaults to None.
        """
        super().__init__(mesh=trimesh.creation.box(extents=extents, transform=transform), **kwargs)

class CylinderAsset(TrimeshAsset):
    def __init__(self, radius, height, transform=None, sections=32, **kwargs):
        """A cylinder primitive.

        Args:
            radius (float): Radius of cylinder.
            height (float): Height of cylinder.
            transform (np.ndarray, optional):  4x4 homogeneous transformation matrix for cylinder center. Defaults to None.
            sections (int, optional): Number of facets in circle. Defaults to 32.
        """
        super().__init__(mesh=trimesh.primitives.Cylinder(radius=radius, height=height, transform=transform, sections=sections), **kwargs)

class SphereAsset(TrimeshAsset):
    def __init__(self, radius, transform=None, subdivisions=3, **kwargs):
        """A sphere primitive.

        Args:
            radius (float): Radius of sphere.
            transform (np.ndarray, optional):  4x4 homogeneous transformation matrix for sphere center. Defaults to None.
            subdivisions (int, optional): Number of subdivisions for icosphere. Defaults to 3.
        """
        super().__init__(mesh=trimesh.primitives.Sphere(radius=radius, transform=transform, subdivisions=subdivisions), **kwargs)

class CapsuleAsset(TrimeshAsset):
    def __init__(self, radius, height, transform=None, sections=32, **kwargs):
        """A capsule primitive.

        Args:
            radius (float): Radius of cylindrical part (and spherical end parts).
            height (float): Height of cylindrical part. Total height of capsule will be height + 2*radius.
            transform (np.ndarray, optional):  4x4 homogeneous transformation matrix for capsule center. Defaults to None.
            sections (int, optional): Number of facets in circle. Defaults to 32.
        """
        super().__init__(mesh=trimesh.primitives.Capsule(radius=radius, height=height, transform=transform, sections=sections), **kwargs)


class TrimeshSceneAsset(Asset):
    def __init__(self, scene, **kwargs):
        """An asset based on a trimesh.Scene

        Args:
            scene (trimesh.Scene): The scene.
            **kwargs: See Asset() constructor.
        """
        self._origin = np.eye(4)
        self._attributes = kwargs

        if 'name' in kwargs:
            self._name = kwargs['name']

        self._model = scene

    def _as_trimesh_scene(self, namespace="object", use_collision_geometry=True):
        result = trimesh.Scene(base_frame=namespace)

        # save geometry in dict
        geometry = {}
        # save transforms as edge tuples
        edges = []

        # Attention: Metadata for geoms is not copied when using copy()
        scene_to_add = self._model

        map_geom = {}
        for name, geom in scene_to_add.geometry.items():
            # store geometry with new name
            map_geom[name] = f"{namespace}/{name}"
            geometry[map_geom[name]] = geom.copy()
            geometry[map_geom[name]].visual = geom.visual.copy()
            geometry[map_geom[name]].metadata = geom.metadata.copy()

        for a, b, attr in scene_to_add.graph.to_edgelist():
            if a == scene_to_add.graph.base_frame:
                a = result.graph.base_frame

                # add origin transform
                attr["matrix"] = self._origin @ attr["matrix"]
            else:
                a = f"{namespace}/{a}"

            # remap node names from local names
            b = f"{namespace}/{b}"

            # keep geometry
            if "geometry" in attr:
                attr["geometry"] = map_geom[attr["geometry"]]

            edges.append((a, b, attr))

        result.graph.from_edgelist(edges)
        result.geometry.update(geometry)

        return result


class LPrismAsset(TrimeshSceneAsset):
    def __init__(self, extents, recess, **kwargs):
        if recess == 0.:
            scene = trimesh.Scene([
                trimesh.primitives.Box(extents=extents)
            ])
        else:
            box_0 = trimesh.primitives.Box(extents=[extents[0], extents[1] - recess, extents[2]], transform=tra.translation_matrix((0, recess/2.0, 0)))
            box_1 = trimesh.primitives.Box(extents=[extents[0] - recess, recess, extents[2]], transform=tra.translation_matrix([+recess/2.0, -(extents[1] - recess)/2.0, 0.0]))
            scene = trimesh.Scene([box_0, box_1])
        
        super().__init__(scene=scene, **kwargs)

class BoxWithHoleAsset(TrimeshSceneAsset):
    def __init__(
        self,
        width,
        depth,
        height,
        hole_width=None,
        hole_depth=None,
        hole_height=None,
        hole_offset=(0, 0),
        use_primitives=False,
        **kwargs,
    ):
        """An asset representing a box with a hole. The hole can be specified along any of the three dimensions.
        To specify the hole, only pass two out of the three hole arguments:
        If hole_width=None, then the hole will be along x.
        If hole_depth=None, then the hole will be along y.
        If hole_height=None, then the hole will be along z.

        Args:
            width (float): Width of the box.
            depth (float): Depth of the box
            height (float): Height of the box.
            hole_width (float, optional): Width of the hole. Defaults to None.
            hole_depth (float, optional): Depth of the hole. Defaults to None.
            hole_height (float, optional): Height of the hole. Defaults to None.
            hole_offset (tuple[float], optional). The offset of the hole from the center of the box. Defaults to (0, 0).
            use_primitives (bool, optional). Whether to use four boxes instead of constructing a single mesh. Defaults to False.
        """
        fn = BoxWithHoleAsset.create_primitives if use_primitives else BoxWithHoleAsset.create_mesh

        super().__init__(
            trimesh.Scene(
                fn(
                    width=width,
                    depth=depth,
                    height=height,
                    hole_width=hole_width,
                    hole_depth=hole_depth,
                    hole_height=hole_height,
                    hole_offset=hole_offset,
                )
            ),
            **kwargs,
        )

    @staticmethod
    def create_primitives(
        width,
        depth,
        height,
        hole_width=None,
        hole_depth=None,
        hole_height=None,
        hole_offset=(0, 0),
    ):
        if sum(h is None for h in [hole_width, hole_depth, hole_height]) != 1:
            raise ValueError(
                "Please specify exactly two parameters of hole_width, hole_depth, and hole_height."
            )

        hole_transform = tra.identity_matrix()
        if hole_width is None:
            width, height = height, width
            hole_width = hole_height
            hole_transform = tra.euler_matrix(0, np.pi / 2.0, 0)
            hole_offset = (-hole_offset[1], hole_offset[0])
        elif hole_depth is None:
            height, depth = depth, height
            hole_depth = hole_height
            hole_transform = tra.euler_matrix(np.pi / 2.0, 0, 0)

        north_depth = (depth - hole_depth) / 2.0 - hole_offset[1]
        south_depth = depth - hole_depth - north_depth
        west_width = (width - hole_width) / 2.0 - hole_offset[0]
        east_width = width - hole_width - west_width

        countertop_north = trimesh.primitives.Box(
            [width - (east_width + west_width), north_depth, height],
            transform=hole_transform
            @ tra.translation_matrix(
                (hole_offset[0], +hole_depth / 2.0 + north_depth / 2.0 + hole_offset[1], 0)
            ),
        )
        countertop_south = trimesh.primitives.Box(
            [width - (east_width + west_width), south_depth, height],
            transform=hole_transform
            @ tra.translation_matrix(
                (hole_offset[0], -hole_depth / 2.0 - south_depth / 2.0 + hole_offset[1], 0)
            ),
        )
        countertop_east = trimesh.primitives.Box(
            [east_width, depth, height],
            transform=hole_transform
            @ tra.translation_matrix((-hole_width / 2.0 - east_width / 2.0 + hole_offset[0], 0, 0)),
        )
        countertop_west = trimesh.primitives.Box(
            [west_width, depth, height],
            transform=hole_transform
            @ tra.translation_matrix((+hole_width / 2.0 + west_width / 2.0 + hole_offset[0], 0, 0)),
        )

        return (countertop_north, countertop_south, countertop_east, countertop_west)

    @staticmethod
    def create_mesh(
        width,
        depth,
        height,
        hole_width=None,
        hole_depth=None,
        hole_height=None,
        hole_offset=(0, 0),
    ):
        if sum(h is None for h in [hole_width, hole_depth, hole_height]) != 1:
            raise ValueError(
                "Please specify exactly two parameters of hole_width, hole_depth, and hole_height."
            )

        hole_transform = tra.identity_matrix()
        if hole_width is None:
            width, height = height, width
            hole_width = hole_height
            hole_transform = tra.euler_matrix(0, np.pi / 2.0, 0)
            hole_offset = (-hole_offset[1], hole_offset[0])
        elif hole_depth is None:
            height, depth = depth, height
            hole_depth = hole_height
            hole_transform = tra.euler_matrix(np.pi / 2.0, 0, 0)

        vertices = [
            [hole_width / 2.0 + hole_offset[0], hole_depth / 2.0 + hole_offset[1], -height / 2.0],
            [width / 2.0, depth / 2.0, -height / 2.0],
            [width / 2.0, depth / 2.0, height / 2.0],
            [hole_width / 2.0 + hole_offset[0], hole_depth / 2.0 + hole_offset[1], height / 2.0],
            [-hole_width / 2.0 + hole_offset[0], hole_depth / 2.0 + hole_offset[1], -height / 2.0],
            [-width / 2.0, depth / 2.0, -height / 2.0],
            [-width / 2.0, depth / 2.0, height / 2.0],
            [-hole_width / 2.0 + hole_offset[0], hole_depth / 2.0 + hole_offset[1], height / 2.0],
            [-hole_width / 2.0 + hole_offset[0], -hole_depth / 2.0 + hole_offset[1], -height / 2.0],
            [-width / 2.0, -depth / 2.0, -height / 2.0],
            [-width / 2.0, -depth / 2.0, height / 2.0],
            [-hole_width / 2.0 + hole_offset[0], -hole_depth / 2.0 + hole_offset[1], height / 2.0],
            [hole_width / 2.0 + hole_offset[0], -hole_depth / 2.0 + hole_offset[1], -height / 2.0],
            [width / 2.0, -depth / 2.0, -height / 2.0],
            [width / 2.0, -depth / 2.0, height / 2.0],
            [hole_width / 2.0 + hole_offset[0], -hole_depth / 2.0 + hole_offset[1], height / 2.0],
        ]

        vertices = tra.transform_points(vertices, hole_transform)

        faces = [
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 6],
            [2, 6, 3],
            [3, 6, 7],
            [3, 7, 0],
            [0, 7, 4],
            [4, 8, 5],
            [5, 8, 9],
            [5, 9, 6],
            [6, 9, 10],
            [6, 10, 7],
            [7, 10, 11],
            [7, 11, 4],
            [4, 11, 8],
            [8, 12, 9],
            [9, 12, 13],
            [9, 13, 10],
            [10, 13, 14],
            [10, 14, 11],
            [11, 14, 15],
            [11, 15, 8],
            [8, 15, 12],
            [12, 0, 13],
            [13, 0, 1],
            [13, 1, 14],
            [14, 1, 2],
            [14, 2, 15],
            [15, 2, 3],
            [15, 3, 12],
            [12, 3, 0],
        ]

        return trimesh.Trimesh(vertices=vertices, faces=faces)


class PlaneAsset(TrimeshAsset):
    def __init__(
        self,
        width=1,
        depth=1,
        normal=(0, 0, 1),
        center=(0, 0, 0),
    ):
        """An asset representing a plane. Can be used to model a ground plane, often used in simulators.

        Note: During USD export, only a square plane will be considered with the length of min(width, depth). Also normals are restricted to
              'X', 'Y', 'Z', and their negative counterparts.

        Args:
            width (float, optional): Width of the plane. Defaults to 1.
            depth (float, optional): Depth of the plane. Defaults to 1.
            normal (tuple[float]): A 3-dimensional unit vector of the normal of the plane. Defaults to (0, 0, 1), ie., pointing in the "z" direction.
            center (tuple[float]): A 3-dimensional vector of the origin of the plane. Defaults to (0, 0, 0).
        """
        if np.linalg.norm(normal) != 1.0:
            raise ValueError(f"Normal vector of plane {normal} needs to have unit length.")

        corners = [
            (-width / 2.0, -depth / 2.0, 0),
            (width / 2.0, -depth / 2.0, 0),
            (-width / 2.0, depth / 2.0, 0),
            (width / 2.0, depth / 2.0, 0),
        ]
        transform = trimesh.geometry.align_vectors((0, 0, 1), normal)
        transform[:3, 3] = center

        vertices = tra.transform_points(corners, transform)
        faces = [[0, 1, 3, 2]]
        plane = trimesh.Trimesh(vertices=vertices, faces=faces)

        super().__init__(plane)


class CQAsset(TrimeshAsset):
    def __init__(
        self,
        cq_object,
        tesselation_tolerance=0.1,
        **kwargs,
    ):
        """An asset based on a CadQuery object.

        Args:
            cq_object (CadQuery): The cadquery object.
            tesselation_tolerance (float): Tolerance for tesselation of the CadQuery object.
        """
        if hasattr(cq_object, "build"):
            cq_object = cq_object.build()

        mesh = utils.cq_to_trimesh(cq_object=cq_object, tolerance=tesselation_tolerance)

        super().__init__(mesh=mesh, **kwargs)


class OpenSCADAsset(MeshAsset):
    def __init__(
        self,
        fname,
        tmp_fname=None,
        openscad_to_stdout=False,
        **kwargs,
    ):
        """A file-based asset, loaded from a OpenSCAD file.

        Args:
            fname (str): File name. Needs to end in .scad.
            tmp_fname (str, optional): File name for temporary file used for conversion via openscad. None means a random name (ending in STL) in the /tmp folder. Defaults to None.
            openscad_to_stdout (bool, optional): Whether to print the output of openscad to stdout. Defaults to False.
        """
        if tmp_fname is None:
            tmp_fname = utils.get_random_filename(suffix=".stl", dir="/tmp")

        cmd = f"openscad {fname} -o {tmp_fname}"

        with open(os.devnull, "w") as devnull:
            stdout = None if openscad_to_stdout else devnull
            subprocess.check_call(cmd.split(), stdout=stdout, stderr=subprocess.STDOUT)
        super().__init__(fname=tmp_fname, **kwargs)
