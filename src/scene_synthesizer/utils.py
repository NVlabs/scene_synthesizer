# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
utils.py
-----------
Standalone functions.
"""

# Standard Library
import colorsys
import itertools
import logging
import os
import re
import uuid

# Third Party
import numpy as np
import trimesh
import trimesh.path
import trimesh.transformations as tra
from shapely.geometry import Point

from .constants import EDGE_KEY_METADATA

try:
    # Third Party
    from scipy.spatial import QhullError
except ImportError:
    # Third Party
    from scipy.spatial.qhull import QhullError

# create a default logger
log = logging.getLogger("scene_synthesizer")


def get_watertight_trimesh_geometry(trimesh_geometry):
    """Returns a watertight version of a trimesh geometry.
    Returns either the geometry itself, its convex hull, or axis-aligned bounding box.

    Args:
        trimesh_geometry (trimesh.Trimesh): Mesh.

    Returns:
        trimesh.Trimesh: Watertight approximation of the input mesh (or input mesh itself if it's already watertight).
    """
    if trimesh_geometry.is_watertight:
        return trimesh_geometry
    else:
        try:
            return trimesh_geometry.convex_hull
        except QhullError:
            return trimesh_geometry.bounding_box


def get_mass_properties(trimesh_geometry, min_thickness=1e-3):
    """Calculate mass properties of trimesh geometry; even if not watertight. Uses the density information of the original geometry.
    If mesh is watertight will return original mass properties.
    If mesh is not watertight the method will try to approximate via convex hull or - if impossible - via axis-aligned bounding box.
    The mass properties of the approximations will then be returned (assuming equal density as the original geometry).
    In case the bounding box has zero length along one of its dimensions, the min_thickness argument will be used.

    Args:
        trimesh_geometry (trimesh.Trimesh): Mesh whose mass properties should be calculated.
        min_thickness (float, optional): Minimum thickness of mesh along all bounding box directions. Defaults to 1e-3.

    Returns:
        float: Mass.
        np.ndarray: Center of mass, 3-vector.
        np.ndarray: Inertia tensor, 3x3 matrix.
        float: Volume.
    """
    center_mass = trimesh_geometry.metadata.get("center_mass", None)

    watertight_geometry = get_watertight_trimesh_geometry(trimesh_geometry)
    
    if watertight_geometry.density != trimesh_geometry.density:
        # Sometimes the density is configured immutable (ValueError)
        # e.g. a bounding box of box with extents 0.0 in one dimension
        watertight_geometry.density = trimesh_geometry.density

    if watertight_geometry.volume == 0.0:
        min_bbox = np.max([watertight_geometry.extents, [min_thickness] * 3], axis=0)
        volume = min_bbox[0] * min_bbox[1] * min_bbox[2]
        mass = volume * watertight_geometry.density
        if center_mass is None:
            center_mass = watertight_geometry.centroid
        mass_12th = mass / 12.0
        inertia_tensor = np.zeros((3, 3))
        inertia_tensor[0, 0] = mass_12th * (min_bbox[1] ** 2 + min_bbox[2] ** 2)
        inertia_tensor[1, 1] = mass_12th * (min_bbox[0] ** 2 + min_bbox[2] ** 2)
        inertia_tensor[2, 2] = mass_12th * (min_bbox[0] ** 2 + min_bbox[1] ** 2)
        return (mass, center_mass, inertia_tensor, volume)

        # alternative: use area * thickness
    elif not watertight_geometry.is_volume:
        # Does this create problems?
        # For primitives it does (since their faces are immutable).
        # But primitives should all be watertight to begin with.
        watertight_geometry.fix_normals()

    return (
        watertight_geometry.mass,
        watertight_geometry.center_mass if center_mass is None else center_mass,
        watertight_geometry.mass_properties["inertia"],
        watertight_geometry.volume,
    )


def center_mass(trimesh_scene, node_names):
    """Return the center of mass of a selection of nodes in a scene.

    Args:
        trimesh_scene (trimesh.Scene): A scene.
        node_names (list[str]): A list of node names.

    Returns:
        np.ndarray: 3D vector describing the center of mass.
    """
    total_mass = 0.0
    total_com = np.zeros(3)
    for n in node_names:
        T, geomn = trimesh_scene.graph[n]
        mass, com, _, _ = get_mass_properties(trimesh_scene.geometry[geomn])
        total_com += mass * (T @ np.append(com, 1.0))[:3]
        total_mass += mass
    result = total_com / total_mass

    return result


def distribute_center_mass(center_mass, geoms):
    """Return center-of-masses for each geom_type_etry based on a single desired joint CoM.

    Args:
        center_mass (list[float]): 3D center of mass vector.
        geoms (list[trimesh.Geometry]): List of trimesh geometries.

    Returns:
        np.ndarray: Nx3 matrix of center-of-masses. N equals len(geoms).
    """
    assert len(center_mass) == 3

    if len(geoms) < 2:
        return np.array(center_mass).reshape(-1, 3)

    mass_matrix = np.array([get_mass_properties(geom)[0] for geom in geoms])
    total_mass = sum(mass_matrix)
    mass_matrix = (mass_matrix / total_mass).reshape(1, -1)

    if total_mass == 0.0:
        mass_matrix = np.array([[1.0 / len(geoms)] * len(geoms)])

    x_coms = np.linalg.lstsq(a=mass_matrix, b=[center_mass[0]], rcond=None)[0]
    y_coms = np.linalg.lstsq(a=mass_matrix, b=[center_mass[1]], rcond=None)[0]
    z_coms = np.linalg.lstsq(a=mass_matrix, b=[center_mass[2]], rcond=None)[0]

    coms = np.vstack([x_coms, y_coms, z_coms]).T

    return coms


def create_yourdfpy_scene(
    yourdfpy_model,
    use_collision_geometry,
    load_geometry=True,
    force_mesh=False,
    force_single_geometry_per_link=False,
):
    """This function builds a trimesh scene from a yourdfpy model.
    TODO: This should be integrated into yourdfpy, since it uses private methods from yourdfpy.

    Args:
        See yourdfpy._create_scene(...)

        One difference: `use_collision_geometry` can be `None` in which case the scene contains both, visual and collision geometry.

    Returns:
        trimesh.Scene: A scene representing
    """
    s = trimesh.scene.Scene(base_frame=yourdfpy_model._base_link)

    for j in yourdfpy_model.robot.joints:
        matrix, _ = yourdfpy_model._forward_kinematics_joint(j)
        s.graph.update(frame_from=j.parent, frame_to=j.child, matrix=matrix)

    for l in yourdfpy_model.robot.links:
        if l.name not in s.graph.nodes and l.name != s.graph.base_frame:
            print(f"{l.name} not connected via joints. Will add link to base frame.")
            s.graph.update(frame_from=s.graph.base_frame, frame_to=l.name)

        if use_collision_geometry is None or use_collision_geometry == False:
            yourdfpy_model._add_geometries_to_scene(
                s,
                geometries=l.visuals,
                link_name=l.name,
                load_geometry=load_geometry,
                force_mesh=force_mesh,
                force_single_geometry=False,
                skip_materials=False,
            )

            # add metadata that this is visual geometry
            new_geoms = []
            for _, v in s.geometry.items():
                if "layer" not in v.metadata:
                    v.metadata["layer"] = "visual"
                    new_geoms.append(v)

            # read inertial properties
            if l.inertial is not None:
                if l.inertial.mass is not None:
                    # distribute mass over all geometries
                    volume = sum(get_mass_properties(geom)[3] for geom in new_geoms)
                    for geom in new_geoms:
                        geom.density = l.inertial.mass / volume

                if l.inertial.origin is not None:
                    # distribute center-of-mass over all geometries
                    coms = distribute_center_mass(
                        center_mass=l.inertial.origin[:3, 3], geoms=new_geoms
                    )

        if use_collision_geometry is None or use_collision_geometry == True:
            yourdfpy_model._add_geometries_to_scene(
                s,
                geometries=l.collisions,
                link_name=l.name,
                load_geometry=load_geometry,
                force_mesh=force_mesh,
                force_single_geometry=force_single_geometry_per_link,
                skip_materials=True,
            )

            # add metadata that this is collision geometry
            new_geoms = []
            for _, v in s.geometry.items():
                if "layer" not in v.metadata:
                    v.metadata["layer"] = "collision"
                    new_geoms.append(v)

            # read inertial properties
            if l.inertial is not None:
                if l.inertial.mass is not None:
                    # distribute mass over all geometries
                    volume = sum(get_mass_properties(geom)[3] for geom in new_geoms)
                    for geom in new_geoms:
                        geom.density = l.inertial.mass / volume

                if l.inertial.origin is not None:
                    # distribute center-of-mass over all geometries
                    coms = distribute_center_mass(
                        center_mass=l.inertial.origin[:3, 3], geoms=new_geoms
                    )

                    # convert coms to individual reference frames
                    for geom, com in zip(new_geoms, coms):
                        geom.metadata["center_mass"] = com

    return s


def is_regex(string):
    """Unofficial check whether a string is a regular expression, based on special characters (except '/' which is used for namespacing in scene_synthesizer).

    Args:
        string (str): String.

    Returns:
        bool: Whether string contains special characters which hint at regular expressions.
    """
    return any(not (c.isalnum() or c in ("/", "-", "_", ".")) for c in string)


def select_sublist(query, all_items):
    """Select a sublist of a given list, using a query.
    The query can be a regular expression, string, or list of strings.

    Args:
        query (str or list): A regular expression, string or list.
        all_items (list[str]): A list of options to select from.

    Raises:
        ValueError: Raises an error if no element from all_items is selected. Usually hints to an unintended query.

    Returns:
        list[str]: A list of items from all_items matching the query.
    """
    if type(query) is list or type(query) is tuple:
        # query is a list
        res = [item for item in query if item in all_items]

        if len(res) != len(query):
            raise ValueError(f"Not all items from {query} can be selected.")
    elif is_regex(query):
        # query is a regular expression
        x = re.compile(query)
        res = list(filter(x.search, all_items))
    elif isinstance(query, str):
        # query is a string
        res = [query] if query in all_items else []
    else:
        raise ValueError(f"Query '{query}' needs to be of type list, tuple, or str.")

    if len(res) > 0:
        return res
    else:
        raise ValueError(f"{query} does not select any element from {all_items}.")


def cycle_list(l, perm):
    """Generator function to cycle infinitely through a fixed order of elements."""
    while True:
        for p in perm:
            yield l[p]


def get_random_filename(suffix, prefix="", dir="/tmp"):
    """Generate a random 8-character file name that won't exist in the provided directory.

    Args:
        suffix (str): File ending / extension
        prefix (str, optional): Start of the file base name. Defaults to empty string "".
        dir (str, optional): Directory of the file. Defaults to "/tmp".

    Returns:
        str: File name / path.
    """
    while True:
        name = str(uuid.uuid4())[:8]
        candidate_fname = os.path.join(dir, prefix + name + suffix)

        if not os.path.exists(candidate_fname):
            return candidate_fname


def object_id_generator(base_name):
    c = itertools.count()
    while True:
        yield f"{base_name}{next(c)}"


def random_id_generator(base_name=""):
    while True:
        yield base_name + str(uuid.uuid1())


def max_support_area_generator(support_data):
    max_support = max(support_data, key=lambda x: x.polygon.area)
    while True:
        yield max_support


def support_area_generator(support_data, seed=None):
    """A generator that randomly returns supports. Their probability of being returned is proportional to their area.

    Args:
        support_data (list[scene.Support]): A list of support surfaces in a scene.
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Yields:
        scene.Support: Support
    """
    rng = np.random.default_rng(seed)

    weights = np.array([s.polygon.area for s in support_data])
    while True:
        index = rng.choice(len(support_data), p=weights / weights.sum())
        yield support_data[index]


def orientation_generator_const(orientation):
    while True:
        yield orientation


def orientation_generator_uniform_around_z(lower=0.0, upper=2.0 * np.pi, seed=None):
    """Generator that yields homogeneous transformations that are randomly rotated around the z axis.
    The rotations are uniform between lower and upper.

    Args:
        lower (float, optional): Lower bound for uniform distribution of angles. Defaults to 0.0.
        upper (float, optional): Upper bound for uniform distribution of angles. Defaults to 2.0*np.pi.
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Yields:
        np.ndarray: A 4x4 homogeneous transform.
    """
    rng = np.random.default_rng(seed)

    while True:
        yield tra.rotation_matrix(angle=rng.uniform(lower, upper), direction=[0, 0, 1])


def orientation_generator_stable_poses(asset, seed=None, **kwargs):
    """Generator that yields random homogeneous transformations that represent stable poses of an asset.
    Internally, calls sample_stable_pose of asset.

    Args:
        asset (scene_synth.Asset): An asset with a `sample_stable_pose(seed)` function.
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Yields:
        np.ndarray: A 4x4 homogeneous transform.
    """
    rng = np.random.default_rng(seed)
    while True:
        yield asset.sample_stable_pose(seed=rng, **kwargs)


class PositionIterator3D(object):
    def __init__(self, seed=None):
        self.container = None
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __call__(self, container):
        self.container = container
        return self

    def update(self, *args, **kwargs):
        pass


class PositionIteratorUniform3D(PositionIterator3D):
    def __next__(self):
        while True:
            pts = sample_volume_mesh(self.container.geometry, count=1, seed=self.rng)
            if pts.size > 0:
                return pts[0]


class PositionIterator2D(object):
    def __init__(self, seed=None):
        self.polygon = None
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __call__(self, support):
        self.polygon = support.polygon
        return self

    def update(self, *args, **kwargs):
        pass


class PositionIteratorUniform(PositionIterator2D):
    def __next__(self):
        while True:
            pts = sample_polygon(self.polygon, count=1, seed=self.rng)
            if pts.size > 0:
                return pts


class PositionIteratorDisk(PositionIterator2D):
    def __init__(self, r, center, seed=None):
        """Sample positions on a 2D plane uniformly within a disk of radius r and center"""
        super().__init__(seed=seed)
        self.r = r
        self.center = center

    def __next__(self):
        while True:
            alpha = np.pi * 2 * self.rng.random()
            radius = self.r * np.sqrt(self.rng.random())

            candidate = np.array([radius * np.cos(alpha), radius * np.sin(alpha)]) + self.center

            p = Point(candidate[0], candidate[1])
            if p.within(self.polygon):
                return candidate


class PositionIteratorList(object):
    def __init__(self, positions):
        super().__init__()
        self.positions = positions
        self.counter = -1

    def __next__(self):
        self.counter = (self.counter + 1) % len(self.positions)
        return self.positions[self.counter]

    def __iter__(self):
        return self

    def __call__(self, support):
        return self

    def update(self, *args, **kwargs):
        pass


class PositionIteratorGaussian(PositionIterator2D):
    def __init__(self, params, seed=None):
        """Gaussian 2D position sampler

        Args:
            params (list, np.ndarray): 2D mean and 2D std of gaussian as a single 4D vector.
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        """
        super().__init__(seed=seed)
        self.params = params

    def __next__(self):
        while True:
            p = Point(
                self.rng.normal(
                    loc=np.array(self.params[:2])
                    + np.array([self.polygon.centroid.x, self.polygon.centroid.y]),
                    scale=self.params[2:],
                )
            )
            if p.within(self.polygon):
                return np.array([p.x, p.y])


class PositionIteratorGrid(PositionIterator2D):
    def __init__(
        self,
        step_x,
        step_y,
        noise_std_x=0.0,
        noise_std_y=0.0,
        direction="x",
        stop_on_new_line=False,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.step = np.array([step_x, step_y])
        self.noise_std_x = noise_std_x
        self.noise_std_y = noise_std_y
        self.direction = direction

        self.new_line = False
        self.stop_on_new_line = stop_on_new_line

        # if self.direction
        #     raise ValueError(f"Unknown direction: {self.direction}")
        self.start_point = None
        self.end_point = None
        self.i = 0
        self.j = 0

    def __next__(self):
        while True:
            if self.stop_on_new_line and self.new_line:
                self.new_line = False
                raise StopIteration

            current_point = self.start_point + np.array([self.i, self.j]) * self.step

            if self.noise_std_x > 0 or self.noise_std_y > 0:
                p = Point(
                    self.rng.normal(
                        loc=current_point,
                        scale=[self.noise_std_x, self.noise_std_y],
                    )
                )
            else:
                p = Point(current_point)

            if np.all(current_point > self.end_point):
                break

            self.new_line = False
            if self.direction == "x":
                if current_point[0] > self.end_point[0]:
                    self.i = 0
                    self.j += 1
                    self.new_line = True
                else:
                    self.i += 1
            elif self.direction == "y":
                if current_point[1] > self.end_point[1]:
                    self.j = 0
                    self.i += 1
                    self.new_line = True
                else:
                    self.j += 1

            if p.within(self.polygon):
                return np.array([p.x, p.y])

        raise StopIteration

    def __call__(self, support):
        if support.polygon != self.polygon:
            self.polygon = support.polygon

            minx, miny, maxx, maxy = self.polygon.bounds
            self.start_point = np.array([minx, miny])
            self.end_point = np.array([maxx, maxy])
            self.i = 0
            self.j = 0

            self.new_line = False

        return self


class PositionIteratorPoissonDisk(PositionIterator2D):
    def __init__(self, k=30, r=0.01, seed=None):
        super().__init__(seed=seed)
        self.k = k
        self.r = r

        self.points = []
        self.active_points = []

    def __next__(self):
        if len(self.points) == 0:
            while True:
                pts = sample_polygon(self.polygon, count=1, seed=self.rng)
                if pts.size > 0:
                    return pts[0]

        while len(self.active_points) > 0:
            active_point = self.active_points[0]

            for _ in range(self.k):
                # sample in annulus
                alpha = np.pi * 2 * self.rng.random()
                radius = np.sqrt(self.rng.random() * self.r + self.r)

                candidate = (
                    np.array([radius * np.cos(alpha), radius * np.sin(alpha)]) + active_point
                )

                # check if close to any other point
                # other_indices = np.delete(np.arange(len(points)), active_point_index)
                distances = np.linalg.norm(self.points - candidate, axis=1)
                if all(distances >= self.r):
                    p = Point(candidate[0], candidate[1])
                    if p.within(self.polygon):
                        return candidate

            self.active_points.pop(0)

        raise StopIteration

    def update(self, point):
        self.points.append(point)
        self.active_points.append(point)


class PositionIteratorFarthestPoint(PositionIterator2D):
    def __init__(self, sample_count=100):
        super().__init__()
        self.sample_count = sample_count

        self.all_possible_points = []
        self.points = []

    def distance_L2(self, p1, p2):
        return np.linalg.norm(p1 - p2, axis=1)

    def __next__(self):
        while True:
            if len(self.all_possible_points) == 0:
                while True:
                    pts = sample_polygon(self.polygon, count=self.sample_count, seed=self.rng)
                    if pts.size > 0:
                        self.distances = np.ones((len(pts),), dtype=np.float32) * 1e7

                        self.all_possible_points = np.array(pts)
                        break

            index = np.argmax(self.distances)

            next_point = self.all_possible_points[index]

            self.all_possible_points = np.delete(self.all_possible_points, index, axis=0)
            self.distances = np.delete(self.distances, index)

            return next_point

    def update(self, point):
        self.points.append(point)

        broadcasted_next_point = np.tile(
            np.expand_dims(point, 0), (len(self.all_possible_points), 1)
        )
        new_distances = self.distance_L2(broadcasted_next_point, self.all_possible_points)
        self.distances = np.minimum(self.distances, new_distances)


def collision_manager_transform(collision_manager, transform, premultiply=False):
    """Helper function to transform all objects of a collision manager.

    Args:
        collision_manager (trimesh.collision.CollisionManager): A trimesh collision manager.
        transform (np.ndarray): A 4x4 homogeneous transform matrix.
        premultiply (bool, optional): Whether to post- or pre-multiply the transform. Defaults to False, i.e., post-multiplication.
    """
    for name in collision_manager._objs:
        o = collision_manager._objs[name]["obj"]

        t = np.eye(4)
        t[:3, :3] = o.getRotation()
        t[:3, 3] = o.getTranslation()

        if premultiply:
            t = transform @ t
        else:
            t = t @ transform

        o.setRotation(t[:3, :3])
        o.setTranslation(t[:3, 3])
        collision_manager._manager.update(o)


def collision_manager_get_transforms(collision_manager):
    """Helper function to get a dictionary of all object transformations in a collision manager.

    Args:
        collision_manager (trimesh.collision.CollisionManager): A trimesh collision manager.

    Returns:
        transforms (dict[str, np.ndarray]): A dictionary of all transforms in the collision manager indexed by the object's name.
    """
    result = {}
    for name in collision_manager._objs:
        o = collision_manager._objs[name]["obj"]

        t = np.eye(4)
        t[:3, :3] = o.getRotation()
        t[:3, 3] = o.getTranslation()

        result[name] = t
    return result


def collision_manager_set_transforms(collision_manager, transforms):
    """Helper function to set all transformations in a collision manager.

    Args:
        collision_manager (trimesh.collision.CollisionManager): A trimesh collision manager.
        transforms (dict[str, np.ndarray]): A dictionary of all transforms in the collision manager indexed by the object's name.
    """
    for name, transform in transforms.items():
        o = collision_manager._objs[name]["obj"]
        o.setRotation(transform[:3, :3])
        o.setTranslation(transform[:3, 3])
        collision_manager._manager.update(o)


def construct_obj_collision_manager(scene, transform):
    """Return a trimesh collision manager for a trimesh scene, with a pre-multiplied transform for all objects.

    Args:
        scene (trimesh.Scene): A scene.
        transform (np.ndarray): A 4x4 homogeneous matrix.

    Returns:
        trimesh.CollisionManager: A collision manager of the scene.
        dict (str: str): A dictionary that maps scene nodes to names for objects in the collision manager.
    """
    # Third Party
    from trimesh.collision import CollisionManager

    manager = CollisionManager()
    objects = {}
    for node in scene.graph.nodes_geometry:
        T, geometry = scene.graph[node]
        objects[node] = manager.add_object(
            name=node, mesh=scene.geometry[geometry], transform=transform @ T
        )
    return manager, objects


def get_reference_frame(bounds, center_mass, centroid, x, y, z):
    # Seems to be called from 4 places
    translation = [0.0, 0.0, 0.0]

    for i, alignment in enumerate([x.lower(), y.lower(), z.lower()]):
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


def add_node_to_scene(scene, **kwargs):
    """Add a new node (and edge) to the scene graph. Uses the same arguments as trimesh.Scene.add_geometry(...).
    But will just add a transformation to the scene graph if geometry is None or missing, by delegating to trimesh.Scene.graph.update(...).

    Args:
        scene (trimesh.scene.Scene): The scene to add a new node to.
        **node_name (str): Name of the added node.
        **parent_node_name (str): Name of the parent node in the graph.
        **geometry (trimesh.Trimesh, ...): Added geometry.
        **geom_name (str): Name of the added geometry.
        **transform (np.ndarray): 4x4 homomgeneous transformation matrix that applies to the added node.
        **constants.EDGE_KEY_METADATA (dict): Optional metadata for the node.
        **node_data (dict): A node data dictionary.
    """
    node_data = kwargs.pop("node_data", None)
    if "geometry" not in kwargs or kwargs["geometry"] is None:
        scene.graph.update(
            frame_from=kwargs["parent_node_name"],
            frame_to=kwargs["node_name"],
            matrix=kwargs["transform"],
            **{EDGE_KEY_METADATA: kwargs.get(EDGE_KEY_METADATA, None)},
        )
        node_name = kwargs["node_name"]
    else:
        node_name = scene.add_geometry(**kwargs)

    if node_name is not None and node_data is not None:
        scene.graph.transforms.node_data[node_name].update(node_data)


def add_filename_to_trimesh_metadata(mesh_or_scene, fname, file_element=None):
    """Add file_path and file_name properties to trimesh scene or individual mesh.

    Args:
        mesh_or_scene (trimesh.Trimesh or trimesh.scene.Scene): The geometry or a scene, i.e., collection of geometries.
        fname (str): The original source filename of the geometry that will be added to the metadata.
        file_element (str or int, optional): The specific part of the source file, e.g. an integer for an OBJ with multiple parts. Or a prim path for a USD. Defaults to None.
    """
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        if "file_path" not in mesh_or_scene.metadata:
            mesh_or_scene.metadata["file_path"] = os.path.abspath(fname)
            mesh_or_scene.metadata["file_name"] = os.path.basename(fname)

            if file_element is not None:
                mesh_or_scene.metadata["file_element"] = file_element
    else:
        file_path = (
            mesh_or_scene.metadata["file_path"]
            if "file_path" in mesh_or_scene.metadata
            else os.path.abspath(fname)
        )
        file_name = (
            mesh_or_scene.metadata["file_name"]
            if "file_name" in mesh_or_scene.metadata
            else os.path.basename(fname)
        )

        for i, (_, geom) in enumerate(mesh_or_scene.geometry.items()):
            if "file_path" not in geom.metadata:
                geom.metadata["file_path"] = file_path
                geom.metadata["file_name"] = file_name
                geom.metadata["file_element"] = i


def add_extents_to_trimesh_metadata(mesh_or_scene):
    """Add extents to trimesh scene or individual mesh.

    Args:
        mesh_or_scene (trimesh.Trimesh or trimesh.scene.Scene): The geometry or a scene, i.e., collection of geometries.
    """
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh_or_scene.metadata["extents"] = mesh_or_scene.extents.tolist()
    else:
        for _, geom in mesh_or_scene.geometry.items():
            geom.metadata["extents"] = geom.extents.tolist()

def hash_trimesh_geometry_metadata(geom):
    """Return a hash value for the metadata of trimesh.Geometry
    
    Args:
        geom (trimesh.Geometry): The geometry whose metadata should be hashed.

    Returns:
        hash (int): A hash value to compare different metadata.
    """
    m = geom.metadata
    # Once these values are not saved as geometry metadata (but as geometry node data)
    # this can be changed again to only include vertex/triangle information
    return hash((m.get('file_path', ''), m.get('file_name', ''), m.get('file_element', ''), m.get('layer', ''), str(m.get('center_mass', '')), geom.density, geom.mass, str(geom.center_mass.tolist())))


def sample_random_direction_R3(number_of_directions, seed=None):
    """Uniformly distributed directions on S2. Sampled from a multivariate Gaussian, followed by normalization.

    Args:
        number_of_directions (int): Number of directions to sample.
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Returns:
        np.array: number_of_directionsx3 array of directions
    """
    rng = np.random.default_rng(seed)

    # sample multivariate Gaussian and normalize
    dir = rng.normal(0, 1, (number_of_directions, 3))
    dir = dir / np.linalg.norm(dir, axis=1)[:, np.newaxis]

    return dir


def snake_case(s):
    """Return string s in snake_case.

    Args:
        s (str): String to convert

    Returns:
        str: s converted to snake_case.
    """
    return re.sub("(?!^)([A-Z]+)", r"_\1", s).lower()


def get_transform(trimesh_scene, frame_to, frame_from=None):
    # Addresses:
    # https://github.com/mikedh/trimesh/issues/1321
    # https://github.com/mikedh/trimesh/issues/1726

    trans_b = trimesh_scene.graph.get(frame_to=frame_to)[0]
    if frame_from is None:
        return trans_b

    trans_a = tra.inverse_matrix(trimesh_scene.graph.get(frame_to=frame_from)[0])

    return trans_a @ trans_b


def right_and_aligned_back_bottom():
    """Helper function that returns arguments for add_object(..).

    Returns:
        dict: A dictionary of arguments.
    """
    return {
        "connect_parent_anchor": ("top", "top", "bottom"),
        "connect_obj_anchor": ("bottom", "top", "bottom"),
    }


def left_and_aligned_back_bottom():
    """Helper function that returns arguments for add_object(..).

    Returns:
        dict: A dictionary of arguments.
    """
    return {
        "connect_parent_anchor": ("bottom", "top", "bottom"),
        "connect_obj_anchor": ("top", "top", "bottom"),
    }


def above_and_aligned_right_back():
    """Helper function that returns arguments for add_object(..).

    Returns:
        dict: A dictionary of arguments.
    """
    return {
        "connect_parent_anchor": ("top", "top", "top"),
        "connect_obj_anchor": ("top", "top", "bottom"),
    }


def late_bind_exception(exc):
    """Helper to bind exceptions to function calls, e.g. when an import fails but import is only used for specific functionality.

    Args:
        exc (BaseException): An exception that is raised when the returned function is called.

    Returns:
        fn: A function which when called raises the exception exc.
    """

    def failed(*args, **kwargs):
        raise exc

    return failed


def scaled_trimesh_scene(scene, scale):
    """
    Return a copy of the trimesh scene, with meshes and scene
    transforms scaled to the requested factor.

    Parameters
    -----------
    scale : float or (3,) float
        Factor to scale meshes and transforms

    Returns
    -----------
    scaled : trimesh.Scene
        A copy of the current scene but scaled
    """
    # convert 2D geometries to 3D for 3D scaling factors
    scale_is_3D = isinstance(scale, (list, tuple, np.ndarray)) and len(scale) == 3

    if not scale_is_3D:
        scale = float(scale)
        scale = [scale, scale, scale]

    if np.allclose(scale, [1, 1, 1]):
        return scene

    # result is a copy
    result = scene.copy()

    # Copy all geometries that appear multiple times in the scene,
    # such that no two nodes share the same geometry.
    # This is required since the non-uniform scaling will most likely
    # affect the same geometry in different poses differently.
    # Note, that this is not needed in the case of uniform scaling.
    for geom_name in result.graph.geometry_nodes:
        nodes_with_geom = result.graph.geometry_nodes[geom_name]
        if len(nodes_with_geom) > 1:
            geom = result.geometry[geom_name]
            for n in nodes_with_geom:
                p = result.graph.transforms.parents[n]
                result.add_geometry(
                    geometry=geom.copy(),
                    geom_name=geom_name,
                    node_name=n,
                    parent_node_name=p,
                    transform=result.graph.transforms.edge_data[(p, n)].get("matrix", None),
                    **{EDGE_KEY_METADATA: result.graph.transforms.edge_data[(p, n)].get(EDGE_KEY_METADATA, None)},
                )
            result.delete_geometry(geom_name)

    # Convert all 2D paths to 3D paths
    for geom_name in result.geometry:
        if result.geometry[geom_name].vertices.shape[1] == 2:
            result.geometry[geom_name] = result.geometry[geom_name].to_3D()

    # Scale all geometries by un-doing their local rotations first
    for key in result.graph.nodes_geometry:
        T, geom_name = result.graph.get(key)
        T = np.copy(T)
        T[:3, 3] = 0.0

        # Get geometry transform w.r.t. base frame
        geometry = result.geometry[geom_name]

        # Scale primitives differently, since trimesh suffers from this
        # See Issue: https://github.com/mikedh/trimesh/issues/1790
        if hasattr(geometry, "primitive"):
            if not np.allclose(scale, scale[0]):
                raise NotImplementedError(
                    f"Can't scale primitives non-uniformly ({scale}) since result usually won't be"
                    " a primtitive."
                )

            if isinstance(geometry, trimesh.primitives.Sphere):
                geometry.primitive.radius *= scale[0]
            elif isinstance(geometry, trimesh.primitives.Box):
                geometry.primitive.extents *= scale[0]
            elif isinstance(geometry, trimesh.primitives.Cylinder) or isinstance(
                geometry, trimesh.primitives.Capsule
            ):
                geometry.primitive.height *= scale[0]
                geometry.primitive.radius *= scale[0]
            else:
                raise NotImplementedError("Can't scale unknown primitive.")
        else:
            T_inv = homogeneous_inv(T)
            geometry.apply_transform(T).apply_scale(scale).apply_transform(T_inv)
            geometry.metadata["scale"] = list(np.abs(T_inv[:3, :3] @ scale))

    # Record all joint origins
    joint_origins_world_scaled = {}
    edge_data = result.graph.transforms.edge_data
    for uv in edge_data:
        props = edge_data[uv]
        if EDGE_KEY_METADATA in props and props[EDGE_KEY_METADATA] is not None and "joint" in props[EDGE_KEY_METADATA]:
            if "origin" in props[EDGE_KEY_METADATA]["joint"]:
                origin = np.array(props[EDGE_KEY_METADATA]["joint"]["origin"])
                T = np.copy(result.graph[uv[0]][0])
                origin_w = T @ origin
                origin_w[:3, 3] *= scale
                joint_origins_world_scaled[uv] = origin_w

    # Scale all transformations in the scene graph
    scaled_Ts = {}
    for n in result.graph.nodes:
        T = np.copy(result.graph[n][0])
        T[:3, 3] *= scale
        scaled_Ts[n] = T
    new_Ts = []
    for n in scaled_Ts:
        if n == result.graph.base_frame:
            continue
        parent = result.graph.transforms.parents[n]
        new_Ts.append((parent, n, {"matrix": tra.inverse_matrix(scaled_Ts[parent]) @ scaled_Ts[n]}))

    for a, b, attr in new_Ts:
        assert (a, b) in result.graph.transforms.edge_data
        result.graph.transforms.edge_data[(a, b)]["matrix"] = np.array(
            attr["matrix"], dtype=np.float64
        )

    # Scale all joint origins
    for uv in joint_origins_world_scaled:
        joint_data = edge_data[uv][EDGE_KEY_METADATA]["joint"]

        origin = tra.inverse_matrix(scaled_Ts[uv[0]]) @ joint_origins_world_scaled[uv]
        joint_data["origin"] = origin

        # Scale all prismatic joint limits
        if joint_data["type"] == "prismatic":
            for limit in ("limit_lower", "limit_upper"):
                # project scale vector onto joint axis
                projected_scale = np.abs(
                    (joint_origins_world_scaled[uv][:3, :3] @ joint_data["axis"]).dot(scale)
                )
                if limit in joint_data:
                    joint_data[limit] *= projected_scale

    # Clear cache
    result._cache.clear()
    result.graph._cache.clear()
    result.graph._modified = str(uuid.uuid4())
    result.graph.transforms._cache.clear()
    result.graph.transforms._modified = str(uuid.uuid4())
    return result


def homogeneous_inv(matrix):
    """Inverse homogeneous matrix.
    Slightly faster than np.linalg.inv or trimesh.transformations.inverse_matrix.

    Args:
        matrix (np.ndarray): 4x4 homogeneous matrix.

    Returns:
        np.ndarray: Inverse matrix of the input matrix.
    """
    res = np.eye(4)
    res[:3, :3] = matrix[:3, :3].T
    res[:3, 3] = np.dot(-res[:3, :3], matrix[:3, 3])
    return res


def forward_kinematics(scene, joint_names=None, configuration=None):
    """Helper function to update the scene graph by running forward kinematics on all edges specified through joint_names.

    Args:
        scene (trimesh.Scene): Scene.
        joint_names (list[str], optional): Names of joints to update. If None, use all joints. Defaults to None.
        configuration (list[float], optional): Configurations. Needs to be same size as joint_names or None. If None will use configuration stored in graph. Defaults to None.
    """
    if configuration is not None and joint_names is not None:
        assert len(joint_names) == len(configuration)

    # Create joint map
    scene_edge_data = scene.graph.transforms.edge_data
    joint_map = {}
    for k in scene.graph.transforms.edge_data:
        edge_data = scene_edge_data[k]
        if (
            EDGE_KEY_METADATA in edge_data
            and edge_data[EDGE_KEY_METADATA] is not None
            and "joint" in edge_data[EDGE_KEY_METADATA]
        ):
            joint_data = edge_data[EDGE_KEY_METADATA]["joint"]
            joint_map[joint_data["name"]] = k

    if joint_names is None:
        joint_names = joint_map.keys()

    for i, joint_name in enumerate(joint_names):
        matrix = None
        edge_data = scene_edge_data[joint_map[joint_name]]
        joint_data = edge_data[EDGE_KEY_METADATA]["joint"]

        if joint_data["type"] == "fixed" or joint_data["type"] == "floating":
            matrix = None
            if 'origin' in joint_data:
                matrix = np.array(joint_data.get("origin"))
        elif joint_data["type"] == "revolute" or joint_data["type"] == "continuous":
            if configuration is None:
                matrix = (
                    joint_data["origin"]
                    @ tra.rotation_matrix(joint_data["q"], joint_data["axis"])
                )
            else:
                matrix = (
                    joint_data["origin"]
                    @ tra.rotation_matrix(configuration[i], joint_data["axis"])
                )
                joint_data["q"] = configuration[i]
        elif joint_data["type"] == "prismatic":
            if configuration is None:
                matrix = (
                    joint_data["origin"]
                    @ tra.translation_matrix(joint_data["q"] * np.asarray(joint_data["axis"]))
                )
            else:
                matrix = (
                    joint_data["origin"]
                    @ tra.translation_matrix(configuration[i] * np.asarray(joint_data["axis"]))
                )
                joint_data["q"] = configuration[i]

        if matrix is not None:
            edge_data["matrix"] = matrix

    invalidate_scenegraph_cache(scene)


def invalidate_scenegraph_cache(scene):
    """Invalidate cache of scene's graph.

    Args:
        scene (trimesh.Scene): The scene.
    """
    scene.graph._modified = str(uuid.uuid4())
    scene.graph.transforms._modified = str(uuid.uuid4())
    scene._cache.clear()
    scene.graph._cache.clear()
    scene.graph.transforms._cache.clear()


def cq_to_trimesh(cq_object, tolerance=0.1):
    """Convert cadquery object into Trimesh.

    Args:
        cq_object (cadquery): CadQuery object.
        tolerance (float, optional): Tesselation tolerance. Defaults to 0.1.

    Returns:
        trimesh.Trimesh: A triangular mesh representing the tesselated CadQuery object.
    """
    result = cq_object.tessellate(tolerance=tolerance)
    vertices = np.array([x.toTuple() for x in result[0]])
    return trimesh.Trimesh(vertices=vertices, faces=result[1])

def random_color(dtype=np.uint8, seed=None):
    """Return a random RGB color using datatype specified.
    Same version as trimesh.visual.random_color but with seed.

    Args:
        dtype (np.dtype): numpy dtype of result
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator. 

    Returns:
        np.ndarray(4,): random color of type dtype
    """
    rng = np.random.default_rng(seed)
    
    hue = rng.random() + 0.61803
    hue %= 1.0
    color = np.array(colorsys.hsv_to_rgb(hue, 0.99, 0.99))
    if np.dtype(dtype).kind in "iu":
        max_value = (2 ** (np.dtype(dtype).itemsize * 8)) - 1
        color *= max_value
    color = np.append(color, max_value).astype(dtype)
    return color

def adjust_color(color=None, brightness=1.0, transparency=1.0, seed=None):
    """Adjust the brightness and transparency of a color.

    Args:
        color (list[float], optional): The RGB color to be used. If None, a random color will be chosen. Defaults to None.
        brightness (float, optional): Brightness of colors. Defaults to 1.0.
        transparency (float, optional): Transparency of colors. Defaults to 1.0.
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator. 

    Returns:
        list[float]: The adjusted color.
    """
    if color is None:
        full_color = random_color(seed=seed)
    elif len(color) == 3:
        full_color = color + [255]
    else:
        full_color = color
    full_color = np.array(full_color, dtype=np.float32)
    full_color[:3] *= brightness
    full_color[3] *= transparency
    return full_color.astype(np.uint8)

def append_scenes(iterable, common=["world"], base_frame="world", share_geometry=False):
    """
    Concatenate multiple scene objects into one scene.
    Parameters
    -------------
    iterable : (n,) Trimesh or Scene
       Geometries that should be appended
    common : (n,) str
       Nodes that shouldn't be remapped
    base_frame : str
       Base frame of the resulting scene
    share_geometry : bool
       Whether to share geometry, based on hashing. This improves performance for large scenes but might lead to unintended side effects since properties can be overwritten. Defaults to False.
    Returns
    ------------
    result : trimesh.Scene
       Scene containing all geometry
    """
    if isinstance(iterable, trimesh.Scene):
        return iterable

    # save geometry in dict
    geometry = {}
    # save transforms as edge tuples
    edges = []

    # nodes which shouldn't be remapped
    common = set(common)
    # nodes which are consumed and need to be remapped
    consumed = set()

    def node_remap(node):
        """
        Remap node to new name if necessary
        Parameters
        -------------
        node : hashable
           Node name in original scene
        Returns
        -------------
        name : hashable
           Node name in concatenated scene
        """

        # if we've already remapped a node use it
        if node in map_node:
            return map_node[node]

        # if a node is consumed and isn't one of the nodes
        # we're going to hold common between scenes remap it
        if node not in common and node in consumed:
            # generate a name not in consumed
            name = node + trimesh.util.unique_id()
            map_node[node] = name
            node = name

        # keep track of which nodes have been used
        # in the current scene
        current.add(node)
        return node

    # loop through every geometry
    for s in iterable:
        # allow Trimesh/Path2D geometry to be passed
        if hasattr(s, "scene"):
            s = s.scene()
        # if we don't have a scene raise an exception
        if not isinstance(s, trimesh.Scene):
            raise ValueError("{} is not a scene!".format(type(s).__name__))

        # remap geometries if they have been consumed
        map_geom = {}
        for k, v in s.geometry.items():
            # check if geometry hash already exists
            geometry_found = False

            for k_geom, v_geom in geometry.items():
                if share_geometry and hash(v) == hash(v_geom) and hash_trimesh_geometry_metadata(v) == hash_trimesh_geometry_metadata(v_geom):
                    map_geom[k] = k_geom
                    geometry_found = True
                    break

            if not geometry_found:
                # if a geometry already exists add a UUID to the name
                name = trimesh.util.unique_name(start=k, contains=geometry.keys())
                # store name mapping
                map_geom[k] = name
                # store geometry with new name
                geometry[name] = v

        # remap nodes and edges so duplicates won't
        # stomp all over each other
        map_node = {}
        # the nodes used in this scene
        current = set()
        for a, b, attr in s.graph.to_edgelist():
            # remap node names from local names
            a, b = node_remap(a), node_remap(b)
            # remap geometry keys
            # if key is not in map_geom it means one of the scenes
            # referred to geometry that doesn't exist
            # rather than crash here we ignore it as the user
            # possibly intended to add in geometries back later
            if "geometry" in attr and attr["geometry"] in map_geom:
                attr["geometry"] = map_geom[attr["geometry"]]
            # save the new edge
            edges.append((a, b, attr))
        # mark nodes from current scene as consumed
        consumed.update(current)

    # add all data to a new scene
    result = trimesh.Scene(base_frame=base_frame)
    result.graph.from_edgelist(edges)
    result.geometry.update(geometry)

    return result

def sample_volume_mesh(mesh, count, seed=None):
    """Use rejection sampling to produce points randomly distributed in the volume of a mesh.
    This is the same as trimesh.sample.volume_mesh but using a specified RNG.

    Args:
        mesh (trimesh.Trimesh): Geometry to sample
        count (int): Number of points to return
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Returns:
        (n, 3) float: Points in the volume of the mesh where n <= count
    """
    rng = np.random.default_rng(seed)

    points = (rng.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples

def sample_polygon(polygon, count, factor=1.5, max_iter=10, seed=None):
    """Use rejection sampling to generate random points inside a polygon.
    This is the same as sample_polygon but using a specified RNG.

    Args:
        polygon (shapely.geometry.Polygon): Polygon that will contain points
        count (int): Number of points to return
        factor (float): How many points to test per loop
        max_iter (int): Maximum number of intersection checks is: > count * factor * max_iter
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.

    Returns:
        (n, 2) float: Random points inside polygon where n <= count
    """
    rng = np.random.default_rng(seed)

    # do batch point-in-polygon queries
    from shapely import vectorized

    # get size of bounding box
    bounds = np.reshape(polygon.bounds, (2, 2))
    extents = np.ptp(bounds, axis=0)

    # how many points to check per loop iteration
    per_loop = int(count * factor)

    # start with some rejection sampling
    points = bounds[0] + extents * np.random.random((per_loop, 2))
    # do the point in polygon test and append resulting hits
    mask = vectorized.contains(polygon, *points.T)
    hit = [points[mask]]
    hit_count = len(hit[0])
    # if our first non-looping check got enough samples exit
    if hit_count >= count:
        return hit[0][:count]

    # if we have to do iterations loop here slowly
    for _ in range(max_iter):
        # generate points inside polygons AABB
        points = rng.random((per_loop, 2))
        points = (points * extents) + bounds[0]
        # do the point in polygon test and append resulting hits
        mask = vectorized.contains(polygon, *points.T)
        hit.append(points[mask])
        # keep track of how many points we've collected
        hit_count += len(hit[-1])
        # if we have enough points exit the loop
        if hit_count > count:
            break

    # stack the hits into an (n,2) array and truncate
    hit = np.vstack(hit)[:count]

    return hit

def create_torus(major_radius, minor_radius, num_major_segments=30, num_minor_segments=20):
    """Create torus mesh.

    Args:
        major_radius (float, optional): Radius from the center of the torus to the center of the tube. Defaults to 1.0.
        minor_radius (float, optional): Radius of the tube.
        num_major_segments (int, optional): Number of segments around the major radius. Defaults to 30.
        num_minor_segments (int, optional): Number of segments around the minor radius. Defaults to 20.

    Returns:
        trimesh.Trimesh: Torus mesh.
    """

    vertices = []
    faces = []

    for i in range(num_major_segments):
        theta = 2 * np.pi * i / num_major_segments
        for j in range(num_minor_segments):
            phi = 2 * np.pi * j / num_minor_segments

            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
            y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
            z = minor_radius * np.sin(phi)

            vertices.append([x, y, z])

            # Create faces
            a = i * num_minor_segments + j
            b = ((i + 1) % num_major_segments) * num_minor_segments + j
            c = ((i + 1) % num_major_segments) * num_minor_segments + (j + 1) % num_minor_segments
            d = i * num_minor_segments + (j + 1) % num_minor_segments

            faces.append([a, b, c])
            faces.append([a, c, d])

    # Create a trimesh object from vertices and faces
    torus_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return torus_mesh

def normalize_and_bake_scale(trimesh_scene):
    """Takes a trimesh scene and bakes all scale information contained in the transform tree
    into the vertices of all meshes.
    The resulting tree transforms will all have scale 1.

    Args:
        trimesh_scene (trimesh.Scene): A trimesh scene object.

    Returns:
        trimesh.Scene: The transformed scene (or same if scene transform tree doesn't change scale).
    """
    # check for all meshes whether scale is part of their transform path
    # if yes, bake the scale information into vertices
    for node in trimesh_scene.graph.nodes_geometry:
        T, geom_name = trimesh_scene.graph[node]

        scale, _, _, _, _ = tra.decompose_matrix(T)

        if not np.allclose(scale, 1.0):
            log.debug(f"Geometry {geom_name} has non-normalized scale {scale} in transform tree. Will normalize.")

            trimesh_scene.geometry[geom_name].apply_scale(scale)
        
    # Go through the entire transform tree and normalize all transforms
    for (frame_from, frame_to) in list(trimesh_scene.graph.transforms.edge_data.keys()):
        attr = trimesh_scene.graph.transforms.edge_data[(frame_from, frame_to)]
        T = attr['matrix']

        scale, _, T_angles, T_translate, _ = tra.decompose_matrix(T)

        if not np.allclose(scale, 1.0):
            log.debug(f"Normalizing transform (frame_from={frame_from}, frame_to={frame_to}) with scale {scale}.")

            T_new = tra.compose_matrix(angles=T_angles, translate=T_translate)
            attr['matrix'] = T_new

            trimesh_scene.graph.update(
                frame_from=frame_from,
                frame_to=frame_to,
                **attr,
            )

    return trimesh_scene
