# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Procedurally generate assets with randomized parameters."""

# Standard Library
import os
import copy
from typing import List
from functools import partial

# Third Party
import numpy as np
import trimesh
import trimesh.transformations as tra
import yourdfpy
from scipy.interpolate import BSpline

# Local Folder
from . import utils
from .assets import BoxWithHoleAsset, TrimeshAsset, TrimeshSceneAsset, URDFAsset

_DRAWER_DEPTH_PERCENT = 0.9


class TableAsset(TrimeshSceneAsset):
    """A table asset consisting of four legs and one surface."""

    def __init__(
        self,
        width,
        depth,
        height,
        thickness=0.03,
        leg_thickness=0.04,
        leg_margin=0.2,
        leg_as_box=False,
        **kwargs,
    ):
        """A table asset consisting of four legs and one surface.

        .. image:: /../imgs/table_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of table.
            depth (float): Depth of table.
            height (float): Height of table. Includes thickness of surface.
            thickness (float): Thickness of table surface.
            leg_thickness (float): Diameter (in case of cylindric legs) or side length (in case of
                                   box legs) of the table legs.
            leg_margin (float): Percentage between 0 and 1. Distance between edge of table and leg.
                                0 means no distance. 1 means legs are centered (results in one single leg for table). Defaults to 0.1.
            leg_as_box (bool): Wheter to model legs as cylinders or boxes. Defaults to False.
            **kwargs: Arguments will be delegated to constructor of TrimeshSceneAsset.
        """
        assert leg_margin <= 1.0 and leg_margin >= 0.0

        surface = trimesh.primitives.Box(extents=[width, depth, thickness])
        surface.apply_translation([0, 0, height / 2.0])

        legs = []
        for x, y in zip([1, 1, -1, -1], [1, -1, 1, -1]):
            if leg_as_box:
                leg = trimesh.primitives.Box(
                    extents=[leg_thickness, leg_thickness, height - thickness]
                )
            else:
                leg = trimesh.primitives.Cylinder(
                    radius=leg_thickness / 2.0, height=height - thickness
                )

            leg.apply_translation(
                [x * (1.0 - leg_margin) * width / 2.0, y * (1.0 - leg_margin) * depth / 2.0, 0]
            )

            legs.append(leg)

            if leg_margin == 1.0:
                # Only one central leg needed
                break

        scene = trimesh.Scene()
        for mesh, name in zip(legs + [surface], [f'leg_{i}' for i in range(4)] + ['top']):
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)


class CageAsset(TrimeshSceneAsset):
    """A cage-like asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        top_scale=0.7,
        thickness=0.04,
        **kwargs,
    ):
        """A cage asset inspired by the MotionBenchMaker paper, Fig. 6 (column 6).

        .. image:: /../imgs/cage_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of cage.
            depth (float): Depth of cage.
            height (float): Height of cage.
            thickness (float): Thickness of boards.
        """
        boxes = {
            "base": (
                [width - 2.0 * thickness, depth - 2.0 * thickness, thickness],
                [0, 0, 0],
            ),
            "left": (
                [thickness, depth - thickness, height],
                [(-width + thickness) / 2.0, thickness / 2.0, (height - thickness) / 2.0],
            ),
            "right": (
                [thickness, depth - thickness, height],
                [(width - thickness) / 2.0, thickness / 2.0, (height - thickness) / 2.0],
            ),
            "front_upper_bar": (
                [width, thickness, thickness],
                [0, (-depth + thickness) / 2.0, 2.0 * height / 3.0],
            ),
            "front_lower_bar": (
                [width, thickness, thickness],
                [0, (-depth + thickness) / 2.0, 1.0 * height / 3.0],
            ),
            "top": (
                [width - 2.0 * thickness, depth * top_scale, thickness],
                [0, (depth - depth * top_scale) / 2.0, height - thickness],
            ),
            "back": (
                [width - 2.0 * thickness, thickness, height - thickness],
                [0, (depth - thickness) / 2.0, (height / 2.0) - thickness],
            ),
        }

        scene = trimesh.Scene()
        for name in boxes.keys():
            extents, position = boxes[name]
            mesh = trimesh.primitives.Box(extents=extents, transform=tra.translation_matrix(position))
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)


class OpenBoxAsset(TrimeshSceneAsset):
    """An open box asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        front_scale=0.7,
        thickness=0.04,
        angle=45.0,
        **kwargs,
    ):
        """An open box asset inspired by the MotionBenchMaker paper, Fig. 6 (last column).

        .. image:: /../imgs/openbox_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of box.
            depth (float): Depth of box.
            height (float): Height of box.
            front_scale (float): Between 0 and 1. Defaults to 0.7.
            thickness (float): Thickness of boards. Defaults to 0.04.
            angle (float): Angle of lid opening in degrees. Defaults to 45.
            **kwargs: Arguments will be delegated to constructor of TrimeshSceneAsset.
        """
        sin_alpha = np.sin(np.deg2rad(angle))
        cos_alpha = 1.0 - np.cos(np.deg2rad(angle))
        boxes = {
            "base": (
                [width - 2.0 * thickness, depth - 2.0 * thickness, thickness],
                [0, 0, 0],
                [0, 0, 0],
            ),
            "left": (
                [thickness, depth, height - thickness],
                [(-width + thickness) / 2.0, 0, height / 2.0 - thickness],
                [0, 0, 0],
            ),
            "right": (
                [thickness, depth, height - thickness],
                [(width - thickness) / 2.0, 0, height / 2.0 - thickness],
                [0, 0, 0],
            ),
            "front": (
                [width - 2.0 * thickness, thickness, height * front_scale],
                [0, (-depth + thickness) / 2.0, height * front_scale / 2.0 - thickness / 2.0],
                [0, 0, 0],
            ),
            "top": (
                [width, depth, thickness],
                [
                    0,
                    cos_alpha * (depth - thickness) / 2.0,
                    height - thickness + sin_alpha * depth / 2.0,
                ],
                [-np.deg2rad(angle), 0.0, 0.0],
            ),
            "back": (
                [width - 2.0 * thickness, thickness, height - thickness],
                [0, (depth - thickness) / 2.0, height / 2.0 - thickness],
                [0, 0, 0],
            ),
        }

        scene = trimesh.Scene()
        for name in boxes.keys():
            extents, position, angles = boxes[name]
            mesh = trimesh.primitives.Box(
                extents=extents,
                transform=tra.compose_matrix(translate=position, angles=angles),
            )
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name
            )

        super().__init__(scene=scene, **kwargs)


class TableWithBarsAsset(TrimeshSceneAsset):
    """A table asset with four vertical bars."""

    def __init__(
        self,
        width,
        depth,
        height,
        lower_part_scale=0.7,
        lower_shelf_height=0.5,
        thickness=0.04,
        bar_radius=0.04,
        bar_positions=[-0.25, 0.25],
        **kwargs,
    ):
        """A table with four vertical bars in front of it inspired by the MotionBenchMaker, Fig. 6.

        .. image:: /../imgs/tablewithbars_asset.png
            :align: center
            :width: 250px


        Args:
            width (float): Width of table.
            depth (float): Depth of table.
            height (float): Height of table.
            lower_part_scale (float): Between 0 and 1. Defaults to 0.7.
            lower_shelf_height (float): Height of lower shelf in relative coordinates (between 0 and 1). Defaults to 0.5.
            thickness (float): Thickness of boards. Defaults to 0.04.
            bar_radius (float, optional): Radius of the four bars in front of the lower shelf. Defaults to 0.04.
            bar_positions (list[float], optional): List of 2 floats, each in [-0.5, 0.5], describing the relative x-coordinates of the two pairs of bars.
            **kwargs: Arguments will be delegated to constructor of TrimeshSceneAsset.
        """
        boxes = {
            "top": (
                [width, depth, thickness],
                [0, -depth * (1.0 - lower_part_scale) / 2.0, height - thickness],
            ),
            "lower_shelf": (
                [width - 2.0 * thickness, lower_part_scale * depth, thickness],
                [0, 0, lower_shelf_height * height],
            ),
            "left": (
                [thickness, lower_part_scale * depth, height - thickness],
                [-width / 2.0 + thickness / 2.0, 0, height / 2.0 - thickness],
            ),
            "right": (
                [thickness, lower_part_scale * depth, height - thickness],
                [width / 2.0 - thickness / 2.0, 0, height / 2.0 - thickness],
            ),
        }

        cylinders = {
            "left_bar_1": (
                [bar_radius, height - thickness],
                [
                    bar_positions[0] * width - 2.0 * bar_radius,
                    -depth / 2.0 + bar_radius,
                    height / 2.0 - thickness,
                ],
            ),
            "left_bar_2": (
                [bar_radius, height - thickness],
                [
                    bar_positions[0] * width + 2.0 * bar_radius,
                    -depth / 2.0 - bar_radius,
                    height / 2.0 - thickness,
                ],
            ),
            "right_bar_1": (
                [bar_radius, height - thickness],
                [
                    bar_positions[1] * width - 2.0 * bar_radius,
                    -depth / 2.0 - bar_radius,
                    height / 2.0 - thickness,
                ],
            ),
            "right_bar_2": (
                [bar_radius, height - thickness],
                [
                    bar_positions[1] * width + 2.0 * bar_radius,
                    -depth / 2.0 + bar_radius,
                    height / 2.0 - thickness,
                ],
            ),
        }

        scene = trimesh.Scene()
        for name in boxes.keys():
            extents, position = boxes[name]
            mesh = trimesh.primitives.Box(extents=extents, transform=tra.translation_matrix(position))
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )
        for name in cylinders.keys():
            extents, position = cylinders[name]
            mesh = trimesh.primitives.Cylinder(
                radius=extents[0], height=extents[1], transform=tra.translation_matrix(position)
            )
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)


class ShelfAsset(TrimeshSceneAsset):
    """A shelf asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        num_boards,
        board_thickness=0.03,
        backboard_thickness=0.0,
        num_vertical_boards=0,
        num_side_columns=2,
        column_thickness=0.03,
        bottom_board=True,
        cylindrical_columns=True,
        **kwargs,
    ):
        """A shelf asset consisting of an optional back board, N equally spaced shelf boards, and side posts or boards.

        For example, to create the shelf of the MotionBenchMaker paper, Fig. 6 (first column), do:

        .. code-block:: python

            from scene_synthesizer import procedural_assets as pa
            pa.ShelfAsset(
                width=0.8,
                depth=0.8,
                height=1.8,
                num_boards=5,
                num_side_columns=float("inf"),
                bottom_board=False,
                cylindrical_columns=False,
                num_vertical_boards=0,
            ).scene().colorize().show()
        .. image:: /../imgs/shelf_closed_asset.png
            :align: center
            :width: 250px

        To create the shelf of the MotionBenchMaker paper, Fig. 6 (second column), do:

        .. code-block:: python

            from scene_synthesizer import procedural_assets as pa
            pa.ShelfAsset(
                width=0.8,
                depth=0.8,
                height=1.8,
                num_boards=5,
                num_side_columns=2,
                bottom_board=False,
                cylindrical_columns=True,
                num_vertical_boards=1,
            ).scene().colorize().show()
        .. image:: /../imgs/shelf_open_asset.png
            :align: center
            :width: 250px

        To create the cubby of the MotionBenchMaker paper, Fig. 6 (fourth column), do:

        .. code-block:: python

            from scene_synthesizer import procedural_assets as pa
            pa.ShelfAsset(
                width=0.7,
                depth=0.7,
                height=0.35,
                num_boards=2,
                num_side_columns=float("inf"),
                bottom_board=True,
                cylindrical_columns=False,
                num_vertical_boards=0,
            ).scene().colorize().show()
        .. image:: /../imgs/shelf_cubby_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of shelf.
            depth (float): Depth of shelf.
            height (float): Height of shelf.
            num_boards (int): Number of boards, equally spaced between 0.0 (depending on `bottom_board`) and height.
            board_thickness (float): Thickness of each board. Defaults to 0.03.
            backboard_thickness (float, optional): Thickness of back board. If zero no back board is added. Defaults to 0.0.
            num_vertical_boards (int, optional): Number of vertical boards that divide each shelf equally. Defaults to 0.
            num_side_columns (int or float('inf'), optional): Number of columns on each side. If `float('inf')` a side board is added. Defaults to 2.
            column_thickness (float, optional). Radius or side length of side columns (depending on `cylindrical_columns`). Defaults to 0.03.
            bottom_board (bool, optional): Whether to start with the shelf boards at the bottom level. Defaults to True.
            cylindrical_columns (bool, optional): Cylindrical or box-shaped side columns. Defaults to True.
            **kwargs: Arguments will be delegated to constructor of TrimeshSceneAsset.
        """
        boards = []
        board_names = []
        if backboard_thickness > 0:
            back = trimesh.primitives.Box(
                extents=[width, backboard_thickness, height],
                transform=tra.translation_matrix(
                    [0, depth / 2.0 + backboard_thickness / 2.0, height / 2.0]
                ),
            )
            boards.append(back)
            board_names.append('back')

        min_z = +float("inf")
        max_z = -float("inf")
        cnt = 0
        for h in np.linspace(
            0.0 + board_thickness / 2.0, height - board_thickness / 2.0, num_boards
        ):
            if h == board_thickness / 2.0 and not bottom_board:
                continue

            boards.append(
                trimesh.primitives.Box(
                    extents=[width, depth, board_thickness],
                    transform=tra.translation_matrix([0, 0, h]),
                )
            )
            board_names.append(f"board_{cnt}")
            cnt += 1

            min_z = min(min_z, h)
            max_z = max(max_z, h)

        cnt = 0
        for v in np.linspace(-width / 2.0, width / 2.0, num_vertical_boards + 2)[1:-1]:
            boards.append(
                trimesh.primitives.Box(
                    extents=[board_thickness, depth, max_z - min_z],
                    transform=tra.translation_matrix([v, 0, min_z + (max_z - min_z) / 2.0]),
                )
            )
            board_names.append(f"separator_{cnt}")
            cnt += 1

        int_num_side_columns = 1 if np.isinf(num_side_columns) else num_side_columns
        offset = depth / 2.0 if int_num_side_columns == 1 else 0.0
        for j in range(2):
            cnt = 0
            for c in np.linspace(-depth / 2.0, depth / 2.0, int_num_side_columns):
                if cylindrical_columns:
                    column = trimesh.primitives.Cylinder(
                        radius=column_thickness,
                        height=height,
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                else:
                    column = trimesh.primitives.Box(
                        extents=[
                            column_thickness,
                            depth if np.isinf(num_side_columns) else column_thickness,
                            height,
                        ],
                        transform=tra.translation_matrix(
                            [-width / 2.0 + j * width, c + offset, height / 2.0]
                        ),
                    )
                boards.append(column)
                board_names.append(f"post_{j}_{cnt}")
                cnt += 1

        scene = trimesh.Scene()
        for mesh, name in zip(boards, board_names):
            scene.add_geometry(
                geometry=mesh,
                geom_name=name,
                node_name=name,
            )

        super().__init__(scene=scene, **kwargs)


def _add_drawer(
    name,
    model,
    parent,
    width,
    height,
    depth,
    x,
    y,
    cabinet_depth,
    cabinet_outer_wall_thickness,
    cabinet_inner_wall_thickness,
    frontboard_width,
    frontboard_height,
    frontboard_thickness=0.02,
    frontboard_offset=(0, 0),
    wall_thickness=0.005,
    handle_width=None,
    handle_depth=None,
    handle_height=None,
    handle_offset=(0.0, 0.0),
    handle_shape_args=None,
    door_shape_args=None,
):
    """Add a drawer with a handle and a prismatic joint.

    Args:
        parent (str): Name of parent link of prismatic joint.
        width (float): Width of drawer front.
        height (float): Height of drawer front.
        depth (float): Depth of drawer.
        x (float, optional): Local x-coordinate. Defaults to 0.0.
        y (float, optional): Local y-coordinate. Defaults to 0.0.
        cabinet_depth (float): Depth of entire cabinet.
        cabinet_outer_wall_thickness (float): Thickness of outer cabinet walls.
        cabinet_inner_wall_thickness (float): Thickness of outer cabinet walls.
        frontboard_thickness (float, optional): Thickness of front board. Defaults to 0.02.
        wall_thickness (float, optional): Thickness of drawer walls. Defaults to 0.004.
        handle_shape_args (dict, optional): Arguments for procedural handle asset. If None will use box handle. Defaults to None.
        door_shape_args (dict, optional): Arguments for procedural door moldings. If None, will create box-shaped door. Defaults to None.
    """
    boxes = [
        # front
        {
            "origin": tra.translation_matrix(
                (
                    frontboard_offset[0],
                    -cabinet_outer_wall_thickness + frontboard_thickness / 2,
                    frontboard_offset[1],
                )
            ),
            "size": (frontboard_width, frontboard_thickness, frontboard_height),
        },
        # bottom
        {
            "origin": tra.translation_matrix(
                (
                    0,
                    -cabinet_outer_wall_thickness
                    + (depth - wall_thickness + frontboard_thickness) / 2,
                    (wall_thickness + cabinet_inner_wall_thickness - height) / 2,
                )
            ),
            "size": (
                width - 2 * wall_thickness - cabinet_inner_wall_thickness,
                depth - wall_thickness - frontboard_thickness,
                wall_thickness,
            ),
        },
        # left
        {
            "origin": tra.translation_matrix(
                (
                    (-width + wall_thickness + cabinet_inner_wall_thickness) / 2,
                    -cabinet_outer_wall_thickness
                    + (-wall_thickness + depth + frontboard_thickness) / 2,
                    0,
                )
            ),
            "size": (
                wall_thickness,
                depth - wall_thickness - frontboard_thickness,
                height - cabinet_inner_wall_thickness,
            ),
        },
        # right
        {
            "origin": tra.translation_matrix(
                (
                    (-wall_thickness - cabinet_inner_wall_thickness + width) / 2,
                    -cabinet_outer_wall_thickness
                    + (-wall_thickness + depth + frontboard_thickness) / 2,
                    0,
                )
            ),
            "size": (
                wall_thickness,
                depth - wall_thickness - frontboard_thickness,
                height - cabinet_inner_wall_thickness,
            ),
        },
        # back
        {
            "origin": tra.translation_matrix(
                (0, -cabinet_outer_wall_thickness + depth - wall_thickness / 2, 0)
            ),
            "size": (
                width - 2 * wall_thickness,
                wall_thickness,
                height - cabinet_inner_wall_thickness,
            ),
        },
    ]

    visuals = []
    collisions = []

    for i, board in enumerate(boxes):
        if i == 0 and door_shape_args is not None:  # front board
            door_shape_args_wo_mesh_dir = door_shape_args.copy()
            door_shape_args_wo_mesh_dir.pop("tmp_mesh_dir", None)

            # Create nice panel
            door_mesh = CabinetDoorAsset._create_door_mesh(
                *board["size"], **door_shape_args_wo_mesh_dir
            )
            door_mesh_fname = utils.get_random_filename(
                dir=door_shape_args.get("tmp_mesh_dir", "/tmp"),
                prefix="cabinet_door_",
                suffix=".obj",
            )
            door_mesh.export(door_mesh_fname)
            panel_geometry_vis = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))
            panel_geometry_coll = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))
        else:
            panel_geometry_vis = yourdfpy.Geometry(box=yourdfpy.Box(size=board["size"]))
            panel_geometry_coll = yourdfpy.Geometry(box=yourdfpy.Box(size=board["size"]))

        visuals.append(
            yourdfpy.Visual(
                name=f"{name}_board_{i}",
                origin=board["origin"],
                geometry=panel_geometry_vis,
            )
        )
        collisions.append(
            yourdfpy.Collision(
                name=f"{name}_board_{i}",
                origin=board["origin"],
                geometry=panel_geometry_coll,
            )
        )
    inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
    link = yourdfpy.Link(
        name=name,
        inertial=inertial,
        visuals=visuals,
        collisions=collisions,
    )
    model.links.append(link)

    # Create handle link
    handle_frame = name + "_handle"
    handle_link = _create_handle_link(
        name=handle_frame,
        inertial=inertial,
        handle_width=handle_width,
        handle_depth=handle_depth,
        handle_height=handle_height,
        handle_offset=handle_offset,
        handle_shape_args=handle_shape_args,
    )

    # Position the handle link
    handle_pos = np.array(
        (
            frontboard_offset[0],
            -cabinet_outer_wall_thickness,
            frontboard_offset[1],
        )
    )
    handle_rot = np.array((0, np.pi / 2, 0))
    handle_joint_origin = tra.compose_matrix(
        translate=handle_pos,
        angles=handle_rot,
    )
    assert len(handle_link.visuals) == len(handle_link.collisions)
    for v, c in zip(handle_link.visuals, handle_link.collisions):
        v.origin = handle_joint_origin @ v.origin
        c.origin = handle_joint_origin @ c.origin

        model.links[-1].visuals.append(v)
        model.links[-1].collisions.append(c)

    # create prismatic joint
    d_xyz = [
        x - width / 2,
        -cabinet_depth / 2.0 + cabinet_outer_wall_thickness,
        y + height / 2,
    ]
    model.joints.append(
        yourdfpy.Joint(
            name=parent + "_to_" + name,
            type="prismatic",
            parent=parent,
            child=name,
            origin=tra.translation_matrix(d_xyz),
            axis=np.array([0, -1, 0]),
            limit=yourdfpy.Limit(
                effort=1000.0, lower=0.0, upper=depth * _DRAWER_DEPTH_PERCENT, velocity=1.0
            ),
        )
    )


def _add_door(
    model,
    name,
    parent,
    width,
    height,
    asset_width,
    asset_depth,
    asset_height,
    x,
    y,
    frontboard_thickness=0.019,
    opening="right",
    handle_width=None,
    handle_depth=None,
    handle_height=None,
    handle_offset=None,
    handle_shape_args=None,
    door_shape_args=None,
):
    """Adds a cabinet door with a handle and a revolute joint.

    Args:
        parent (str): Name of parent link of revolute joint.
        width (float): Width of door.
        height (height): Height of door.
        x (float, optional): Local x-coordinate of door. Defaults to 0.0.
        y (float, optional): Local y-coordinate of door. Defaults to 0.0.
        frontboard_thickness (float, optional): Thickness of door. Defaults to 0.019.
        opening (str, optional): In which of four directions the door opens. Defaults to "right".
        handle_args (dict, optional): Arguments for procedural handle asset. If None will use box handle. Defaults to None.
        door_shape_args (dict, optional): Arguments for procedural door moldings. If None, will create box-shaped door. Defaults to None.
    """
    # Create door link
    offset_x, offset_y = 0.0, 0.0
    handle_rotation = [0, 0, 0]
    if handle_offset is None:
        handle_offset = (0.0, 0.05)
    if opening == "left":
        offset_x = -width / 2.0
        pos = [x + width, y]
        axis = np.array([0, 0, 1])
        handle_pos = [-width + handle_offset[1] + handle_depth / 2.0, 0, handle_offset[0]]
    elif opening == "right":
        offset_x = width / 2.0
        pos = [x - width, y + height / 2.0]
        axis = np.array([0, 0, -1])
        handle_pos = [width - handle_offset[1] - handle_depth / 2.0, 0, handle_offset[0]]
    elif opening == "top":
        offset_x = +width / 2.0
        offset_y = height / 2.0
        pos = [x - width, y]
        axis = np.array([1, 0, 0])
        handle_pos = [
            width / 2.0 + handle_offset[0],
            0,
            height - handle_offset[1] - handle_depth / 2.0,
        ]
        handle_rotation = [0, np.pi / 2.0, 0]
    elif opening == "bottom":
        offset_x = +width / 2.0
        offset_y = -height / 2.0
        pos = [x - width, y + height]
        axis = np.array([-1, 0, 0])
        handle_pos = [
            width / 2.0 + handle_offset[0],
            0,
            -height + handle_offset[1] + handle_depth / 2.0,
        ]
        handle_rotation = [0, np.pi / 2.0, 0]

    inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))

    # create fancy door
    door_shape_args_dict = door_shape_args if door_shape_args is not None else {}
    visual_geometries, collision_geometries = CabinetDoorAsset._create_urdf_geometries(
        name, width, frontboard_thickness, height, geometry_origin=tra.translation_matrix([offset_x, frontboard_thickness / 2, offset_y]), **door_shape_args_dict
    )
    door_link = yourdfpy.Link(
        name=name,
        inertial=inertial,
        visuals=visual_geometries,
        collisions=collision_geometries,
    )

    # Create handle link
    handle_frame = name + "_handle"
    handle_link = _create_handle_link(
        name=handle_frame,
        inertial=inertial,
        handle_width=handle_width,
        handle_depth=handle_depth,
        handle_height=handle_height,
        handle_offset=handle_offset,
        handle_shape_args=handle_shape_args,
    )

    # Position the handle link
    handle_joint_origin = tra.compose_matrix(
        translate=handle_pos,
        angles=handle_rotation,
    )

    # Add links to drawer model
    model.links.append(door_link)

    # add handle to link
    assert len(handle_link.visuals) == len(handle_link.collisions)
    for v, c in zip(handle_link.visuals, handle_link.collisions):
        v.origin = handle_joint_origin @ v.origin
        c.origin = handle_joint_origin @ c.origin

        model.links[-1].visuals.append(v)
        model.links[-1].collisions.append(c)

    d_xyz = [
        pos[0],
        -asset_depth / 2.0,
        pos[1],
    ]
    model.joints.append(
        yourdfpy.Joint(
            name=parent + "_to_" + name,
            type="revolute",
            parent=parent,
            child=name,
            origin=tra.translation_matrix(d_xyz),
            axis=axis,
            limit=yourdfpy.Limit(
                effort=1000.0,
                velocity=0.1,
                lower=0.0,
                upper=np.pi / 2.0,
            ),
        )
    )

class CabinetAsset(URDFAsset):
    """A cabinet asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        compartment_mask,
        compartment_types,
        compartment_interior_masks=None,
        outer_wall_thickness=0.01,
        inner_wall_thickness=0.01,
        drawer_wall_thickness=0.005,
        frontboard_thickness=0.02,
        frontboard_overlap=0.0,
        compartment_widths=None,
        compartment_heights=None,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """A cabinet.

        .. image:: /../imgs/cabinet_asset.png
            :align: center
            :width: 250px
        
        Args:
            width (float): Width of the cabinet or None. If None compartment_widths are used as absolute sizes otherwise relative ones.
            depth (float): Depth of the cabinet (excluding possible handles).
            height (float): Height of the cabinet or None. If None compartment_heights are used as absolute sizes otherwise relative ones.
            compartment_mask (list[list[int]] or np.ndarray): A 2D matrix of type int which represents the segmentation map of the cabinet layout. Same numbers indicate same compartment.
            compartment_types (list[str]): A list of strings of ["none", "open", "closed", "door_left", "door_right", "door_top", "door_bottom", "drawer"] depending on the type of the i-th compartment (i being the entry in the compartment_mask).
            compartment_interior_masks (dict[list[list[int]] or dict[np.ndarray], optional): A dictionary of compartment masks that represent the internal structure of a single compartment (e.g. multiple shelves behind a door). The dictionary key is the entry to the compartment_mask. Defaults to None.
            outer_wall_thickness (float, optional): Thickness of outer walls of the cabinet. Defaults to 0.01.
            inner_wall_thickness (float, optional): Thickness of inner walls and surfaces of the cabinet.
            drawer_wall_thickness (float, optional): Thickness of drawer walls and surfaces.
            frontboard_thickness (float, optional): Thickness of front boards.
            frontboard_overlap (float, optional): Overlap of front boards with outer and inner walls, between 0 (no overlap) and 1 (maximum overlap). Defaults to 0.0.
            compartment_widths (list[float], optional): List of widths of compartment columns. Must have as many elements as compartment_mask has columns. If None all columns have equal width that sum to width. Is considered relative if width is defined. Defaults to None.
            compartment_heights (list[float], optional): List of heights of compartment rows. Must have as many elements as compartment_mask has rows. If None all rows have equal heights that sum to height. Is considered relative if height is defined. Defaults to None.
            handle_width (float, optional): Defaults to 0.1682.
            handle_height (float, optional): Defaults to 0.038.
            handle_depth (float, optional): Defaults to 0.024.
            handle_offset (tupe(float, float), optional): Defaults to None.
            handle_shape_args (dict, optional): Arguments for procedural handles. If None, will create handle made out of boxes. Defaults to None.
            door_shape_args (dict, optional): Arguments for procedural door moldings. If None, will create box-shaped door. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.

        Raises:
            ValueError: If neither width nor compartment_widths is defined.
            ValueError: If neither height nor compartment_heights is defined.
            ValueError: If compartment type is unknown.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=CabinetAsset._create_yourdfpy_model(
                width=width,
                depth=depth,
                height=height,
                compartment_mask=compartment_mask,
                compartment_types=compartment_types,
                compartment_interior_masks=compartment_interior_masks,
                compartment_widths=compartment_widths,
                compartment_heights=compartment_heights,
                outer_wall_thickness=outer_wall_thickness,
                inner_wall_thickness=inner_wall_thickness,
                drawer_wall_thickness=drawer_wall_thickness,
                frontboard_thickness=frontboard_thickness,
                frontboard_overlap=frontboard_overlap,
                handle_width=handle_width,
                handle_depth=handle_depth,
                handle_height=handle_height,
                handle_offset=handle_offset,
                handle_shape_args=handle_shape_args,
                door_shape_args=door_shape_args,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @staticmethod
    def _create_yourdfpy_model(
        width,
        depth,
        height,
        compartment_mask,
        compartment_types,
        compartment_widths,
        compartment_heights,
        compartment_interior_masks=None,
        outer_wall_thickness=0.01,
        inner_wall_thickness=0.01,
        drawer_wall_thickness=0.005,
        frontboard_thickness=0.02,
        frontboard_overlap=0.0,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
    ):
        if width is None and compartment_widths is None:
            raise ValueError("Either width or compartment_widths or both need to be defined.")

        if height is None and compartment_heights is None:
            raise ValueError("Either height or compartment_heights or both need to be defined.")

        compartment_mask = np.asarray(compartment_mask)
        if len(compartment_mask.shape) != 2:
            raise ValueError(
                f"compartment_mask.shape={compartment_mask.shape} needs to have two dimensions."
            )

        if not np.all(
            sorted(np.unique(compartment_mask)) == list(range(np.max(compartment_mask) + 1))
        ):
            raise ValueError(
                "The compartment_mask needs to contain all increasing integers from"
                f" min(compartment_mask) to max(compartment_mask): {compartment_mask}"
            )

        assert len(compartment_types) == len(
            np.unique(compartment_mask)
        ), f"{len(compartment_types)} vs {len(np.unique(compartment_mask))}"

        possible_types = {
            "none",
            "open",
            "closed",
            "door_left",
            "door_right",
            "door_top",
            "door_bottom",
            "drawer",
        }
        for type in compartment_types:
            if type not in possible_types:
                raise ValueError(
                    f"Compartment type {type} unknown. Must be one of {possible_types}."
                )

        # check door shape arguments
        if not isinstance(door_shape_args, list):
            door_shape_args_list = [door_shape_args]*len(compartment_types)
        else:
            door_shape_args_list = door_shape_args

        assert len(compartment_types) == len(door_shape_args_list)

        if compartment_widths is None:
            # set default compartment_widths
            compartment_widths = np.ones(compartment_mask.shape[1])

            # remove one level of thickness since there +1 number of walls than compartments
            compartment_widths *= (width - 2.0 * outer_wall_thickness) / compartment_mask.shape[1]
        else:
            assert len(compartment_widths) == compartment_mask.shape[1]

            compartment_widths = compartment_widths.copy()
            if width is None:
                # set total width
                width = np.sum(compartment_widths) + outer_wall_thickness * 2.0
                compartment_widths = np.array(compartment_widths)
            else:
                # interpret compartment_widths as relative sizes
                total_width = np.sum(compartment_widths)
                compartment_widths = (compartment_widths / total_width) * (
                    width - outer_wall_thickness * 2.0
                )

        if compartment_heights is None:
            # set default compartment_heights
            compartment_heights = np.ones(compartment_mask.shape[0])

            # remove one level of thickness since there +1 number of surfaces than compartments
            compartment_heights *= (height - 2.0 * outer_wall_thickness) / compartment_mask.shape[0]
        else:
            assert len(compartment_heights) == compartment_mask.shape[0]

            compartment_heights = compartment_heights.copy()
            if height is None:
                # set total height
                height = np.sum(compartment_heights) + outer_wall_thickness * 2.0
                compartment_heights = np.array(compartment_heights)
            else:
                # interpret compartment_heights as relative sizes
                total_height = np.sum(compartment_heights)
                compartment_heights = (compartment_heights / total_height) * (
                    height - outer_wall_thickness * 2.0
                )

        compartment_widths_without_walls = compartment_widths.copy()
        if len(compartment_widths_without_walls) > 1:
            compartment_widths_without_walls[0] -= inner_wall_thickness / 2.0
            compartment_widths_without_walls[-1] -= inner_wall_thickness / 2.0
        if len(compartment_widths_without_walls) > 2:
            compartment_widths_without_walls[1:-1] -= inner_wall_thickness

        compartment_heights_without_walls = compartment_heights.copy()
        if len(compartment_heights_without_walls) > 1:
            compartment_heights_without_walls[0] -= inner_wall_thickness / 2.0
            compartment_heights_without_walls[-1] -= inner_wall_thickness / 2.0
        if len(compartment_heights_without_walls) > 2:
            compartment_heights_without_walls[1:-1] -= inner_wall_thickness

        model = yourdfpy.Robot(name="Cabinet")

        model.links.append(yourdfpy.Link(name="corpus"))

        # build corpus
        boxes = {
            "top": (
                [width, depth - frontboard_thickness, outer_wall_thickness],
                tra.translation_matrix(
                    [
                        0,
                        frontboard_thickness / 2.0,
                        height - outer_wall_thickness / 2.0,
                    ]
                ),
            ),
            # "bottom": (
            #     [width - 2.0 * thickness, depth - 2.0 * thickness, thickness],
            #     tra.translation_matrix([0, 0, thickness / 2.0]),
            # ),
            "left": (
                [
                    outer_wall_thickness,
                    depth - outer_wall_thickness - frontboard_thickness,
                    height - outer_wall_thickness,
                ],
                tra.translation_matrix(
                    [
                        -width / 2.0 + outer_wall_thickness / 2.0,
                        -(outer_wall_thickness - frontboard_thickness) / 2.0,
                        height / 2.0 - outer_wall_thickness / 2.0,
                    ]
                ),
            ),
            "right": (
                [
                    outer_wall_thickness,
                    depth - outer_wall_thickness - frontboard_thickness,
                    height - outer_wall_thickness,
                ],
                tra.translation_matrix(
                    [
                        width / 2.0 - outer_wall_thickness / 2.0,
                        -(outer_wall_thickness - frontboard_thickness) / 2.0,
                        height / 2.0 - outer_wall_thickness / 2.0,
                    ]
                ),
            ),
            "back": (
                [width, outer_wall_thickness, height - outer_wall_thickness],
                tra.translation_matrix(
                    [
                        0,
                        depth / 2.0 - outer_wall_thickness / 2.0,
                        (height - outer_wall_thickness) / 2.0,
                    ]
                ),
            ),
        }

        def compartment_neighboring_wall_thicknesses(x1, y1, x2, y2):
            # Returns the wall thicknesses in order: left, right, top, bottom
            # Returns half of the inner wall thickness, since this function is only used for frontboard overlap calculations
            thicknesses = []
            thicknesses.append(outer_wall_thickness if x1 == 0 else inner_wall_thickness / 2.0)
            thicknesses.append(
                outer_wall_thickness
                if x2 == compartment_mask.shape[1] - 1
                else inner_wall_thickness / 2.0
            )
            thicknesses.append(outer_wall_thickness if y1 == 0 else inner_wall_thickness / 2.0)
            thicknesses.append(
                outer_wall_thickness
                if y2 == compartment_mask.shape[0] - 1
                else inner_wall_thickness / 2.0
            )

            return thicknesses

        # go through all compartments
        translation_x = np.insert(np.cumsum(compartment_widths), 0, 0)
        translation_z = np.insert(np.cumsum(compartment_heights), 0, 0)
        
        wall_widths = np.array([outer_wall_thickness] + [inner_wall_thickness] * (compartment_mask.shape[1] - 1) + [outer_wall_thickness])
        wall_heights = np.array([outer_wall_thickness] + [inner_wall_thickness] * (compartment_mask.shape[0] - 1) + [outer_wall_thickness])
        translation_x_interior = np.insert(np.cumsum(compartment_widths_without_walls), 0, 0) + np.cumsum(wall_widths)
        translation_z_interior = np.insert(np.cumsum(compartment_heights_without_walls), 0, 0) + np.cumsum(wall_heights)
        
        translation_x_drawer = [outer_wall_thickness]
        for w in compartment_widths_without_walls[:-1]:
            translation_x_drawer.append(translation_x_drawer[-1] + w + inner_wall_thickness)
        translation_z_drawer = [outer_wall_thickness]
        for h in compartment_heights_without_walls[:-1]:
            translation_z_drawer.append(translation_z_drawer[-1] + h + inner_wall_thickness)

        frontboard_translation_x = [outer_wall_thickness]
        frontboard_translation_z = [outer_wall_thickness]
        for i, x in enumerate(translation_x[1:]):
            if i != 0 or i != len(translation_x[1:]) - 1:
                frontboard_translation_x.append(
                    x + outer_wall_thickness + inner_wall_thickness / 2.0
                )
            else:
                frontboard_translation_x.append(x + outer_wall_thickness)
        for i, z in enumerate(translation_z[1:]):
            if i != 0 or i != len(translation_z[1:]) - 1:
                frontboard_translation_z.append(
                    z + outer_wall_thickness + inner_wall_thickness / 2.0
                )
            else:
                frontboard_translation_z.append(z + outer_wall_thickness)

        # add "internal" boards
        for yi, xi in np.ndindex(*compartment_mask.shape):
            surface_width = compartment_widths[xi]
            wall_height = compartment_heights[yi]

            # check for right wall
            if (
                xi + 1 < compartment_mask.shape[1]
                and compartment_mask[yi, xi] != compartment_mask[yi, xi + 1]
            ):
                # add wall
                box_size = [
                    inner_wall_thickness,
                    depth - outer_wall_thickness - frontboard_thickness,
                    wall_height,
                ]

                transform = tra.translation_matrix(
                    [
                        -width / 2.0 + outer_wall_thickness + translation_x[xi] + surface_width,
                        -(outer_wall_thickness - frontboard_thickness) / 2.0,
                        height - translation_z[yi] - wall_height / 2.0 - outer_wall_thickness,
                    ]
                )
                boxes[f"separator_{xi}_{yi}"] = (box_size, transform)

            # check for surface below
            last_row = yi + 1 == compartment_mask.shape[0]
            if (
                yi + 1 < compartment_mask.shape[0]
                and compartment_mask[yi, xi] != compartment_mask[yi + 1, xi]
            ) or last_row:
                surface_thickness = outer_wall_thickness if last_row else inner_wall_thickness
                box_size = [
                    surface_width,
                    depth - outer_wall_thickness - frontboard_thickness,
                    surface_thickness,
                ]
                transform = tra.translation_matrix(
                    [
                        -width / 2.0
                        + outer_wall_thickness
                        + translation_x[xi]
                        + surface_width / 2.0,
                        -(outer_wall_thickness - frontboard_thickness) / 2.0,
                        height
                        - outer_wall_thickness
                        - translation_z[yi]
                        - wall_height
                        - last_row * surface_thickness / 2.0,
                    ]
                )
                boxes[f"surface_{xi}_{yi}"] = (box_size, transform)

        if compartment_interior_masks is not None:
            # add shelving inside non drawer compartments
            for i in compartment_interior_masks:
                if compartment_types[i] in ("drawer", "closed"):
                    continue

                interior_mask = np.asarray(compartment_interior_masks[i])
                # make sure shape of interior mask is consistent with compartment mask

                # assume rectangular mask
                # figure out translation in x and y for i-th compartment
                compartment_yi_start, compartment_xi_start = np.min(np.argwhere(compartment_mask == i), axis=0)

                compartment_x = translation_x_interior[compartment_xi_start]
                compartment_y = translation_z_interior[compartment_yi_start]

                # figure out width and height
                compartment_width = compartment_widths_without_walls[compartment_xi_start]
                compartment_height = compartment_heights_without_walls[compartment_yi_start]
            
                interior_widths = [compartment_width / interior_mask.shape[1]] * interior_mask.shape[1]
                interior_heights = [compartment_height / interior_mask.shape[0]] * interior_mask.shape[0]
                
                for yi, xi in np.ndindex(*interior_mask.shape):
                    # check for right wall
                    if (
                        xi + 1 < interior_mask.shape[1]
                        and interior_mask[yi, xi] != interior_mask[yi, xi + 1]
                    ):
                        box_size = [
                            inner_wall_thickness,
                            depth - outer_wall_thickness - frontboard_thickness,
                            interior_heights[yi],
                        ]

                        transform = tra.translation_matrix(
                            [
                                -width / 2.0 + compartment_x + sum(interior_widths[:xi+1]),
                                -(outer_wall_thickness - frontboard_thickness) / 2.0,
                                height - compartment_y - sum(interior_heights[:yi]) - box_size[2] / 2.0,
                            ]
                        )
                        boxes[f"shelf_separator_{i}_{xi}_{yi}"] = (box_size, transform)
                    
                    # check for surface below
                    if (
                        yi + 1 < interior_mask.shape[0]
                        and interior_mask[yi, xi] != interior_mask[yi + 1, xi]
                    ):
                        box_size = [
                            interior_widths[xi],
                            depth - outer_wall_thickness - frontboard_thickness,
                            inner_wall_thickness,
                        ]

                        transform = tra.translation_matrix(
                            [
                                -width / 2.0
                                + compartment_x
                                + sum(interior_widths[:xi])
                                + box_size[0] / 2.0,
                                -(outer_wall_thickness - frontboard_thickness) / 2.0,
                                height
                                - compartment_y
                                - sum(interior_heights[:yi+1])
                                # - outer_wall_thickness
                                # - wall_height,
                            ]
                        )
                        boxes[f"shelf_{i}_{xi}_{yi}"] = (box_size, transform)

        # add doors, drawers, closings
        missing_closed_compartments = []
        for compartment, compartment_type in enumerate(compartment_types):
            if compartment_type in ["open", "none"]:
                continue
            
            # extract rectangular mask
            y, x = np.where(compartment_mask == compartment)
            x1, x2 = x.min(), x.max()
            y1, y2 = y.min(), y.max()

            # Ensure that all masks have a rectangular shape
            if not (compartment_mask[y1 : y2 + 1, x1 : x2 + 1] == compartment).all():
                if compartment_type not in ["open", "none", "closed"]:
                    raise ValueError(
                        f"compartment_mask entry '{compartment}' has a non-rectangular shape. Can't"
                        " construct non-rectangular drawers or doors."
                    )
                else:
                    missing_closed_compartments.extend(list(zip(y, x)))
                    continue

            x_first, x_last = (x1 == 0), (x2 == compartment_mask.shape[1] - 1)
            y_first, y_last = (y1 == 0), (y2 == compartment_mask.shape[0] - 1)

            yi = y1
            xi = x1

            surrounding_wall_thicknesses = compartment_neighboring_wall_thicknesses(x1, y1, x2, y2)
            surface_width = compartment_widths[x1 : x2 + 1].sum()
            wall_height = compartment_heights[y1 : y2 + 1].sum()

            compartment_w = (
                compartment_widths_without_walls[x1 : x2 + 1].sum()
                + (x2 - x1) * inner_wall_thickness
            )
            compartment_h = (
                compartment_heights_without_walls[y1 : y2 + 1].sum()
                + (y2 - y1) * inner_wall_thickness
            )

            frontboard_width, frontboard_height = None, None
            if not (x_first or x_last):
                # inner front board
                frontboard_width = (
                    surface_width - inner_wall_thickness + frontboard_overlap * inner_wall_thickness
                )
            elif (x_first and not x_last) or (not x_first and x_last):
                # one side over outer wall
                frontboard_width = (
                    surface_width
                    - inner_wall_thickness / 2.0
                    + frontboard_overlap * (inner_wall_thickness / 2.0 + outer_wall_thickness)
                )
            else:
                # both ends over outer wall
                frontboard_width = surface_width + frontboard_overlap * outer_wall_thickness * 2.0

            if not (y_first or y_last):
                # inner front board
                frontboard_height = (
                    wall_height + frontboard_overlap * inner_wall_thickness - inner_wall_thickness
                )
            elif (y_first and not y_last) or (not y_first and y_last):
                # one side over outer wall
                frontboard_height = (
                    wall_height
                    - inner_wall_thickness / 2.0
                    + frontboard_overlap * (inner_wall_thickness / 2.0 + outer_wall_thickness)
                )
            else:
                # both ends over outer wall
                frontboard_height = wall_height + frontboard_overlap * outer_wall_thickness * 2.0

            if compartment_type == "closed":
                boxes[f"closed_{xi}_{yi}"] = (
                    [
                        frontboard_width,
                        frontboard_thickness,
                        frontboard_height,
                    ],
                    tra.translation_matrix(
                        [
                            -width / 2.0
                            + frontboard_translation_x[xi]
                            - surrounding_wall_thicknesses[0] * frontboard_overlap
                            + frontboard_width / 2.0,
                            -depth / 2.0 + frontboard_thickness / 2.0,
                            height
                            - frontboard_translation_z[yi]
                            + surrounding_wall_thicknesses[2] * frontboard_overlap
                            - frontboard_height / 2.0,
                        ]
                    ),
                )
            elif compartment_type == "door_left":
                _add_door(
                    model=model,
                    name=f"door_{xi}_{yi}",
                    parent=model.links[0].name,
                    x=-width / 2.0
                    + frontboard_translation_x[xi]
                    - surrounding_wall_thicknesses[0] * frontboard_overlap,
                    y=height
                    - frontboard_translation_z[yi]
                    - frontboard_height / 2.0
                    + surrounding_wall_thicknesses[2] * frontboard_overlap,
                    width=frontboard_width,
                    height=frontboard_height,
                    frontboard_thickness=frontboard_thickness,
                    asset_width=width,
                    asset_depth=depth,
                    asset_height=height,
                    opening="left",
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args_list[compartment],
                )
            elif compartment_type == "door_right":
                _add_door(
                    model=model,
                    name=f"door_{xi}_{yi}",
                    parent=model.links[0].name,
                    x=-width / 2.0
                    + frontboard_translation_x[xi]
                    - surrounding_wall_thicknesses[0] * frontboard_overlap
                    + frontboard_width,
                    y=height
                    - frontboard_translation_z[yi]
                    - frontboard_height
                    + surrounding_wall_thicknesses[2] * frontboard_overlap,
                    width=frontboard_width,
                    height=frontboard_height,
                    frontboard_thickness=frontboard_thickness,
                    asset_width=width,
                    asset_depth=depth,
                    asset_height=height,
                    opening="right",
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args_list[compartment],
                )
            elif compartment_type == "door_top":
                _add_door(
                    model=model,
                    name=f"door_{xi}_{yi}",
                    parent=model.links[0].name,
                    x=-width / 2.0
                    + frontboard_translation_x[xi]
                    - surrounding_wall_thicknesses[0] * frontboard_overlap
                    + frontboard_width,
                    y=height
                    - frontboard_translation_z[yi]
                    - frontboard_height
                    + surrounding_wall_thicknesses[2] * frontboard_overlap,
                    width=frontboard_width,
                    height=frontboard_height,
                    frontboard_thickness=frontboard_thickness,
                    asset_width=width,
                    asset_depth=depth,
                    asset_height=height,
                    opening="top",
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args_list[compartment],
                )
            elif compartment_type == "door_bottom":
                _add_door(
                    model=model,
                    name=f"door_{xi}_{yi}",
                    parent=model.links[0].name,
                    x=-width / 2.0
                    + frontboard_translation_x[xi]
                    - surrounding_wall_thicknesses[0] * frontboard_overlap
                    + frontboard_width,
                    y=height
                    - frontboard_translation_z[yi]
                    - frontboard_height
                    + surrounding_wall_thicknesses[2] * frontboard_overlap,
                    width=frontboard_width,
                    height=frontboard_height,
                    frontboard_thickness=frontboard_thickness,
                    asset_width=width,
                    asset_depth=depth,
                    asset_height=height,
                    opening="bottom",
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args_list[compartment],
                )
            elif compartment_type == "drawer":
                # add drawer
                _add_drawer(
                    model=model,
                    name=f"drawer_{xi}_{yi}",
                    parent=model.links[0].name,
                    x=-width / 2.0 + translation_x_drawer[xi] + compartment_w,
                    y=height - translation_z_drawer[yi] - compartment_h,
                    width=compartment_w,
                    height=compartment_h,
                    depth=(depth - outer_wall_thickness) * _DRAWER_DEPTH_PERCENT,
                    frontboard_thickness=frontboard_thickness,
                    frontboard_width=frontboard_width,
                    frontboard_height=frontboard_height,
                    frontboard_offset=(
                        (surrounding_wall_thicknesses[1] - surrounding_wall_thicknesses[0])
                        / 2.0
                        * frontboard_overlap,
                        (surrounding_wall_thicknesses[2] - surrounding_wall_thicknesses[3])
                        / 2.0
                        * frontboard_overlap,
                    ),
                    cabinet_depth=depth,
                    cabinet_outer_wall_thickness=outer_wall_thickness,
                    cabinet_inner_wall_thickness=inner_wall_thickness,
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args_list[compartment],
                )
            else:
                raise ValueError(f"Unknown compartment type: {compartment_type}")

        # Add closings for compartments that do not have a rectangular front
        for yi, xi in missing_closed_compartments:
            surface_width = compartment_widths[xi]
            wall_height = compartment_heights[yi]

            boxes[f"closed_{xi}_{yi}"] = (
                [surface_width, frontboard_thickness, wall_height],
                tra.translation_matrix(
                    [
                        -width / 2.0
                        + outer_wall_thickness / 2.0
                        + translation_x[xi]
                        + surface_width / 2.0,
                        -depth / 2.0 + frontboard_thickness / 2.0,
                        height - translation_z[yi] - wall_height / 2.0 - outer_wall_thickness / 2.0,
                    ]
                ),
            )

        # add boxes to first link
        for box_name, box_props in boxes.items():
            model.links[0].visuals.append(
                yourdfpy.Visual(
                    name=box_name,
                    origin=box_props[1],
                    geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=box_props[0])),
                )
            )
            model.links[0].collisions.append(
                yourdfpy.Collision(
                    name=box_name,
                    origin=box_props[1],
                    geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=box_props[0])),
                )
            )

        return model


class CubbyShelfAsset(URDFAsset):
    """A cubby shelf asset."""

    def __init__(
        self,
        width,
        depth,
        height,
        compartment_mask,
        compartment_types,
        compartment_widths=None,
        compartment_heights=None,
        wall_thickness=0.01,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """A shelf composed of individual cubbies.
        In contrast to CabinetAsset, each cubby has a separate left, right, top, bottom, and back wall geometry.
        This allows e.g. to render segmentation maps that highlight individual cubbies based on their geometry (this wouldn't be possible with CabinetAsset).
        In contrast to CabinetAsset, the resulting shelf will have more geometries/boxes.

        .. image:: /../imgs/cubby_shelf_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of the shelf or None. If None compartment_widths are used as absolute sizes otherwise relative ones.
            depth (float): Depth of the shelf (excluding possible handles).
            height (float): Height of the shelf or None. If None compartment_heights are used as absolute sizes otherwise relative ones.
            compartment_mask (list[list[float]] or np.ndarray): A 2D matrix of type int which represents the segmentation map of the shelf layout. Same numbers indicate same compartment.
            compartment_types (list[str]): A list of strings of ["none", "open", "closed", "door_left", "door_right", "door_top", "door_bottom", "drawer"] depending on the type of the i-th compartment (i being the entry in the compartment_mask).
            compartment_widths (list[float], optional): List of widths of compartment columns. Must have as many elements as compartment_mask has columns. If None all columns have equal width that sum to width. Is considered relative if width is defined. Defaults to None.
            compartment_heights (list[float], optional): List of heights of compartment rows. Must have as many elements as compartment_mask has rows. If None all rows have equal heights that sum to height. Is considered relative if height is defined. Defaults to None.
            wall_thickness (float, optional): Thickness of walls of the cubbies. Defaults to 0.01.
            handle_width (float, optional): Defaults to 0.1682.
            handle_height (float, optional): Defaults to 0.038.
            handle_depth (float, optional): Defaults to 0.024.
            handle_offset (tupe(float, float), optional): Defaults to None. x-z offset for handle placement location.
            handle_shape_args (dict, optional): Arguments for handles. If None, will create handle made out of boxes. Defaults to None.
            door_shape_args (dict, optional): Arguments for procedural door moldings. If None, will create box-shaped door. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.

        Raises:
            ValueError: If neither width nor compartment_widths is defined.
            ValueError: If neither height nor compartment_heights is defined.
            ValueError: If compartment type is unknown.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=CubbyShelfAsset.create_yourdfpy_model(
                width=width,
                depth=depth,
                height=height,
                compartment_mask=compartment_mask,
                compartment_types=compartment_types,
                compartment_widths=compartment_widths,
                compartment_heights=compartment_heights,
                wall_thickness=wall_thickness,
                handle_width=handle_width,
                handle_depth=handle_depth,
                handle_height=handle_height,
                handle_offset=handle_offset,
                handle_shape_args=handle_shape_args,
                door_shape_args=door_shape_args,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @staticmethod
    def _mask_to_bbox(mask):
        """Returns bounding box corners as 4-tuple given a binary mask.

        Args:
            mask (np.ndarray): Binary mask.

        Returns:
            tuple(float): Two corners of the bounding box: (x1, y1, x2, y2)
        """
        y, x = np.where(mask)
        if len(x) == 0:
            raise ValueError("Mask should have non-zero values")

        x1, x2 = x.min(), x.max()
        y1, y2 = y.min(), y.max()

        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def create_cubby_URDF_geometries(width, depth, height, thickness, closed=False, name_prefix=""):
        visual_geoms = []
        collision_geoms = []

        boxes = {
            f"{name_prefix}right_wall": (
                [thickness, depth - thickness, height - thickness],
                tra.translation_matrix(
                    [
                        (width - thickness) / 2.0,
                        -thickness / 2.0,
                        -thickness / 2.0,
                    ]
                ),
            ),
            f"{name_prefix}left_wall": (
                [thickness, depth - thickness, height - thickness],
                tra.translation_matrix(
                    [
                        (-width + thickness) / 2.0,
                        -thickness / 2.0,
                        -thickness / 2.0,
                    ]
                ),
            ),
            f"{name_prefix}back_wall": (
                [width, thickness, height - thickness],
                tra.translation_matrix([0, (depth - thickness) / 2.0, -thickness / 2.0]),
            ),
            f"{name_prefix}bottom_wall": (
                [width - 2.0 * thickness, depth - thickness, thickness],
                tra.translation_matrix([0, -thickness / 2.0, (-height + thickness) / 2.0]),
            ),
            f"{name_prefix}top_wall": (
                [width, depth, thickness],
                tra.translation_matrix([0, 0, (height - thickness) / 2.0]),
            ),
        }

        if closed:
            boxes[f"{name_prefix}front_wall"] = (
                [width, thickness, height],
                tra.translation_matrix([0, (-depth - thickness) / 2.0, 0]),
            )

        for box_name, box_props in boxes.items():
            visual_geoms.append(
                yourdfpy.Visual(
                    name=box_name,
                    origin=box_props[1],
                    geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=box_props[0])),
                )
            )
            collision_geoms.append(
                yourdfpy.Collision(
                    name=box_name,
                    origin=box_props[1],
                    geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=box_props[0])),
                )
            )

        return visual_geoms, collision_geoms

    @staticmethod
    def transform_geometries(geoms, transform):
        for geom in geoms:
            geom.origin = transform @ geom.origin

    @staticmethod
    def _add_door(
        model,
        name,
        parent,
        width,
        height,
        x,
        y,
        z,
        frontboard_thickness,
        opening="right",
        handle_width=None,
        handle_depth=None,
        handle_height=None,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
    ):
        """Adds a cabinet door with a handle and a revolute joint.

        Args:
            model (yourdfpy.URDF): URDF model.
            name (str): Name of the new door link.
            parent (str): Name of parent link of revolute joint.
            width (float): Width of door.
            height (height): Height of door.
            x (float): Local x-coordinate of door.
            y (float): Local y-coordinate of door.
            z (float): Local z-coordinate of door.
            frontboard_thickness (float): Thickness of door.
            opening (str, optional): In which of four directions the door opens. Defaults to "right".
            handle_args (dict, optional): Arguments for procedural handle asset. If None will use box handle. Defaults to None.
            door_shape_args (dict, optional): Arguments for procedural door moldings. If None, will create box-shaped door. Defaults to None.
        """

        # Create door link
        shape_offset = [0.0, frontboard_thickness / 2.0, 0]
        handle_rotation = [0, 0, 0]
        if handle_offset is None:
            handle_offset = (0.0, 0.05)
        if opening == "left":
            shape_offset[0] = -width / 2.0
            pos = [x + width / 2.0, y, z]
            axis = np.array([0, 0, 1])
            handle_pos = [-width + handle_offset[1] + handle_depth / 2.0, 0, handle_offset[0]]
        elif opening == "right":
            shape_offset[0] = width / 2.0
            pos = [x - width / 2.0, y, z]
            axis = np.array([0, 0, -1])
            handle_pos = [width - handle_offset[1] - handle_depth / 2.0, 0, handle_offset[0]]
        elif opening == "top":
            shape_offset[0] = +width / 2.0
            shape_offset[2] = height / 2.0
            pos = [x - width / 2.0, y, z - height / 2.0]
            axis = np.array([1, 0, 0])
            handle_pos = [
                width / 2.0 + handle_offset[0],
                0,
                height - handle_offset[1] - handle_depth / 2.0,
            ]
            handle_rotation = [0, np.pi / 2.0, 0]
        elif opening == "bottom":
            shape_offset = [+width / 2.0, -frontboard_thickness / 2.0, -height / 2.0]
            pos = [x - width / 2.0, y + frontboard_thickness, z + height / 2.0]
            axis = np.array([-1, 0, 0])
            handle_pos = [
                width / 2.0 + handle_offset[0],
                -frontboard_thickness,
                -height + handle_offset[1] + handle_depth / 2.0,
            ]
            handle_rotation = [0, np.pi / 2.0, 0]

        inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))

        if door_shape_args is None:
            visual_geometry = yourdfpy.Geometry(
                box=yourdfpy.Box(size=(width, frontboard_thickness, height))
            )
            collision_geometry = yourdfpy.Geometry(
                box=yourdfpy.Box(size=(width, frontboard_thickness, height))
            )
        else:
            door_shape_args_wo_mesh_dir = door_shape_args.copy()
            door_shape_args_wo_mesh_dir.pop("tmp_mesh_dir", None)

            door_mesh = CabinetDoorAsset._create_door_mesh(
                width, frontboard_thickness, height, **door_shape_args_wo_mesh_dir
            )
            door_mesh_fname = utils.get_random_filename(
                dir=door_shape_args.get("tmp_mesh_dir", "/tmp"),
                prefix="cabinet_door_",
                suffix=".obj",
            )
            door_mesh.export(door_mesh_fname)
            visual_geometry = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))
            collision_geometry = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))

        door_link = yourdfpy.Link(
            name=name,
            inertial=inertial,
            visuals=[
                yourdfpy.Visual(
                    name=f"{name}_door",
                    geometry=visual_geometry,
                    origin=tra.translation_matrix(shape_offset),
                )
            ],
            collisions=[
                yourdfpy.Collision(
                    name=f"{name}_door",
                    geometry=collision_geometry,
                    origin=tra.translation_matrix(shape_offset),
                )
            ],
        )

        # Create handle link
        handle_frame = name + "_handle"
        handle_link = _create_handle_link(
            name=handle_frame,
            inertial=inertial,
            handle_width=handle_width,
            handle_depth=handle_depth,
            handle_height=handle_height,
            handle_offset=handle_offset,
            handle_shape_args=handle_shape_args,
        )

        # Position the handle link
        handle_joint_origin = tra.compose_matrix(
            translate=handle_pos,
            angles=handle_rotation,
        )

        # Add links to drawer model
        model.links.append(door_link)

        # add handle to link
        assert len(handle_link.visuals) == len(handle_link.collisions)
        for v, c in zip(handle_link.visuals, handle_link.collisions):
            v.origin = handle_joint_origin @ v.origin
            c.origin = handle_joint_origin @ c.origin

            model.links[-1].visuals.append(v)
            model.links[-1].collisions.append(c)

        model.joints.append(
            yourdfpy.Joint(
                name=parent + "_to_" + name,
                type="revolute",
                parent=parent,
                child=name,
                origin=tra.translation_matrix(pos),
                axis=axis,
                limit=yourdfpy.Limit(
                    effort=1000.0,
                    velocity=0.1,
                    lower=0.0,
                    upper=np.pi / 2.0,
                ),
            )
        )

    @staticmethod
    def create_yourdfpy_model(
        width,
        depth,
        height,
        compartment_mask,
        compartment_types,
        compartment_widths,
        compartment_heights,
        wall_thickness,
        handle_width,
        handle_depth,
        handle_height,
        handle_offset,
        handle_shape_args,
        door_shape_args,
    ):
        # Check arguments
        if width is None and compartment_widths is None:
            raise ValueError("Either width or compartment_widths or both need to be defined.")

        if height is None and compartment_heights is None:
            raise ValueError("Either height or compartment_heights or both need to be defined.")

        compartment_mask = np.asarray(compartment_mask)
        assert len(compartment_mask.shape) == 2
        assert len(compartment_types) == len(np.unique(compartment_mask)), (
            f"Wrong length of compartment_types {len(compartment_types)} vs"
            f" {len(np.unique(compartment_mask))}"
        )

        possible_types = {
            "none",
            "open",
            "closed",
            "door_left",
            "door_right",
            "door_top",
            "door_bottom",
            "drawer",
        }
        for type in compartment_types:
            if type not in possible_types:
                raise ValueError(
                    f"Compartment type {type} unknown. Must be one of {possible_types}."
                )

        if compartment_widths is None:
            # set default compartment_widths
            compartment_widths = np.ones(compartment_mask.shape[1])

            # remove one level of thickness since there +1 number of walls than compartments
            compartment_widths *= width / compartment_mask.shape[1]
        else:
            assert len(compartment_widths) == compartment_mask.shape[1]

            compartment_widths = compartment_widths.copy()
            if width is None:
                # set total width
                width = np.sum(compartment_widths)
                compartment_widths = np.array(compartment_widths)
            else:
                # interpret compartment_widths as relative sizes
                total_width = np.sum(compartment_widths)
                compartment_widths = width * compartment_widths / total_width

        if compartment_heights is None:
            # set default compartment_heights
            compartment_heights = np.ones(compartment_mask.shape[0])

            # remove one level of thickness since there +1 number of surfaces than compartments
            compartment_heights *= height / compartment_mask.shape[0]
        else:
            assert len(compartment_heights) == compartment_mask.shape[0]

            compartment_heights = compartment_heights.copy()
            if height is None:
                # set total height
                height = np.sum(compartment_heights)
                compartment_heights = np.array(compartment_heights)
            else:
                # interpret compartment_heights as relative sizes
                total_height = np.sum(compartment_heights)
                compartment_heights = height * compartment_heights / total_height

        model = yourdfpy.Robot(name="CubbyShelf")
        model.links.append(yourdfpy.Link(name="corpus"))

        base_link = model.links[-1]

        # add cubbies
        for m in np.unique(compartment_mask):
            mask = compartment_mask == m
            x, y, w, h = CubbyShelfAsset._mask_to_bbox(mask == 1)

            if not np.all(mask[y : y + h + 1, x : x + w + 1]):
                raise ValueError(
                    f"The compartment_mask needs to have only rectangular cubbies. Segment {m} is"
                    " not rectangular."
                )

            compartment_width = compartment_widths[x : x + w + 1].sum()
            compartment_height = compartment_heights[y : y + h + 1].sum()

            visual_geoms, collision_geoms = CubbyShelfAsset.create_cubby_URDF_geometries(
                width=compartment_width,
                depth=depth,
                height=compartment_height,
                thickness=wall_thickness,
                closed=compartment_types[m] == "closed",
                name_prefix=f"cubby_{m}_",
            )

            compartment_x = compartment_widths[:x].sum() + compartment_width / 2.0
            compartment_z = compartment_heights[:y].sum() + compartment_height / 2.0
            CubbyShelfAsset.transform_geometries(
                visual_geoms + collision_geoms,
                tra.translation_matrix([compartment_x, 0, -compartment_z]),
            )

            base_link.visuals.extend(visual_geoms)
            base_link.collisions.extend(collision_geoms)

            if compartment_types[m] in ["door_left", "door_right", "door_top", "door_bottom"]:
                CubbyShelfAsset._add_door(
                    model=model,
                    name=f"door_{m}",
                    parent=base_link.name,
                    x=compartment_x,
                    y=-depth / 2.0 - wall_thickness,
                    z=-compartment_z,
                    width=compartment_width,
                    height=compartment_height,
                    frontboard_thickness=wall_thickness,
                    opening=compartment_types[m].split("_")[-1],
                    handle_width=handle_width,
                    handle_depth=handle_depth,
                    handle_height=handle_height,
                    handle_offset=handle_offset,
                    handle_shape_args=handle_shape_args,
                    door_shape_args=door_shape_args,
                )

        return model


def _create_path(width, depth, straight_ratio, curvature_ratio, num_segments):
    curvature_ratio = min(max(1e-5, curvature_ratio), 1.0)
    knots = [0, 0, 0, curvature_ratio * 0.5, 1.0 - curvature_ratio * 0.5, 1, 1, 1]

    straight_ratio = max(0.0, straight_ratio)
    coefficients = [
        [-width / 2.0, depth],
        [-straight_ratio * width / 2.0, 0],
        [0, 0],
        [straight_ratio * width / 2.0, 0],
        [width / 2.0, depth],
    ]

    bspline = BSpline(t=knots, c=coefficients, k=2)
    x = np.linspace(0, 1, num_segments + 1)
    y = bspline(x)

    # merge center pieces
    if num_segments > 1:
        id_center = len(x) // 2
        id_center_right = id_center
        id_center_left = id_center
        while True:
            id_center_left -= 1
            if y[id_center_left, 1] != y[id_center, 1]:
                id_center_left += 1
                break
            if id_center_left == 0:
                break
        while True:
            id_center_right += 1
            if y[id_center_right, 1] != y[id_center, 1]:
                id_center_right -= 1
                break
            if id_center_right == 0:
                break

        if id_center_left != id_center_right:
            y = np.vstack((y[: id_center_left + 1], y[id_center_right:]))

    res = np.c_[y, np.zeros(len(y))]
    return res


def _ellipse(radius, segments, aspect_ratio):
    angles = np.linspace(0, 2.0 * np.pi, segments, endpoint=False)
    return np.c_[np.sin(angles + np.pi / 4.0), np.cos(angles + np.pi / 4.0)] * [
        radius * aspect_ratio,
        radius,
    ]


def _create_handle(
    width,
    depth,
    height,
    straight_ratio,
    curvature_ratio,
    num_segments_curvature,
    num_segments_cross_section,
    aspect_ratio_cross_section,
):
    path = _create_path(
        width=width,
        depth=height,
        straight_ratio=straight_ratio,
        curvature_ratio=curvature_ratio,
        num_segments=num_segments_curvature,
    )
    poly = trimesh.path.polygons.Polygon(
        _ellipse(
            radius=depth / 2.0,
            segments=num_segments_cross_section,
            aspect_ratio=aspect_ratio_cross_section,
        )
    )
    mesh = trimesh.creation.sweep_polygon(poly, path, engine="triangle")

    if np.any((mesh.extents / np.hstack((mesh.extents[-1], mesh.extents[1:]))) > 1e10):
        raise ValueError(
            f"This particular combination of handle parameters failed:\n width={width},\n"
            f" depth={depth},\n height={height},\n straight_ratio={straight_ratio},\n"
            f" curvature_ratio={curvature_ratio},\n"
            f" aspect_ratio_cross_section={aspect_ratio_cross_section},\n"
            f" num_segments_curvature={num_segments_curvature},\n"
            f" num_segments_cross_section={num_segments_cross_section}\n\n Try a different one."
        )

    mesh.fix_normals()
    mesh.apply_scale(np.array([width, height, depth]) / mesh.extents)

    return mesh


def _create_handle_link(
    name, inertial, handle_width, handle_depth, handle_height, handle_offset, handle_shape_args
):
    if handle_shape_args is None:
        inner_width = 0.04
        inner_height = max(handle_height - handle_depth, handle_depth / 2.0)
        part_0_height = max(handle_height - inner_height, 0.001)  # Make sure this is not negative
        handle_link = yourdfpy.Link(
            name=name,
            inertial=inertial,
            visuals=[
                yourdfpy.Visual(
                    name=name + "_part_0",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[part_0_height, handle_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix([-handle_height + part_0_height / 2.0, 0, 0]),
                ),
                yourdfpy.Visual(
                    name=name + "_part_1",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[inner_height, inner_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix(
                        [-inner_height / 2.0, -handle_width / 2.0 + inner_width / 2.0, 0]
                    ),
                ),
                yourdfpy.Visual(
                    name=name + "_part_2",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[inner_height, inner_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix(
                        [-inner_height / 2.0, handle_width / 2.0 - inner_width / 2.0, 0]
                    ),
                ),
            ],
            collisions=[
                yourdfpy.Collision(
                    name=name + "_part_0",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[part_0_height, handle_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix([-handle_height + part_0_height / 2.0, 0, 0]),
                ),
                yourdfpy.Collision(
                    name=name + "_part_1",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[inner_height, inner_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix(
                        [-inner_height / 2.0, -handle_width / 2.0 + inner_width / 2.0, 0]
                    ),
                ),
                yourdfpy.Collision(
                    name=name + "_part_2",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[inner_height, inner_width, handle_depth])
                    ),
                    origin=tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                    @ tra.translation_matrix(
                        [-inner_height / 2.0, handle_width / 2.0 - inner_width / 2.0, 0]
                    ),
                ),
            ],
        )
    else:
        handle_mesh_fname = utils.get_random_filename(
            dir=handle_shape_args.get("tmp_mesh_dir", "/tmp"),
            prefix="cabinet_handle_",
            suffix=".obj",
        )

        handle_shape_args_wo_mesh_dir = handle_shape_args.copy()
        handle_shape_args_wo_mesh_dir.pop("tmp_mesh_dir", None)
        handle_mesh = _create_handle(
            width=handle_width,
            depth=handle_depth,
            height=handle_height,
            **handle_shape_args_wo_mesh_dir,
        )
        handle_mesh.export(handle_mesh_fname)
        handle_transform = tra.compose_matrix(
            translate=[0, -handle_mesh.bounds[1, 1], 0],
            angles=[0, np.pi / 2.0, 0],
        )

        handle_link = yourdfpy.Link(
            name="door",
            visuals=[
                yourdfpy.Visual(
                    name="door_handle",
                    origin=handle_transform,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=handle_mesh_fname)),
                ),
            ],
            collisions=[
                yourdfpy.Collision(
                    name="door_handle",
                    origin=handle_transform,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=handle_mesh_fname)),
                ),
            ],
        )
    return handle_link


def _create_knob_link(
    name,
    tmp_mesh_dir="/tmp",
    **knob_kwargs,
):
    knob_mesh_fname = utils.get_random_filename(
        dir=tmp_mesh_dir,
        prefix=f"{name}_knob_",
        suffix=".obj",
    )
    if knob_kwargs:
        knob_asset = KnobAsset(
            width=knob_kwargs.get("width", 0.03),
            height=knob_kwargs.get("height", 0.03),
            depth=knob_kwargs.get("depth", 0.05),
        )
    else:
        knob_asset = KnobAsset.random()
    knob_asset.mesh().export(knob_mesh_fname)
    knob_transform = tra.compose_matrix(
        translate=[0, 0, 0],
        angles=[np.pi / 2, 0, np.pi],
    )
    knob_link = yourdfpy.Link(
        name=name,
        visuals=[
            yourdfpy.Visual(
                name=f"{name}_knob",
                origin=knob_transform,
                geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=knob_mesh_fname)),
            ),
        ],
        collisions=[
            yourdfpy.Collision(
                name=f"{name}_knob",
                origin=knob_transform,
                geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=knob_mesh_fname)),
            ),
        ],
    )
    return knob_link


class HandleAsset(TrimeshAsset):
    """A handle asset."""

    def __init__(
        self,
        width,
        height,
        depth,
        straight_ratio,
        curvature_ratio,
        num_segments_curvature,
        num_segments_cross_section,
        aspect_ratio_cross_section=1.0,
        **kwargs,
    ):
        """Procedural handle asset.

        .. image:: /../imgs/handle_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of handle.
            height (float): Height of handle.
            depth (float): Depth of handle.
            straight_ratio (float): A handle consists of two curved ends and a straight part in-between. This is the length of the straight part. Defined as a ratio of the total width. Must be >=0.
            curvature_ratio (float): A curvature parameter between 0 and 1.0. 0 is not smooth, 1.0 is maximally smooth.
            num_segments_curvature (int): Number of segments of the handle's spine.
            num_segments_cross_section (int): The cross section of the handle is a discritized ellipse with num_segments_cross_segmentation many segments. In case of four, a rectangle/square is created.
            aspect_ratio_cross_section (float, optional): Aspect ratio of the ellipse of the cross section. Defaults to 1.0.
            **kwargs: Arguments will be delegated to constructor of TrimeshAsset.

        Returns:
            TrimeshAsset: A handle.
        """
        assert 0 <= straight_ratio

        super().__init__(
            _create_handle(
                width=width,
                depth=depth,
                height=height,
                straight_ratio=straight_ratio,
                curvature_ratio=curvature_ratio,
                num_segments_curvature=num_segments_curvature,
                num_segments_cross_section=num_segments_cross_section,
                aspect_ratio_cross_section=aspect_ratio_cross_section,
            ),
            **kwargs,
        )

    @classmethod
    def random_shape_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {}
        params["straight_ratio"] = kwargs.get("straight_ratio", rng.uniform(0.3, 2.0))
        params["curvature_ratio"] = kwargs.get("curvature_ratio", rng.uniform(0, 2))
        params["num_segments_curvature"] = kwargs.get(
            "num_segments_curvature", rng.integers(10, 20)
        )
        params["num_segments_cross_section"] = kwargs.get(
            "num_segments_cross_section", rng.integers(5, 15)
        )
        params["aspect_ratio_cross_section"] = kwargs.get(
            "aspect_ratio_cross_section", rng.uniform(0.2, 1.3)
        )

        return params

    @classmethod
    def random_size_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {}
        params["width"] = kwargs.get("width", rng.uniform(0.2 - 0.06, 0.2 + 0.06))
        params["depth"] = kwargs.get("depth", rng.uniform(0.02, 0.04))
        params["height"] = kwargs.get("height", rng.uniform(0.08 - 0.02, 0.08 + 0.06))

        return params

    @classmethod
    def random_params(cls, seed=None, **kwargs):
        params = cls.random_shape_params(seed=seed, **kwargs)
        params.update(cls.random_size_params(seed=seed, **kwargs))
        return params

    @classmethod
    def random(cls, seed=None, **kwargs):
        params = cls.random_params(seed=seed, **kwargs)
        handle = cls(**params)

        return handle


class KnobAsset(TrimeshAsset):
    """A knob asset."""

    def __init__(
        self,
        width=0.06,
        height=0.06,
        depth=0.06,
        base_ratio=0.5,
        knob_ratio=0.5,
        knot_0=0.43,
        knot_1=0.57,
        num_sections=32,
        num_depth_sections=20,
        **kwargs,
    ):
        """Procedural knob asset based on revolving a B-Spline around its x-axis.

        The origin is at the base of the knob, z-axis along protrusion.

        .. image:: /../imgs/knob_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of the resulting knob.
            height (float): Height of the resulting knob.
            depth (float): Depth of the resulting knob.
            base_ratio (float, optional): A ratio \\in ]0, 1[ indicating the diameter of the stem of the knob in relation to the full width. Defaults to 0.5.
            knob_ratio (float, optional): A ratio \\in ]0, 1[ indicating the depth of the knob in relation to the full depth. Defaults to 0.5.
            knot_0 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.43.
            knot_1 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.57.
            num_sections (int, optional): Number of interpolated sections during revolution of B-spline around its x-axis. Can be used to create different knob shapes, such as triangules and squares. Defaults to 32.
            num_depth_sections (int, optional): Number of interpolated sections along B-Spline. Defaults to 20.
        """

        mesh = KnobAsset._create_knob_mesh(
            width=width,
            height=height,
            depth=depth,
            base_ratio=base_ratio,
            knob_ratio=knob_ratio,
            knot_0=knot_0,
            knot_1=knot_1,
            num_sections=num_sections,
            num_depth_sections=num_depth_sections,
        )

        super().__init__(
            mesh,
            **kwargs,
        )

    @staticmethod
    def _create_knob_mesh(
        width,
        height,
        depth,
        base_ratio,
        knob_ratio,
        knot_0,
        knot_1,
        num_sections,
        num_depth_sections,
    ):
        """Procedural knob asset based on revolving a B-Spline around its x-axis.

        Args:
            width (float): Width of the resulting knob.
            height (float): Height of the resulting knob.
            depth (float): Depth of the resulting knob.
            base_ratio (float, optional): A ratio \\in ]0, 1[ indicating the diameter of the stem of the knob in relation to the full width. Defaults to 0.5.
            knob_ratio (float, optional): A ratio \\in ]0, 1[ indicating the depth of the knob in relation to the full depth. Defaults to 0.5.
            knot_0 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.43.
            knot_1 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.57.
            num_sections (int, optional): Number of interpolated sections during revolution of B-spline around its x-axis. Can be used to create different knob shapes, such as triangules and squares. Defaults to 32.
            num_depth_sections (int, optional): Number of interpolated sections along B-Spline. Defaults to 20.

        Raises:
            ValueError: If width != height.
            ValueError: If base_ratio not \\in ]0, 1[.
            ValueError: If knob_ratio not \\in ]0, 1[.
            ValueError: If knot_0 not in \\in ]0, knot_1].
            ValueError: If knot_1 not in \\in [knot_0, 1[.

        Returns:
            trimesh.Trimesh: A mesh representing the knob.
        """
        if width != height:
            raise ValueError(
                f"Currently, only symmetric knobs can be created, i.e., width == height."
            )

        if not (1 > base_ratio > 0):
            raise ValueError(f"base_ratio for KnobAsset must be between 0 and 1")

        if not (1 > knob_ratio > 0):
            raise ValueError(f"knob_ratio for KnobAsset must be between 0 and 1")

        if not (0 < knot_0 <= knot_1):
            raise ValueError(
                "knot_0 for KnobAsset must be between 0 (exclusive) and"
                f" knot_1={knot_1} (inclusive)"
            )

        if not (knot_0 <= knot_1 < 1):
            raise ValueError(
                "knot_0 for KnobAsset must be between 1 (exclusive) and"
                f" knot_0={knot_0} (inclusive)"
            )

        half_height = height / 2.0
        knots = [0, 0, 0, knot_0, knot_1, 1, 1, 1]
        points = [
            [0, base_ratio * half_height],
            [(1.0 - knob_ratio) * depth, base_ratio * half_height],
            [(1.0 - knob_ratio) * depth, 1.0 * half_height],
            [1.0 * depth, 1.0 * half_height],
            [1.0 * depth, 0],
        ]
        bspline = BSpline(t=knots, c=points, k=2)
        x = np.linspace(0, 1, num_depth_sections)

        # add origin to make mesh watertight
        linestring = np.concatenate([[[0.0, 0.0]], bspline(x)[:, ::-1]])

        return trimesh.creation.revolve(
            linestring=linestring,
            sections=num_sections,
        )

    @classmethod
    def random_shape_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {
            "base_ratio": kwargs.get("base_ratio", rng.uniform(0.0, 1.0)),
            "knob_ratio": kwargs.get("knob_ratio", rng.uniform(0.0, 1.0)),
            "knot_0": kwargs.get("knot_0", rng.uniform(0.0, 0.5)),
            "num_sections": kwargs.get("num_sections", rng.integers(16, 64)),
            "num_depth_sections": kwargs.get("num_depth_sections", rng.integers(15, 25)),
        }

        params["knot_1"] = kwargs.get("knot_1", 1 - params["knot_0"])
        return params

    @classmethod
    def random_size_params(cls, seed=None, width=None, height=None, depth=None, **kwargs):
        rng = np.random.default_rng(seed)

        depth = rng.uniform(0.04, 0.06) if depth is None else depth
        if width is None and height is None:
            width = height = rng.uniform(0.02, 0.06)
        elif width is None:
            width = height
        elif height is None:
            height = width
        params = {"width": width, "height": height, "depth": depth}

        return params

    @classmethod
    def random_params(cls, seed=None, **kwargs):
        params = cls.random_shape_params(seed=seed, **kwargs)
        params.update(cls.random_size_params(seed=seed, **kwargs))
        return params

    @classmethod
    def random(cls, seed=None, **kwargs):
        params = cls.random_params(seed=seed, **kwargs)

        knob = cls(**params)

        return knob


class RefrigeratorAsset(URDFAsset):
    """A refrigerator asset."""

    def __init__(
        self,
        width=0.76,
        depth=0.76,
        height=1.64,
        freezer_compartment_height=0.49,
        thickness=0.07,
        num_shelves=4,
        num_door_shelves=3,
        door_shelf_depth=0.1,
        door_shelf_holder_height=0.06,
        shelf_thickness=0.01,
        handle_left=True,
        handle_vertical=True,
        handle_length=0.3,
        handle_thickness=0.07,
        handle_depth=0.05,
        handle_offset=(0.03, 0.05, 0.03),
        handle_shape_args={
            "straight_ratio": 0.5,
            "curvature_ratio": 0.8,
            "num_segments_curvature": 10,
            "num_segments_cross_section": 8,
            "aspect_ratio_cross_section": 0.5,
            "tmp_mesh_dir": "/tmp",
        },
        door_shape_args={
            "straight_ratio": 0.9,
            "curvature_ratio": 0.9,
            "num_segments": 9,
            "tmp_mesh_dir": "/tmp",
        },
        **kwargs,
    ):
        """A refrigerator asset with an optional freezer compartment, procedural handle, and beveled door edges.

        .. image:: /../imgs/refrigerator_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of fridge. Defaults to 0.76.
            depth (float): Depth of fridge. Defaults to 0.76.
            height (float): Height of fridge. Defaults to 1.64.
            freezer_compartment_height (float): Height of the freezer compartment. For no freezer, set to zero. Must be less than height of fridge. Defaults to 0.49
            thickness (float): Thickness of walls (including door). Defaults to 0.07.
            num_shelves (int, optional): Number of shelves inside the refrigerator. Defaults to 4.
            num_door_shelves (int, optional): Number of shelves inside the refrigerator door. Defaults to 3.
            door_shelf_depth (float, optional): The depth of the shelves connected to the inside of the door. The shelves in the fridge corpus will be retracted by the same amoung. Defaults to 0.1.
            door_shelf_holder_height (float, optional): Height of the holding rim above the door shelves. Disappears if zero. Defaults to 0.06.
            shelf_thickness (float, optional). Thickness of shelves inside the refrigerator. Defaults to 0.01.
            handle_left (bool, optional): Whether door opens to left or right. Defaults to True.
            handle_vertical (bool, optional): Whether door handle is vertical or horizontal. Defaults to True.
            handle_length (float, optional): Length of handle(s). Defaults to 0.3.
            handle_thickness (float, optional): Thickness of handle(s). Defaults to 0.07.
            handle_depth (float, optional): Depth of handle(s). Defaults to 0.05.
            handle_offset (3-tuple[float], optional): Offset of handle in lateral direction, direction of height, and direction of depth. Takes handle_left into account. Defaults to (0.03, 0.05, 0.03).
            handle_shape_args (dict, optional): Arguments for procedural handle shape. Defaults to fridge handle.
            door_shape_args (dict, optional): Arguments for procedural door shape. Defaults to fridge door.
            **kwargs: Arguments will be delegated to constructor of URDFAsset.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=self._create_yourdfpy_model(
                width=width,
                depth=depth,
                height=height,
                thickness=thickness,
                freezer_compartment_height=freezer_compartment_height,
                num_shelves=num_shelves,
                num_door_shelves=num_door_shelves,
                door_shelf_depth=door_shelf_depth,
                door_shelf_holder_height=door_shelf_holder_height,
                shelf_thickness=shelf_thickness,
                handle_left=handle_left,
                handle_vertical=handle_vertical,
                handle_length=handle_length,
                handle_thickness=handle_thickness,
                handle_depth=handle_depth,
                handle_offset=handle_offset,
                handle_shape_args=handle_shape_args,
                door_shape_args=door_shape_args,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    def _create_yourdfpy_model(
        self,
        width,
        depth,
        height,
        thickness,
        freezer_compartment_height,
        num_shelves,
        num_door_shelves,
        door_shelf_depth,
        door_shelf_holder_height,
        shelf_thickness,
        handle_left,
        handle_vertical,
        handle_length,
        handle_offset,
        handle_thickness,
        handle_depth,
        handle_shape_args,
        door_shape_args,
    ):
        model = yourdfpy.Robot(name="refrigerator")

        boxes = {
            "top": (
                [width, depth - 2.0 * thickness, thickness],
                [0, 0, height - thickness / 2.0],
            ),
            "bottom": (
                [width - 2.0 * thickness, depth - 2.0 * thickness, thickness],
                [0, 0, thickness / 2.0],
            ),
            "left": (
                [thickness, depth - 2.0 * thickness, height - thickness],
                [-width / 2.0 + thickness / 2.0, 0, height / 2.0 - thickness / 2.0],
            ),
            "right": (
                [thickness, depth - 2.0 * thickness, height - thickness],
                [width / 2.0 - thickness / 2.0, 0, height / 2.0 - thickness / 2.0],
            ),
            "back": ([width, thickness, height], [0, depth / 2.0 - thickness / 2.0, height / 2.0]),
        }
        shelf_size = [
            width - 2.0 * thickness,
            depth - door_shelf_depth - 2.0 * thickness,
            shelf_thickness,
        ]
        for i in range(num_shelves):
            boxes[f"shelf_{i}"] = (
                shelf_size,
                [
                    0,
                    door_shelf_depth / 2.0,
                    thickness / 2.0
                    + (height - freezer_compartment_height) * (i + 1) / (num_shelves + 1),
                ],
            )
        # Add bottom shelf on top of outer wall called "bottom"
        boxes[f"shelf_{num_shelves}"] = (
            shelf_size,
            [0, door_shelf_depth / 2.0, thickness + shelf_thickness / 2.0]
        )

        if freezer_compartment_height > height:
            raise ValueError(
                f"freezer_compartment_height={freezer_compartment_height} must be less or equal to"
                f" height={height} of refrigerator."
            )

        boxes["freezer_separator"] = (
            [width - 2.0 * thickness, depth - 2.0 * thickness, thickness],
            [0, 0, height - freezer_compartment_height],
        )

        model.links.append(
            yourdfpy.Link(
                name="corpus",
                visuals=[
                    yourdfpy.Visual(
                        name=key,
                        origin=tra.translation_matrix(boxes[key][1]),
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=boxes[key][0])),
                    )
                    for key in boxes
                ],
                collisions=[
                    yourdfpy.Collision(
                        name=key,
                        origin=tra.translation_matrix(boxes[key][1]),
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=boxes[key][0])),
                    )
                    for key in boxes
                ],
            )
        )

        # lower door
        door_transform = np.eye(4)
        if door_shape_args is None:
            door_transform = tra.translation_matrix([0, thickness/2.0, (height - freezer_compartment_height - thickness / 2.0) / 2.0])
            door_panel_geometry = yourdfpy.Geometry(box=yourdfpy.Box([width, thickness, height - freezer_compartment_height - thickness / 2.0]))
        else:
            coords = _create_path(
                width=width,
                depth=thickness,
                straight_ratio=door_shape_args["straight_ratio"],
                curvature_ratio=door_shape_args["curvature_ratio"],
                num_segments=door_shape_args["num_segments"],
            )
            poly = trimesh.path.polygons.Polygon(coords)
            door_mesh = trimesh.creation.extrude_polygon(
                polygon=poly,
                height=height - freezer_compartment_height - thickness / 2.0,
                # See https://github.com/mikedh/trimesh/pull/1683
                engine="triangle",
            )
            door_mesh_fname = utils.get_random_filename(
                dir=door_shape_args["tmp_mesh_dir"], prefix="refrigerator_door_", suffix=".obj"
            )
            door_mesh.export(door_mesh_fname)
            door_panel_geometry = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))

        if handle_vertical:
            handle_length = min(
                handle_length, height - freezer_compartment_height - 2.0 * handle_offset[1]
            )
        else:
            handle_length = min(handle_length, width - 2.0 * handle_offset[0])

        if handle_shape_args is None:
            door_handle_geometry = yourdfpy.Geometry(box=yourdfpy.Box([handle_length, handle_thickness, handle_depth]))
        else:
            handle_mesh_fname = utils.get_random_filename(
                dir=handle_shape_args["tmp_mesh_dir"], prefix="refrigerator_handle_", suffix=".obj"
            )
            handle_mesh = _create_handle(
                width=handle_length,
                depth=handle_thickness,
                height=handle_depth,
                straight_ratio=handle_shape_args["straight_ratio"],
                curvature_ratio=handle_shape_args["curvature_ratio"],
                num_segments_cross_section=handle_shape_args["num_segments_cross_section"],
                num_segments_curvature=handle_shape_args["num_segments_curvature"],
                aspect_ratio_cross_section=handle_shape_args["aspect_ratio_cross_section"],
            )
            handle_mesh.export(handle_mesh_fname)
            door_handle_geometry = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=handle_mesh_fname))

        if handle_left:
            door_transform = door_transform @ tra.translation_matrix([-width / 2.0, -thickness, thickness / 4.0])
            if handle_vertical:
                handle_transform = (
                    tra.translation_matrix(
                        [
                            -width + handle_thickness / 2.0 + handle_offset[0],
                            -thickness - handle_depth + handle_offset[2],
                            height
                            - freezer_compartment_height
                            - handle_length / 2.0
                            - handle_offset[1],
                        ]
                    )
                    @ tra.euler_matrix(0, np.pi / 2.0, 0)
                )
            else:
                handle_transform = tra.translation_matrix(
                    [
                        -width + handle_length / 2.0 + handle_offset[0],
                        -thickness - handle_depth + handle_offset[2],
                        height
                        - freezer_compartment_height
                        - handle_thickness / 2.0
                        - handle_offset[1],
                    ]
                )
            joint_origin = tra.translation_matrix([width / 2.0, -depth / 2.0 + thickness, 0.0])
            joint_axis = np.array([0, 0, 1.0])
        else:
            door_transform = door_transform @ tra.translation_matrix([width / 2.0, -thickness, thickness / 4.0])
            if handle_vertical:
                handle_transform = (
                    tra.translation_matrix(
                        [
                            width - handle_thickness / 2.0 - handle_offset[0],
                            -thickness - handle_depth + handle_offset[2],
                            height
                            - freezer_compartment_height
                            - handle_length / 2.0
                            - handle_offset[1],
                        ]
                    )
                    @ tra.euler_matrix(0, np.pi / 2.0, 0)
                )
            else:
                handle_transform = tra.translation_matrix(
                    [
                        width - handle_length / 2.0 - handle_offset[0],
                        -thickness - handle_depth + handle_offset[2],
                        height
                        - freezer_compartment_height
                        - handle_thickness / 2.0
                        - handle_offset[1],
                    ]
                )
            joint_origin = tra.translation_matrix([-width / 2.0, -depth / 2.0 + thickness, 0.0])
            joint_axis = -np.array([0, 0, 1.0])

        model.links.append(
            yourdfpy.Link(
                name="door",
                visuals=[
                    yourdfpy.Visual(
                        name="door_panel",
                        origin=door_transform,
                        geometry=door_panel_geometry,
                    ),
                    yourdfpy.Visual(
                        name="door_handle",
                        origin=handle_transform,
                        geometry=door_handle_geometry,
                    ),
                ],
                collisions=[
                    yourdfpy.Collision(
                        name="door_panel",
                        origin=door_transform,
                        geometry=door_panel_geometry,
                    ),
                    yourdfpy.Collision(
                        name="door_handle",
                        origin=handle_transform,
                        geometry=door_handle_geometry,
                    ),
                ],
            )
        )

        if num_door_shelves > 0:
            door_height = height - freezer_compartment_height - thickness / 2.0
            door_shelf_width = width - 2.0 * thickness
            door_holder_height = min(
                door_height / (2.0 * (num_door_shelves + 1)), door_shelf_holder_height
            )
            for i in range(num_door_shelves):
                door_shelf_boxes = {
                    f"door_shelf_{i}": (
                        [door_shelf_width, door_shelf_depth, shelf_thickness],
                        [
                            -width / 2.0 if handle_left else width / 2.0,
                            thickness / 2.0,
                            thickness + shelf_thickness + i * (door_height / (num_door_shelves)),
                        ],
                    )
                }

                if door_shelf_holder_height > 0:
                    door_shelf_boxes.update(
                        {
                            f"door_holder_0_{i}": (
                                [door_shelf_width, shelf_thickness, shelf_thickness],
                                [
                                    -width / 2.0 if handle_left else width / 2.0,
                                    thickness / 2.0 + door_shelf_depth / 2.0,
                                    thickness
                                    + shelf_thickness
                                    + i * (door_height / (num_door_shelves))
                                    + door_holder_height,
                                ],
                            ),
                            f"door_holder_1_{i}": (
                                [shelf_thickness, door_shelf_depth, shelf_thickness],
                                [
                                    -width + thickness if handle_left else width - thickness,
                                    thickness / 2.0,
                                    thickness
                                    + shelf_thickness
                                    + i * (door_height / (num_door_shelves))
                                    + door_holder_height,
                                ],
                            ),
                            f"door_holder_2_{i}": (
                                [shelf_thickness, door_shelf_depth, shelf_thickness],
                                [
                                    -thickness if handle_left else thickness,
                                    thickness / 2.0,
                                    thickness
                                    + shelf_thickness
                                    + i * (door_height / (num_door_shelves))
                                    + door_holder_height,
                                ],
                            ),
                        }
                    )

                for k in door_shelf_boxes:
                    model.links[-1].visuals.append(
                        yourdfpy.Visual(
                            name=k,
                            origin=tra.translation_matrix(door_shelf_boxes[k][1]),
                            geometry=yourdfpy.Geometry(
                                box=yourdfpy.Box(size=door_shelf_boxes[k][0])
                            ),
                        ),
                    )
                    model.links[-1].collisions.append(
                        yourdfpy.Collision(
                            name=k,
                            origin=tra.translation_matrix(door_shelf_boxes[k][1]),
                            geometry=yourdfpy.Geometry(
                                box=yourdfpy.Box(size=door_shelf_boxes[k][0])
                            ),
                        ),
                    )

        model.joints.append(
            yourdfpy.Joint(
                name="door_joint",
                type="revolute",
                origin=joint_origin,
                axis=joint_axis,
                limit=yourdfpy.Limit(effort=100.0, velocity=100.0, lower=0, upper=np.pi),
                parent="corpus",
                child="door",
            )
        )

        if "freezer_separator" in boxes:
            freezer_door_transform = np.eye(4)
            if door_shape_args is None:
                freezer_door_transform = tra.translation_matrix(
                    [
                        0,
                        thickness / 2.0,
                        (freezer_compartment_height - thickness / 4.0) / 2.0,
                    ]
                )
                freezer_door_panel_geometry = yourdfpy.Geometry(
                                box=yourdfpy.Box([width, thickness, freezer_compartment_height - thickness / 4.0])
                            )
            else:
                coords = _create_path(
                    width=width,
                    depth=thickness,
                    straight_ratio=door_shape_args["straight_ratio"],
                    curvature_ratio=door_shape_args["curvature_ratio"],
                    num_segments=door_shape_args["num_segments"],
                )
                poly = trimesh.path.polygons.Polygon(coords)
                freezer_door_mesh = trimesh.creation.extrude_polygon(
                    polygon=poly,
                    height=freezer_compartment_height - thickness / 4.0,
                    # See https://github.com/mikedh/trimesh/pull/1683
                    engine="triangle",
                )
                freezer_door_mesh_fname = utils.get_random_filename(
                    dir=door_shape_args["tmp_mesh_dir"], prefix="freezer_door_", suffix=".obj"
                )
                freezer_door_mesh.export(freezer_door_mesh_fname)
                freezer_door_panel_geometry = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=freezer_door_mesh_fname))

            freezer_handle_thickness = handle_thickness
            freezer_handle_depth = handle_depth
            if handle_vertical:
                freezer_handle_length = min(
                    handle_length, freezer_compartment_height - 2.0 * handle_offset[1]
                )
            else:
                freezer_handle_length = min(handle_length, width - 2.0 * handle_offset[0])

            if handle_shape_args is None:
                freezer_handle_geometry = yourdfpy.Geometry(box=yourdfpy.Box([freezer_handle_length, freezer_handle_thickness, freezer_handle_depth]))
            else:
                freezer_handle_mesh_fname = utils.get_random_filename(
                    dir=handle_shape_args["tmp_mesh_dir"], prefix="freezer_handle_", suffix=".obj"
                )
                freezer_handle_mesh = _create_handle(
                    width=freezer_handle_length,
                    depth=freezer_handle_thickness,
                    height=freezer_handle_depth,
                    straight_ratio=handle_shape_args["straight_ratio"],
                    curvature_ratio=handle_shape_args["curvature_ratio"],
                    num_segments_cross_section=handle_shape_args["num_segments_cross_section"],
                    num_segments_curvature=handle_shape_args["num_segments_curvature"],
                    aspect_ratio_cross_section=handle_shape_args["aspect_ratio_cross_section"],
                )
                freezer_handle_mesh.export(freezer_handle_mesh_fname)
                freezer_handle_geometry = yourdfpy.Geometry(
                                mesh=yourdfpy.Mesh(filename=freezer_handle_mesh_fname)
                            )

            if handle_left:
                freezer_door_transform = freezer_door_transform @ tra.translation_matrix(
                    [
                        -width / 2.0,
                        -thickness,
                        height - freezer_compartment_height + thickness / 4.0,
                    ]
                )
                if handle_vertical:
                    freezer_handle_transform = (
                        tra.translation_matrix(
                            [
                                -width + freezer_handle_thickness / 2.0 + handle_offset[0],
                                -thickness - freezer_handle_depth + handle_offset[2],
                                height
                                - freezer_compartment_height
                                + freezer_handle_length / 2.0
                                + handle_offset[1],
                            ]
                        )
                        @ tra.euler_matrix(0, np.pi / 2.0, 0)
                    )
                else:
                    freezer_handle_transform = tra.translation_matrix(
                        [
                            -width + freezer_handle_length / 2.0 + handle_offset[0],
                            -thickness - freezer_handle_depth + handle_offset[2],
                            height
                            - freezer_compartment_height
                            + freezer_handle_thickness / 2.0
                            + handle_offset[1],
                        ]
                    )
                freezer_joint_origin = tra.translation_matrix(
                    [width / 2.0, -depth / 2.0 + thickness, 0.0]
                )
                freezer_joint_axis = np.array([0, 0, 1.0])
            else:
                freezer_door_transform = freezer_door_transform @ tra.translation_matrix(
                    [width / 2.0, -thickness, height - freezer_compartment_height + thickness / 4.0]
                )
                if handle_vertical:
                    freezer_handle_transform = (
                        tra.translation_matrix(
                            [
                                width - freezer_handle_thickness / 2.0 - handle_offset[0],
                                -thickness - freezer_handle_depth + handle_offset[2],
                                height
                                - freezer_compartment_height
                                + freezer_handle_length / 2.0
                                + handle_offset[1],
                            ]
                        )
                        @ tra.euler_matrix(0, np.pi / 2.0, 0)
                    )
                else:
                    freezer_handle_transform = tra.translation_matrix(
                        [
                            width - freezer_handle_length / 2.0 - handle_offset[0],
                            -thickness - freezer_handle_depth + handle_offset[2],
                            height
                            - freezer_compartment_height
                            + freezer_handle_thickness / 2.0
                            + handle_offset[1],
                        ]
                    )

                freezer_joint_origin = tra.translation_matrix(
                    [-width / 2.0, -depth / 2.0 + thickness, 0.0]
                )
                freezer_joint_axis = -np.array([0, 0, 1.0])

            model.links.append(
                yourdfpy.Link(
                    name="freezer_door",
                    visuals=[
                        yourdfpy.Visual(
                            name="freezer_door_panel",
                            origin=freezer_door_transform,
                            geometry=freezer_door_panel_geometry,
                        ),
                        yourdfpy.Visual(
                            name="freezer_door_handle",
                            origin=freezer_handle_transform,
                            geometry=freezer_handle_geometry,
                        ),
                    ],
                    collisions=[
                        yourdfpy.Collision(
                            name="freezer_door_panel",
                            origin=freezer_door_transform,
                            geometry=freezer_door_panel_geometry,
                        ),
                        yourdfpy.Collision(
                            name="freezer_door_handle",
                            origin=freezer_handle_transform,
                            geometry=freezer_handle_geometry,
                        ),
                    ],
                )
            )

            model.joints.append(
                yourdfpy.Joint(
                    name="freezer_door_joint",
                    type="revolute",
                    origin=freezer_joint_origin,
                    axis=freezer_joint_axis,
                    limit=yourdfpy.Limit(effort=100.0, velocity=100.0, lower=0, upper=np.pi),
                    parent="corpus",
                    child="freezer_door",
                )
            )

        return model

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.76 - 0.1, 0.76 + 0.1))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.76 - 0.05, 0.76 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(1.64 - 0.3, 1.64 + 0.2))
        kwargs["handle_left"] = kwargs.get("handle_left", rng.choice([True, False]))

        fridge = cls(**kwargs)

        return fridge


class MicrowaveAsset(URDFAsset):
    """A Microwave asset."""

    def __init__(
        self,
        width=0.58,
        depth=0.39,
        height=0.35,
        thickness=0.04,
        display_panel_width=0.12,
        handle_left=False,
        handle_depth=0.07,
        handle_thickness=0.05,
        handle_length=None,
        handle_straight_ratio=0.4,
        handle_curvature_ratio=1.0,
        handle_num_segments_cross_section=8,
        handle_num_segments_curvature=8,
        handle_aspect_ratio_cross_section=0.5,
        tmp_mesh_dir="/tmp",
        **kwargs,
    ):
        """A microwave oven asset.

        .. image:: /../imgs/microwave_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of microwave.
            depth (float): Depth of microwave.
            height (float): Height of microwave.
            thickness (float): Thickness of walls and door.
            display_panel_width (float): Width of the side panel. Must be smaller than width.
            handle_left (bool, optional): Microwave door opens to the left or right. Defaults to True.
            handle_depth (float, optional): Depth of handle.
            handle_thickness (float, optional): Thickness of handle.
            handle_length (float, optional): Length of handle. Defaults to None which means height - handle_depth.
            handle_straight_ratio (float, optional): Defaults to 0.4.
            handle_num_segments_cross_section (float, optional): Defaults to 8.
            handle_num_segments_curvature (float, optional): Defaults to 5.
            handle_aspect_ratio_cross_section (float, optional): Defaults to 0.5.
            tmp_mesh_dir (str, optional): Directory to generate mesh. Defaults to /tmp.
            **kwargs: Arguments will be delegated to constructor of URDFAsset.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=self._create_yourdfpy_model(
                width=width,
                depth=depth,
                height=height,
                thickness=thickness,
                display_panel_width=display_panel_width,
                handle_left=handle_left,
                handle_depth=handle_depth,
                handle_thickness=handle_thickness,
                handle_length=handle_length,
                handle_straight_ratio=handle_straight_ratio,
                handle_curvature_ratio=handle_curvature_ratio,
                handle_num_segments_cross_section=handle_num_segments_cross_section,
                handle_num_segments_curvature=handle_num_segments_curvature,
                handle_aspect_ratio_cross_section=handle_aspect_ratio_cross_section,
                tmp_mesh_dir=tmp_mesh_dir,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    def _create_yourdfpy_model(
        self,
        width,
        depth,
        height,
        thickness,
        display_panel_width,
        handle_left,
        handle_depth,
        handle_thickness,
        handle_length,
        handle_straight_ratio,
        handle_curvature_ratio,
        handle_num_segments_cross_section,
        handle_num_segments_curvature,
        handle_aspect_ratio_cross_section,
        tmp_mesh_dir,
    ):
        if display_panel_width > width:
            raise ValueError(
                f"display_panel_width={display_panel_width} needs to be less than the total"
                f" width={width} of the microwave."
            )

        model = yourdfpy.Robot(name="microwave")

        if handle_left:
            boxes = {
                "top": (
                    [width - display_panel_width, depth - 2.0 * thickness, thickness],
                    [display_panel_width / 2.0, 0, height - thickness / 2.0],
                ),
                "bottom": (
                    [width - thickness - display_panel_width, depth - 2.0 * thickness, thickness],
                    [display_panel_width / 2.0 - thickness / 2.0, 0, thickness / 2.0],
                ),
                "left": (
                    [display_panel_width, depth - thickness, height],
                    [
                        -width / 2.0 + display_panel_width / 2.0,
                        -thickness / 2.0,
                        height / 2.0,
                    ],
                ),
                "right": (
                    [thickness, depth - 2.0 * thickness, height - thickness],
                    [width / 2.0 - thickness / 2.0, 0, height / 2.0 - thickness / 2.0],
                ),
            }
        else:
            boxes = {
                "top": (
                    [width - display_panel_width, depth - 2.0 * thickness, thickness],
                    [-display_panel_width / 2.0, 0, height - thickness / 2.0],
                ),
                "bottom": (
                    [width - thickness - display_panel_width, depth - 2.0 * thickness, thickness],
                    [-display_panel_width / 2.0 + thickness / 2.0, 0, thickness / 2.0],
                ),
                "left": (
                    [thickness, depth - 2.0 * thickness, height - thickness],
                    [-width / 2.0 + thickness / 2.0, 0, height / 2.0 - thickness / 2.0],
                ),
                "right": (
                    [display_panel_width, depth - thickness, height],
                    [
                        width / 2.0 - display_panel_width / 2.0,
                        -thickness / 2.0,
                        height / 2.0,
                    ],
                ),
            }

        boxes["back"] = (
            [width, thickness, height],
            [0, depth / 2.0 - thickness / 2.0, height / 2.0],
        )

        model.links.append(
            yourdfpy.Link(
                name="corpus",
                visuals=[
                    yourdfpy.Visual(
                        name=key,
                        origin=tra.translation_matrix(boxes[key][1]),
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=boxes[key][0])),
                    )
                    for key in boxes
                ],
                collisions=[
                    yourdfpy.Collision(
                        name=key,
                        origin=tra.translation_matrix(boxes[key][1]),
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=boxes[key][0])),
                    )
                    for key in boxes
                ],
            )
        )

        handle_mesh_fname = utils.get_random_filename(
            dir=tmp_mesh_dir, prefix="microwave_handle_", suffix=".obj"
        )
        if handle_length is None:
            handle_length = height - handle_depth
        handle_mesh = _create_handle(
            width=handle_length,
            depth=handle_thickness,
            height=handle_depth,
            straight_ratio=handle_straight_ratio,
            curvature_ratio=handle_curvature_ratio,
            num_segments_cross_section=handle_num_segments_cross_section,
            num_segments_curvature=handle_num_segments_curvature,
            aspect_ratio_cross_section=handle_aspect_ratio_cross_section,
        )
        handle_mesh.export(handle_mesh_fname)

        if handle_left:
            door_transform = tra.translation_matrix(
                [
                    -width / 2.0 + display_panel_width / 2.0,
                    -thickness / 2.0,
                    height / 2.0,
                ]
            )
            handle_transform = (
                tra.translation_matrix(
                    [
                        -width + display_panel_width + handle_thickness,
                        -thickness - handle_depth,
                        height / 2.0,
                    ]
                )
                @ tra.euler_matrix(0, np.pi / 2.0, 0)
            )

            joint_origin = tra.translation_matrix([width / 2.0, -depth / 2.0 + thickness, 0.0])
            joint_axis = np.array([0, 0, 1.0])
        else:
            door_transform = tra.translation_matrix(
                [
                    +width / 2.0 - display_panel_width / 2.0,
                    -thickness / 2.0,
                    height / 2.0,
                ]
            )
            handle_transform = (
                tra.translation_matrix(
                    [
                        width - display_panel_width - handle_thickness,
                        -thickness - handle_depth,
                        height / 2.0,
                    ]
                )
                @ tra.euler_matrix(0, np.pi / 2.0, 0)
            )

            joint_origin = tra.translation_matrix([-width / 2.0, -depth / 2.0 + thickness, 0.0])
            joint_axis = -np.array([0, 0, 1.0])

        model.links.append(
            yourdfpy.Link(
                name="door",
                visuals=[
                    yourdfpy.Visual(
                        name="door_panel",
                        origin=door_transform,
                        geometry=yourdfpy.Geometry(
                            box=yourdfpy.Box(size=[width - display_panel_width, thickness, height])
                        ),
                    ),
                    yourdfpy.Visual(
                        name="door_handle",
                        origin=handle_transform,
                        geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=handle_mesh_fname)),
                    ),
                ],
                collisions=[
                    yourdfpy.Collision(
                        name="door_panel",
                        origin=door_transform,
                        geometry=yourdfpy.Geometry(
                            box=yourdfpy.Box(size=[width - display_panel_width, thickness, height])
                        ),
                    ),
                    yourdfpy.Collision(
                        name="door_handle",
                        origin=handle_transform,
                        geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=handle_mesh_fname)),
                    ),
                ],
            )
        )

        model.joints.append(
            yourdfpy.Joint(
                name="door_joint",
                type="revolute",
                origin=joint_origin,
                axis=joint_axis,
                limit=yourdfpy.Limit(effort=100.0, velocity=100.0, lower=0, upper=np.pi),
                parent="corpus",
                child="door",
            )
        )
        return model


class RangeAsset(URDFAsset):
    """A range asset."""

    def __init__(
        self,
        width=0.76,
        depth=0.73,
        height=0.92,
        control_panel_height=0.1,
        oven_height=0.7,
        storage_height=0.2,
        stove_plate_height=0.01,
        stove_plate_radius=0.1,
        wall_thickness=0.02,
        handle_length=0.66,
        handle_thickness=0.04,
        handle_depth=0.04,
        handle_offset=0.04,
        handle_shape_args={
            "straight_ratio": 0.7,
            "curvature_ratio": 0.7,
            "num_segments_cross_section": 10,
            "num_segments_curvature": 6,
            "aspect_ratio_cross_section": 0.5,
            "tmp_mesh_dir": "/tmp",
        },
        door_shape_args={
            'use_primitives':True,
            'trim_width_ratio':0.3,
            'inner_depth_ratio':0.1,
        },
        **kwargs,
    ):
        """A range asset.

        .. image:: /../imgs/range_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): The width of the range. Defaults to 0.76.
            depth (float, optional): The depth of the range. Defaults to 0.73.
            height (float, optional): The height of the range. Defaults to 0.92.
            control_panel_height (float, optional): The height of the control panel for the stovetop. Defaults to 0.1.
            oven_height (float, optional): Height of the oven compartment. Defaults to 0.7.
            storage_height (float, optional): Height of the storage drawer at the bottom. Defaults to 0.2.
            stove_plate_height (float, optional): Stovetop burner height. Defaults to 0.01.
            stove_plate_radius (float, optional): Stovetop burner radius. Defaults to 0.1.
            wall_thickness (float, optional): Thickness of outer walls. Defaults to 0.02.
            handle_length (float, optional): Length of handles. Defaults to 0.66.
            handle_thickness (float, optional): Thickness of handles. Defaults to 0.04.
            handle_depth (float, optional): Depth of handles. Defaults to 0.04.
            handle_offset (float, optional): Offset of handles. Defaults to 0.04.
            handle_shape_args (dict, optional): Handle shape parameters. Defaults to { "straight_ratio": 0.7, "curvature_ratio": 0.7, "num_segments_cross_section": 10, "num_segments_curvature": 6, "aspect_ratio_cross_section": 0.5, "tmp_mesh_dir": "/tmp", }.
            door_shape_args (dict, optional): Defaults to {'use_primitives':True, 'trim_width_ratio':0.4, 'inner_depth_ratio':0.1}.

        Raises:
            ValueError: If handle_length is larger than width.
        """
        if handle_length > width:
            raise ValueError(
                f"Range's handle_width {handle_length} needs to be smaller than range's width"
                f" {width}."
            )

        self._init_default_attributes(**kwargs)

        compartment_mask = []
        compartment_types = []
        compartment_heights = []
        compartment_interior_masks = {}
        cnt = 0
        for type, comp_height in zip(
            ["closed", "door_top", "drawer"],
            [control_panel_height, oven_height, storage_height],
        ):
            if comp_height > 0:
                compartment_mask.append([cnt])
                compartment_types.append(type)
                compartment_heights.append(comp_height)

                if type == "door_top":
                    compartment_interior_masks[cnt] = [[0], [1]]

                cnt += 1



        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width,
            depth=depth,
            height=height,
            compartment_mask=np.array(compartment_mask),
            compartment_types=compartment_types,
            compartment_interior_masks=compartment_interior_masks,
            compartment_widths=None,
            compartment_heights=compartment_heights,
            outer_wall_thickness=wall_thickness,
            frontboard_thickness=wall_thickness,
            frontboard_overlap=0.5,
            handle_width=handle_length,
            handle_height=handle_depth,
            handle_depth=handle_thickness,
            handle_offset=(0.0, handle_offset),
            handle_shape_args=handle_shape_args,
            door_shape_args=[None, door_shape_args, None],
        )

        # Replace existing shelf with a wire mesh commonly found in ranges
        v_element = next((item for item in urdf_model.links[0].visuals if item.name == 'shelf_1_0_0'), None)
        c_element = next((item for item in urdf_model.links[0].collisions if item.name == 'shelf_1_0_0'), None)
        
        # Remember transform to add replacement at same position
        v_transform = v_element.origin
        c_transform = c_element.origin

        # Remove existing box - will throw exception if None
        urdf_model.links[0].visuals.remove(v_element)
        urdf_model.links[0].collisions.remove(c_element)

        # Create wire mesh 
        wire_mesh_gap = 0.01
        wire_mesh_height = 0.01
        wire_mesh_primitives = BinAsset.create_primitives(
            width=v_element.geometry.box.size[0] - 2.0 * wire_mesh_gap,
            depth=v_element.geometry.box.size[1] - 2.0 * wire_mesh_gap,
            height=wire_mesh_height,
            thickness=0.005,
            angle=0.0,
            wired=True,
        )

        # Add wire mesh elements to first link at same position as previous one
        for i, b in enumerate(wire_mesh_primitives):
            urdf_model.links[0].visuals.append(
                yourdfpy.Visual(name=f"shelf_1_0_0_{i}", geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=b.extents)), origin=v_transform @ np.array(b.transform))
            )
            urdf_model.links[0].collisions.append(
                yourdfpy.Visual(name=f"shelf_1_0_0_{i}", geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=b.extents)), origin=c_transform @ np.array(b.transform))
            )

        # Add stove plates
        if stove_plate_height > 0 and stove_plate_radius > 0:
            for x, y in zip([0, 1, 0, 1], [0, 0, 1, 1]):
                origin = tra.translation_matrix(
                    [
                        -width / 4.0 + x * width / 2.0,
                        -depth / 4.0 + y * depth / 2.0,
                        height + stove_plate_height / 2.0,
                    ]
                )
                urdf_model.links[0].visuals.append(
                    yourdfpy.Visual(
                        name=f"heater_{x}_{y}",
                        origin=origin,
                        geometry=yourdfpy.Geometry(
                            cylinder=yourdfpy.Cylinder(
                                radius=stove_plate_radius, length=stove_plate_height
                            )
                        ),
                    )
                )
                urdf_model.links[0].collisions.append(
                    yourdfpy.Collision(
                        name=f"heater_{x}_{y}",
                        origin=origin,
                        geometry=yourdfpy.Geometry(
                            cylinder=yourdfpy.Cylinder(
                                radius=stove_plate_radius, length=stove_plate_height
                            )
                        ),
                    )
                )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.76 - 0.1, 0.76 + 0.1))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.73 - 0.05, 0.73 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(0.92 - 0.1, 0.92 + 0.1))

        kwargs["handle_length"] = kwargs.get(
            "handle_length", rng.uniform(kwargs["width"] - 0.1, kwargs["width"] - 0.001)
        )

        oven = cls(**kwargs)

        return oven


class DishwasherAsset(URDFAsset):
    """A dishwasher asset."""

    def __init__(
        self,
        width=0.76,
        depth=0.73,
        height=0.92,
        control_panel_height=0.1,
        foot_panel_height=0.04,
        wall_thickness=0.02,
        door_thickness=0.05,
        basket_height=0.15,
        basket_z_offset=(-0.10, 0.01),
        basket_wire_mesh_gap=0.05,
        handle_length=0.66,
        handle_thickness=0.05,
        handle_depth=0.04,
        handle_offset=0.04,
        handle_shape_args={
            "straight_ratio": 0.7,
            "curvature_ratio": 0.7,
            "num_segments_cross_section": 10,
            "num_segments_curvature": 6,
            "aspect_ratio_cross_section": 0.5,
            "tmp_mesh_dir": "/tmp",
        },
        **kwargs,
    ):
        """A dishwasher.

        .. image:: /../imgs/dishwasher_asset.png
            :align: center
            :width: 250px
        
        Args:
            width (float, optional): Width of dishwasher. Defaults to 0.76.
            depth (float, optional): Depth of dishwasher. Defaults to 0.73.
            height (float, optional): Height of dishwasher. Defaults to 0.92.
            control_panel_height (float, optional): Height of control panel. Defaults to 0.1.
            foot_panel_height (float, optional): Height of base / foot panel. Defaults to 0.04.
            wall_thickness (float, optional): Thickness of outer walls. Defaults to 0.02.
            door_thickness (float, optional): Thickness of front door. Defaults to 0.05.
            basket_height (float, optional): Height of each of two internal wire baskets. Defaults to 0.2.
            basket_z_offset (tuple[float, float], optional): Offset in z direction of top and bottom basket. Defaults to ().
            basket_wire_mesh_gap (float, optional): How dense the wire basket mesh is. Defaults to 0.05.
            handle_length (float, optional): Length of handle. Defaults to 0.66.
            handle_thickness (float, optional): Thickness of handle. Defaults to 0.05.
            handle_depth (float, optional): Depth of handle. Defaults to 0.04.
            handle_offset (float, optional): Offset of handle. Defaults to 0.04.
            handle_shape_args (dict, optional): Handle shape parameters. Defaults to { "straight_ratio": 0.7, "curvature_ratio": 0.7, "num_segments_cross_section": 10, "num_segments_curvature": 6, "aspect_ratio_cross_section": 0.5, "tmp_mesh_dir": "/tmp", }.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.

        Raises:
            ValueError: If handle_width is bigger than width.
        """
        if handle_length > width:
            raise ValueError(
                f"Dishwasher handle_width {handle_length} needs to be smaller than dishwasher's"
                f" width {width}."
            )

        self._init_default_attributes(**kwargs)

        compartment_mask = []
        compartment_types = []
        compartment_heights = []
        cnt = 0
        for type, comp_height in zip(
            ["closed", "door_top", "closed"],
            [
                control_panel_height,
                1.0 - control_panel_height - foot_panel_height,
                foot_panel_height,
            ],
        ):
            if comp_height > 0:
                compartment_mask.append([cnt])
                compartment_types.append(type)
                compartment_heights.append(comp_height)
                cnt += 1
        
        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width,
            depth=depth,
            height=height,
            compartment_mask=np.array(compartment_mask),
            compartment_types=compartment_types,
            compartment_widths=None,
            compartment_heights=compartment_heights,
            outer_wall_thickness=wall_thickness,
            inner_wall_thickness=wall_thickness,
            frontboard_thickness=door_thickness,
            frontboard_overlap=0.9,
            handle_width=handle_length,
            handle_height=handle_depth,
            handle_depth=handle_thickness,
            handle_offset=(0.0, handle_offset),
            handle_shape_args=handle_shape_args,
        )

        # insert two bins
        basket_gap = 0.01
        basket_names = ["top_basket", "bottom_basket"]
        basket_zs = [(1.0 - control_panel_height) * height - 2.0 * wall_thickness - basket_height + basket_z_offset[0], foot_panel_height * height + 1.0 * wall_thickness + door_thickness + basket_z_offset[1]]
        basket_y = (door_thickness - wall_thickness) / 2.0
        for basket_name, basket_z in zip(basket_names, basket_zs):
            basket_top_primitives = BinAsset.create_primitives(
                width=width - wall_thickness - wall_thickness - 2.0 * basket_gap,
                depth=depth - wall_thickness - door_thickness - 2.0 * basket_gap,
                height=basket_height,
                thickness=0.005,
                angle=0.0,
                wired=True,
                wire_gap=basket_wire_mesh_gap,
            )
            urdf_model.links.append(yourdfpy.Link(
                name=basket_name,
                visuals=[yourdfpy.Visual(name=f"{basket_name}_{i}", geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=b.extents)), origin=np.array(b.transform)) for i, b in enumerate(basket_top_primitives)],
                collisions=[yourdfpy.Collision(name=f"{basket_name}_collision_{i}",geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=b.extents)), origin=np.array(b.transform)) for i, b in enumerate(basket_top_primitives)],
            ))
            urdf_model.joints.append(yourdfpy.Joint(name=f"corpse_to_{basket_name}", type="prismatic", origin=tra.translation_matrix((0, basket_y, basket_z)), parent="corpus", child=basket_name, axis=np.array((0, -1.0, 0)), limit=yourdfpy.Limit(
                    effort=1000.0,
                    velocity=1.0,
                    lower=0.0,
                    upper=depth - 0.1,)
            ))
        
        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.7 - 0.1, 0.7 + 0.01))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.73 - 0.05, 0.73 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(0.92 - 0.1, 0.92 + 0.1))

        kwargs["handle_length"] = kwargs.get(
            "handle_length", rng.uniform(kwargs["width"] - 0.1, kwargs["width"] - 0.001)
        )

        dishwasher = cls(**kwargs)

        return dishwasher


class SinkCabinetAsset(URDFAsset):
    """A sink cabinet asset."""

    def __init__(
        self,
        width=0.76,
        depth=0.73,
        height=0.92,
        sink_width=0.5,
        sink_depth=0.4,
        sink_height=0.2,
        sink_offset=(0, -0.1),
        sink_thickness=0.02,
        sink_tmp_mesh_dir="/tmp",
        use_primitives_for_sink=False,
        countertop_thickness=0.03,
        wall_thickness=0.02,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """A cabinet with two doors and a sink on top.

        .. image:: /../imgs/sink_cabinet_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the sink cabinet. Defaults to 0.76.
            depth (float, optional): Depth of the sink cabinet. Defaults to 0.73.
            height (float, optional): Height of the sink cabinet. Defaults to 0.92.
            sink_width (float, optional): Width of the sink. Defaults to 0.4.
            sink_depth (float, optional): Depth of the sink. Defaults to 0.3.
            sink_height (float, optional): Height of the sink. Defaults to 0.2.
            sink_offset (tuple, optional): Offset of the sink. Defaults to (0, 0).
            sink_thickness (float, optional): Thickness of the sink. Defaults to 0.02.
            sink_tmp_mesh_dir (str, optional): Directory for storing (temporary) mesh files. Defaults to "/tmp".
            countertop_thickness (float, optional): Thickness of countertop. Defaults to 0.03.
            wall_thickness (float, optional): Thickness of outer walls. Defaults to 0.02.
            handle_width (float, optional): Width of handles. Defaults to 0.1682.
            handle_height (float, optional): Heights of handles. Defaults to 0.038.
            handle_depth (float, optional): Depth of handles. Defaults to 0.024.
            handle_offset (tuple[float, float], optional): Offset of handles. Defaults to None.
            handle_shape_args (dict, optional): Handle shape parameters. Defaults to None.
            door_shape_args (dict, optional): Cabinet door shape parameters. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.

        Raises:
            ValueError: If sink_height is bigger than cabinet height.
        """
        for name, sinkdim, cabinetdim in zip(('width', 'height', 'depth'), (sink_width, sink_height, sink_depth), (width, height, depth)):
            if cabinetdim < sinkdim:
                raise ValueError(
                    f"sink_{name} {sinkdim} must be smaller than cabinet {name} {cabinetdim} for"
                    " SinkCabinetAsset."
                )

        self._init_default_attributes(**kwargs)

        compartment_mask = np.array([[0, 0], [1, 2]])
        compartment_types = ["closed", "door_right", "door_left"]
        compartment_heights = [sink_height, height - sink_height]

        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width,
            depth=depth,
            height=height - (countertop_thickness - wall_thickness),
            compartment_mask=np.array(compartment_mask),
            compartment_types=compartment_types,
            compartment_widths=None,
            compartment_heights=compartment_heights,
            outer_wall_thickness=wall_thickness,
            handle_width=handle_width,
            handle_height=handle_height,
            handle_depth=handle_depth,
            handle_offset=handle_offset,
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
        )

        # remove top box geometry
        assert (
            urdf_model.links[0].visuals[0].name == urdf_model.links[0].collisions[0].name == "top"
        )
        del urdf_model.links[0].visuals[0]
        del urdf_model.links[0].collisions[0]
        
        sink_visuals = []
        sink_collisions = []
        countertop_origin = tra.translation_matrix([0, 0, height - countertop_thickness / 2.0])
        sink_origin = tra.translation_matrix([0, 0, height - countertop_thickness / 2.0])
        sink_local_translation = [sink_offset[0], sink_offset[1], -sink_height + countertop_thickness / 2.0]
        if use_primitives_for_sink:
            # Add countertop around sink
            countertop_primitives = BoxWithHoleAsset.create_primitives(
                width=width,
                depth=depth,
                height=countertop_thickness,
                hole_width=sink_width,
                hole_depth=sink_depth,
                hole_offset=sink_offset,
            )
            for i, box in enumerate(countertop_primitives):
                sink_visuals.append(
                    yourdfpy.Visual(
                        name=f"countertop_sink_{i}",
                        origin=sink_origin @ box.primitive.transform,
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(box.primitive.extents)),
                    ),
                )
                sink_collisions.append(
                    yourdfpy.Collision(
                        name=f"countertop_sink_{i}",
                        origin=sink_origin @ box.primitive.transform,
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(box.primitive.extents)),
                    ),
                )

            # Add sink itself
            sink_primitives = BinAsset.create_primitives(
                width=sink_width + sink_thickness * 2.0,
                depth=sink_depth + sink_thickness * 2.0,
                height=sink_height - countertop_thickness,
                thickness=sink_thickness,
            )
            sink_local_transform = tra.translation_matrix(sink_local_translation)
            for i, box in enumerate(sink_primitives):
                sink_visuals.append(
                    yourdfpy.Visual(
                        name=f"sink_{i}",
                        origin=sink_origin @ sink_local_transform @ box.primitive.transform,
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(box.primitive.extents)),
                    ),
                )
                sink_collisions.append(
                    yourdfpy.Collision(
                        name=f"sink_{i}",
                        origin=sink_origin @ sink_local_transform @ box.primitive.transform,
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(box.primitive.extents)),
                    ),
                )
        else:
            # create mesh for countertop (with hole for sink)
            countertop_mesh = BoxWithHoleAsset.create_mesh(
                width=width,
                depth=depth,
                height=countertop_thickness,
                hole_width=sink_width,
                hole_depth=sink_depth,
                hole_offset=sink_offset,
            )
            countertop_mesh_fname = utils.get_random_filename(
                dir=sink_tmp_mesh_dir,
                prefix="sink_countertop_",
                suffix=".obj",
            )
            countertop_mesh.export(countertop_mesh_fname)

            # create mesh for sink
            sink_mesh = BinAsset.create_mesh(
                width=sink_width + sink_thickness * 2.0,
                depth=sink_depth + sink_thickness * 2.0,
                height=sink_height - countertop_thickness,
                thickness=sink_thickness,
            )
            sink_mesh.apply_translation(sink_local_translation)
            sink_mesh_fname = utils.get_random_filename(
                dir=sink_tmp_mesh_dir,
                prefix="sink_",
                suffix=".obj",
            )
            sink_mesh.export(sink_mesh_fname)

            sink_visuals = [
                yourdfpy.Visual(
                    name=f"sink_countertop",
                    origin=countertop_origin,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=countertop_mesh_fname)),
                ),
                yourdfpy.Visual(
                    name=f"sink",
                    origin=sink_origin,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=sink_mesh_fname)),
                ),
            ]
            sink_collisions = [
                yourdfpy.Collision(
                    name=f"sink_countertop",
                    origin=countertop_origin,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=countertop_mesh_fname)),
                ),
                yourdfpy.Collision(
                    name=f"sink",
                    origin=sink_origin,
                    geometry=yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=sink_mesh_fname)),
                ),
            ]


        # add top geometry
        urdf_model.links[0].visuals.extend(
            sink_visuals
        )
        urdf_model.links[0].collisions.extend(
            sink_collisions
        )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.76 - 0.1, 0.76 + 0.1))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.73 - 0.05, 0.73 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(0.92 - 0.1, 0.92 + 0.1))

        kwargs["sink_width"] = kwargs.get("sink_width", kwargs['width'] * 0.7)
        kwargs["sink_depth"] = kwargs.get("sink_depth", kwargs['depth'] * 0.65)
        kwargs["sink_height"] = kwargs.get("sink_height", kwargs['height'] * 0.2)

        kwargs["sink_offset"] = kwargs.get("sink_offset", (0, 0.06-(kwargs['depth']-kwargs['sink_depth'])/2.0))

        if "handle_shape_args" not in kwargs:
            kwargs["handle_shape_args"] = HandleAsset.random_shape_params(seed=seed)

        if "door_shape_args" not in kwargs:
            kwargs["door_shape_args"] = CabinetDoorAsset.random_shape_params(seed=seed)

        sink_cabinet = cls(**kwargs)

        return sink_cabinet


class BaseCabinetAsset(URDFAsset):
    """A base cabinet asset."""

    def __init__(
        self,
        width=0.76,
        depth=0.73,
        height=0.92,
        num_drawers_horizontal=None,
        num_drawers_vertical=1,
        lower_compartment_types=("door_right", "door_left"),
        num_shelves=1,
        drawer_height=0.2,
        foot_panel_height=0.04,
        frontboard_thickness=0.02,
        wall_thickness=0.02,
        inner_wall_thickness=0.01,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """A base kitchen cabinet. A specialization of the CabinetAsset. It has optional drawers on top of a row of optional doors and a small foot panel.

        .. image:: /../imgs/base_cabinet_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the cabinet. Defaults to 0.76.
            depth (float, optional): Depth of the cabinet. Defaults to 0.73.
            height (float, optional): Height of the cabinet. Defaults to 0.92.
            num_drawers_horizontal (int, optional): Number of drawers next to each other. None equals number of doors below. Defaults to None.
            num_drawers_vertical (int, optional): Number of drawers in the vertical direction. Defaults to 1.
            lower_compartment_types (tuple, optional): Tuple of door types next to each other. Defaults to ("door_right", "door_left").
            num_shelves (int, optional): Number of shelves inside lower compartment. Defaults to 1.
            drawer_height (float, optional): Height of the drawers. Defaults to 0.2.
            foot_panel_height (float, optional): Height of the foot panel. Defaults to 0.04.
            frontboard_thickness (float, optional): Thickness of the frontboards/doors. Defaults to 0.02.
            wall_thickness (float, optional): Thickness of the exterior walls. Defaults to 0.02.
            inner_wall_thickness (float, optional): Thickness of the interior walls. Defaults to 0.01.
            handle_width (float, optional): Width of handle. Defaults to 0.1682.
            handle_height (float, optional): Height of handle. Defaults to 0.038.
            handle_depth (float, optional): Depth of handle. Defaults to 0.024.
            handle_offset (tuple[float, float], optional): Offset of handle. Defaults to None.
            handle_shape_args (dict, optional): Arguments for procedural handle shape generator. Defaults to None.
            door_shape_args (dict, optional): Arguments for procedural door shape generator. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        if num_drawers_horizontal is None:
            num_drawers_horizontal = max(1, len(lower_compartment_types))

        if (num_drawers_horizontal * num_drawers_vertical) == 0:
            # No drawers, just door(s)
            compartment_mask = np.array(
            [
                [i for i in range(len(lower_compartment_types))],
                [len(lower_compartment_types)] * len(lower_compartment_types),
            ]
            )
            compartment_types = (
                list(lower_compartment_types) + ["closed"]
            )
            compartment_heights = [
                height - foot_panel_height,
                foot_panel_height,
            ]
        elif len(lower_compartment_types) == 0:
            # No doors, just drawer(s)
            compartment_mask = np.array(
                [
                    list(range(num_drawers_horizontal * i, num_drawers_horizontal * (i+1))) for i in range(num_drawers_vertical)
                ] +
                [
                    [(num_drawers_vertical * num_drawers_horizontal)] * num_drawers_horizontal,
                ]
            )
            compartment_types = (
                ["drawer"] * (num_drawers_vertical * num_drawers_horizontal) + ["closed"]
            )
            compartment_heights = [
                (height - foot_panel_height) / num_drawers_vertical for _ in range(num_drawers_vertical)
            ] + [
                foot_panel_height
            ]
        else:
            # Drawer(s) top, door(s) bottom
            compartment_mask = [
                    list(range(num_drawers_horizontal * i, num_drawers_horizontal * (i+1))) for i in range(num_drawers_vertical)
                ] + [
                    [(num_drawers_vertical * num_drawers_horizontal) + i for i in range(len(lower_compartment_types))],
                    [(num_drawers_vertical * num_drawers_horizontal) + len(lower_compartment_types)] * len(lower_compartment_types),
                ]
            compartment_mask = np.array(self._make_segmentation_matrix_shape_homogenous(compartment_mask))

            compartment_types = (
                ["drawer"] * (num_drawers_vertical * num_drawers_horizontal) + list(lower_compartment_types) + ["closed"]
            )
            compartment_heights = [
                drawer_height for _ in range(num_drawers_vertical)
            ] + [
                height - drawer_height * num_drawers_vertical - foot_panel_height,
                foot_panel_height,
            ]

        compartment_interior_masks = {i: [[j] for j in range(num_shelves + 1)] for i in range(len(compartment_types))}

        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width,
            depth=depth,
            height=height,
            compartment_mask=compartment_mask,
            compartment_types=compartment_types,
            compartment_widths=None,
            compartment_heights=compartment_heights,
            compartment_interior_masks=compartment_interior_masks,
            outer_wall_thickness=wall_thickness,
            inner_wall_thickness=inner_wall_thickness,
            frontboard_thickness=frontboard_thickness,
            frontboard_overlap=0.9,
            handle_width=handle_width,
            handle_height=handle_height,
            handle_depth=handle_depth,
            handle_offset=handle_offset,
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
        )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    def _make_segmentation_matrix_shape_homogenous(self, x):
        lcm = np.lcm.reduce([len(row) for row in x])
        y = []
        for row in x:
            alpha = lcm // len(row)
            y_row = [val for val in row for _ in range(alpha)]
            y.append(y_row)
        return y

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.7 - 0.1, 0.7 + 0.01))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.73 - 0.05, 0.73 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(0.92 - 0.1, 0.92 + 0.1))

        if "handle_shape_args" not in kwargs:
            kwargs["handle_shape_args"] = HandleAsset.random_shape_params(seed=seed)

        # randomize number of doors
        num_doors = kwargs.get("num_doors", rng.choice([0, int(np.ceil(kwargs["width"] / rng.uniform(0.35, 0.5)))], p=(0.2, 0.8)))
        kwargs["lower_compartment_types"] = kwargs.get("lower_compartment_types", rng.choice(["door_right", "door_left"], num_doors))

        # randomize number of drawers
        kwargs["num_drawers_horizontal"] = kwargs.get("num_drawers_horizontal", rng.choice([None, 1] if num_doors > 0 else [1]))
        kwargs["num_drawers_vertical"] = kwargs.get("num_drawers_vertical", rng.choice([0, 1, 2] if num_doors > 0 else [1, 2, 3]))

        kwargs["num_shelves"] = kwargs.get("num_shelves", rng.integers(1, 3))

        base_cabinet = cls(
            **kwargs,
        )

        return base_cabinet


class BinAsset(TrimeshSceneAsset):
    """A bin asset."""

    def __init__(self, width=0.5,
                 depth=0.38,
                 height=0.12,
                 thickness=0.005,
                 angle=0.0,
                 wired=False,
                 use_primitives=False,
                 **kwargs):
        """A storage bin asset (rectangular bottom and four rectangular sides).
        Optionally, the sides can be angled and the surfaces can be wire meshes.

        .. image:: /../imgs/bin_asset.png
            :align: center
            :width: 250px

        .. image:: /../imgs/bin_asset_2.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of storage bin. Defaults to 0.5.
            depth (float, optional): Depth of storage bin. Defaults to 0.12.
            height (float, optional): Height of storage bin. Defaults to 0.12.
            thickness (float, optional): Thickness of bottom and walls. Defaults to 0.005.
            angle (float, optional): Angle in radians to create slanted side walls. Needs to be in (-pi/2, +pi/2). Positive means outward slope. Defaults to 0.
            wired (bool, optional): Whether to create a wire basket. Defaults to False.
            use_primitives (bool, optional): Will use five box primitives to construct bin. Note, that angle will be ignored. Defaults to False.
            **kwargs: Arguments will be delegated to constructor of TrimeshAsset.
        """
        fn = BinAsset.create_primitives if use_primitives else BinAsset.create_mesh
        
        if wired:
            fn = partial(BinAsset.create_primitives, wired=True)

        super().__init__(
            trimesh.Scene(
                fn(
                    width=width,
                    depth=depth,
                    height=height,
                    thickness=thickness,
                    angle=angle,
                ),
            ),
            **kwargs,
        )

    @staticmethod
    def create_primitives(width, depth, height, thickness, angle=0, wired=False, wire_gap=0.05, wire_thickness=0.005):
        if angle != 0:
            raise Warning(f"BinAsset: angle is ignored since use_primitives=True")
        
        if not (angle > -np.pi / 2.0 and angle < np.pi / 2.0):
            raise ValueError(f"angle={angle} must be between -np.pi / 2.0 and np.pi / 2.0 (exclusively).")

        if wired and angle != 0:
            raise NotImplementedError("BinAsset cannot be angled (angle={angle}) if wired=True.")

        outer_hyp = height / np.cos(angle)
        outer_widthdepth = np.sin(angle) * outer_hyp

        inner_width = width - 2.0 * thickness
        inner_depth = depth - 2.0 * thickness

        inner_width_bottom = inner_width + 2.0 * thickness * np.sin(angle)
        inner_depth_bottom = inner_depth + 2.0 * thickness * np.sin(angle)
        
        primitives = []
        if wired:
            wire_x = np.arange(-inner_width_bottom/2.0, inner_width_bottom/2.0, wire_gap)[1:]
            for i in range(len(wire_x)):
                # bottom ones
                primitives.append(trimesh.primitives.Box(
                    (wire_thickness, inner_depth_bottom, thickness),
                    transform=tra.translation_matrix((wire_x[i], 0, +thickness / 2.0))
                    ))
                # back
                primitives.append(trimesh.primitives.Box(
                    (wire_thickness, thickness, height),
                    transform=tra.translation_matrix((wire_x[i], +depth / 2.0 - thickness / 2.0, height / 2.0))
                    ))
                # front
                primitives.append(trimesh.primitives.Box(
                    (wire_thickness, thickness, height),
                    transform=tra.translation_matrix((wire_x[i], -depth / 2.0 + thickness / 2.0, height / 2.0))
                    ))
            wire_y = np.arange(-inner_depth_bottom/2.0, inner_depth_bottom/2.0, wire_gap)[1:]
            for i in range(len(wire_y)):
                # bottom ones
                primitives.append(trimesh.primitives.Box(
                    (inner_width_bottom, wire_thickness, thickness),
                    transform=tra.translation_matrix((0, wire_y[i], +thickness / 2.0))
                    ))
                # right
                primitives.append(trimesh.primitives.Box(
                    (thickness, wire_thickness, height),
                    transform=tra.translation_matrix((+width / 2.0 - thickness / 2.0, wire_y[i], height / 2.0))
                    ))
                # left
                primitives.append(trimesh.primitives.Box(
                    (thickness, wire_thickness, height),
                    transform=tra.translation_matrix((-width / 2.0 + thickness / 2.0, wire_y[i], height / 2.0))
                    ))
            wire_z = np.arange(height, 0, -wire_gap)
            for i in range(len(wire_z)):
                # right
                primitives.append(trimesh.primitives.Box(
                    (thickness, inner_depth_bottom + 2.0 * thickness, wire_thickness),
                    transform=tra.translation_matrix((+width / 2.0 - thickness / 2.0, 0, wire_z[i]))
                    ))
                # left
                primitives.append(trimesh.primitives.Box(
                    (thickness, inner_depth_bottom + 2.0 * thickness, wire_thickness),
                    transform=tra.translation_matrix((-width / 2.0 + thickness / 2.0, 0, wire_z[i]))
                    ))
                # back
                primitives.append(trimesh.primitives.Box(
                    (inner_width_bottom, thickness, wire_thickness),
                    transform=tra.translation_matrix((0, +depth / 2.0 - thickness / 2.0, wire_z[i]))
                    ))
                # front
                primitives.append(trimesh.primitives.Box(
                    (inner_width_bottom, thickness, wire_thickness),
                    transform=tra.translation_matrix((0, -depth / 2.0 + thickness / 2.0, wire_z[i]))
                    ))
        else:
            bin_floor = trimesh.primitives.Box((inner_width_bottom, inner_depth_bottom, thickness), transform=tra.translation_matrix((0, 0, +thickness / 2.0)))
            bin_wall_north = trimesh.primitives.Box((inner_width_bottom, thickness, height), transform=tra.translation_matrix((0, +depth / 2.0 - thickness / 2.0, height / 2.0)))
            bin_wall_south = trimesh.primitives.Box((inner_width_bottom, thickness, height), transform=tra.translation_matrix((0, -depth / 2.0 + thickness / 2.0, height / 2.0)))
            bin_wall_west = trimesh.primitives.Box((thickness, inner_depth_bottom + 2.0 * thickness, height), transform=tra.translation_matrix((+width / 2.0 - thickness / 2.0, 0, height / 2.0)))
            bin_wall_east = trimesh.primitives.Box((thickness, inner_depth_bottom + 2.0 * thickness, height), transform=tra.translation_matrix((-width / 2.0 + thickness / 2.0, 0, height / 2.0)))
            primitives = (bin_floor, bin_wall_north, bin_wall_south, bin_wall_west, bin_wall_east)

        return primitives

    @staticmethod
    def create_mesh(width, depth, height, thickness, angle=0):
        if not (angle > -np.pi / 2.0 and angle < np.pi / 2.0):
            raise ValueError(f"angle={angle} must be between -np.pi / 2.0 and np.pi / 2.0 (exclusively).")

        outer_hyp = height / np.cos(angle)
        outer_widthdepth = np.sin(angle) * outer_hyp

        inner_width = width - 2.0 * thickness
        inner_depth = depth - 2.0 * thickness

        inner_width_bottom = width - 2.0 * thickness + 2.0 * thickness * np.sin(angle)
        inner_depth_bottom = depth - 2.0 * thickness + 2.0 * thickness * np.sin(angle)

        vertices = [
            [-width / 2.0 - outer_widthdepth, -depth / 2.0 - outer_widthdepth, height],
            [width / 2.0 + outer_widthdepth, -depth / 2.0 - outer_widthdepth, height],
            [width / 2.0 + outer_widthdepth, depth / 2.0 + outer_widthdepth, height],
            [-width / 2.0 - outer_widthdepth, depth / 2.0 + outer_widthdepth, height],
            [-inner_width / 2.0 - outer_widthdepth, -inner_depth / 2.0 - outer_widthdepth, height],
            [inner_width / 2.0 + outer_widthdepth, -inner_depth / 2.0 - outer_widthdepth, height],
            [inner_width / 2.0 + outer_widthdepth, inner_depth / 2.0 + outer_widthdepth, height],
            [-inner_width / 2.0 - outer_widthdepth, inner_depth / 2.0 + outer_widthdepth, height],
            [-inner_width_bottom / 2.0, -inner_depth_bottom / 2.0, thickness],
            [inner_width_bottom / 2.0, -inner_depth_bottom / 2.0, thickness],
            [inner_width_bottom / 2.0, inner_depth_bottom / 2.0, thickness],
            [-inner_width_bottom / 2.0, inner_depth_bottom / 2.0, thickness],
            [-width / 2.0, -depth / 2.0, 0],
            [width / 2.0, -depth / 2.0, 0],
            [width / 2.0, depth / 2.0, 0],
            [-width / 2.0, depth / 2.0, 0],
        ]

        faces = [
            [0, 1, 4],
            [1, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
            [3, 0, 7],
            [0, 4, 7],
            [4, 5, 8],
            [5, 9, 8],
            [5, 6, 9],
            [6, 10, 9],
            [6, 7, 10],
            [7, 11, 10],
            [7, 4, 11],
            [4, 8, 11],
            [8, 9, 11],
            [9, 10, 11],
            [0, 12, 13],
            [1, 0, 13],
            [2, 1, 13],
            [13, 14, 2],
            [3, 2, 14],
            [15, 3, 14],
            [0, 3, 15],
            [0, 15, 12],
            [13, 12, 15],
            [15, 14, 13],
        ]

        return trimesh.Trimesh(vertices=vertices, faces=faces)


class RecursivelyPartitionedCabinetAsset(URDFAsset):
    """A Recursively partitioned cabinet asset."""

    # distance between the end of door and handle
    _DEFAULT_DOOR_HANDLE_DIST = 0.05

    def __init__(
        self,
        width,
        depth,
        height,
        thickness=0.02,
        wall_thickness=0.01,
        split_prob=1.0,
        split_decay=0.65,
        articulation_type_prob=None,
        door_left_prob=0.5,
        additional_compartment_floor_height=0.0,
        name="Cabinet",
        force_one_primitive_per_link=False,
        handle_types=["procedural"],
        handle_type_prob=[1.0],
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=None,
        knob_kwargs=None,
        handle_shape_args=None,
        seed=None,
        **kwargs,
    ):
        """Procedural cabinet generator based on recursive splitting of compartments.
        Based on: Jan Czarnowski's CabGen.

        .. image:: /../imgs/recursively_partitioned_cabinet_asset.png
            :align: center
            :width: 250px

        Args:
            width (float): Width of cabinet.
            depth (float): Depth of cabinet.
            height (float): Height of cabinet.
            thickness (float): Thickness of outer cabinet walls.
            wall_thickness (float, optional): Thickness of inner cabinet walls, between compartments. Defaults to 0.01.
            split_prob (float, optional): The probability of splitting a compartment into two. Defaults to 1.0.
            split_decay (float, optional): The decay rate of the splitting probability for each level of recursion. Defaults to 0.65.
            articulation_type_prob (list[float], optional): If not None, all compartments will be of type "drawer", "door", or "none" according to the probabilities assigned by this 3-element list/tuple. Defaults to None.
            door_left_prob (float, optional): Cabinet doors will open according to the this probability ("right" == 1-left). Defaults to 0.5.
            additional_compartment_floor_height (float, optional): Adds another box inside each empty/revolute door compartment if value is greater than 0.0 that serves as the floor of the compartment. This helps when identifying container volumes. Defaults to 0.0.
            name (str, optional): Name of cabinet. Will be used for link name. Defaults to "Cabinet".
            force_one_primitive_per_link (bool, optional): Whether to create a single link for the drawer and cabinet bodies (w/ multiple visuals/collsions). Or to create one link per visual/collision. This allows to keep information about geometric primitives. Defaults to False.
            handle_type (str, optional): Types of the handle. Defaults to procedural. Valid options "procedural", "box", "knob".
            handle_width (float, optional): Defaults to 0.1682.
            handle_depth (float, optional): Defaults to 0.038.
            handle_height (float, optional): Defaults to 0.024.
            handle_offset (float, optional): Defaults to None.
            handle_shape_args (dict, optional): Arguments for procedural handles. If None, will create handle made out of boxes. Defaults to None.
            seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self.width = width
        self.depth = depth
        self.height = height
        self.handle_width = handle_width
        self.handle_depth = handle_depth
        self.handle_height = handle_height
        self.handle_offset = handle_offset
        self.handle_shape_args = handle_shape_args
        self.knob_kwargs = knob_kwargs

        self._rng = np.random.default_rng(seed)

        if articulation_type_prob is not None:
            if len(articulation_type_prob) != 3:
                raise ValueError(
                    "articulation_type_prob probability distribution needs to have 3 elements!"
                )
            if sum(articulation_type_prob) != 1.0:
                raise ValueError(
                    "articulation_type_prob probability distribution needs to sum to 1.0!"
                )

        if len(handle_type_prob) != len(handle_types):
            raise ValueError(
                "handle_type_prob probability distribution and handle_types must to have same"
                " number of elements!"
            )

        if sum(handle_type_prob) == 0:
            raise ValueError("handle_type_prob probability distribution cannot sum to 0")

        # normalize probability
        handle_type_prob = np.array(handle_type_prob) / sum(handle_type_prob)

        for handle_type in handle_types:
            if handle_type not in ["procedural", "box", "knob"]:
                raise ValueError(
                    f"Invalide handle type `{handle_type}`. Valid options: [procedural, box, knob]"
                )

        self.num_drawers = 0
        self.num_doors = 0

        self.additional_compartment_floor_height = additional_compartment_floor_height

        self.handle_types = handle_types
        self.handle_type_prob = handle_type_prob
        current_dir = os.path.dirname(os.path.realpath(__file__))
        asset_root_dir = os.path.join(current_dir, os.pardir, os.pardir, "tests", "data", "assets")
        self.mesh_handle_fname = os.path.join(asset_root_dir, "handles/handle.obj")

        self._cabinet = yourdfpy.Robot(name=name)

        self.body_name = f"{name}_body"
        self._add_body(
            name=self.body_name,
            width=width,
            depth=depth,
            height=height,
            thickness=thickness,
            single_link=not force_one_primitive_per_link,
        )

        split_args = {
            "wt": wall_thickness,
            "prob": split_prob,
            "decay": split_decay,
            "min_width": 0.3,
            "min_height": 0.3,
            "articulation_type_prob": articulation_type_prob,
            "door_left_prob": door_left_prob,
        }

        # run recursive splits
        self._split(
            x=0,
            y=0,
            width=self.width,
            height=self.height,
            force_one_primitive_per_link=force_one_primitive_per_link,
            **split_args,
        )

        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=self._cabinet,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    def _sigmoid_pdf(self, r, k=4.5):
        return 1 / (1 + np.exp(-k * (r - 1)))

    def _articulation_type(self, r):
        """This function returns the probability of the element being a drawer given its width to height ratio r
        For ratio > 1, we want to increase that probability
        For ratio = 1, it can be 0.5

        Args:
            r (float): Aspect ratio (width/height).

        Returns:
            str: Either "drawer" or "door".
        """
        return "drawer" if self._rng.random() < self._sigmoid_pdf(r) else "door"

    def _split(
        self,
        x,
        y,
        width,
        height,
        wt,
        prob=0.8,
        decay=0.9,
        min_width=1,
        min_height=1,
        articulation_type_prob=None,
        door_left_prob=0.5,
        force_one_primitive_per_link=False,
    ):
        """Recursive cabinet splitting.

        Args:
            x (float): x-coordinate of 2D coordinates of cabinet front.
            y (float): y-coordinate of 2D coordinates of cabinet front.
            width (float): Width of 2D cabinet front.
            height (float): Height of 2D cabinet front.
            wt (float): Wall thickness.
            prob (float, optional): Probabiliy of splitting further. Defaults to 0.8.
            decay (float, optional): Decay rate of splitting probability. Defaults to 0.9.
            min_width (int, optional): Minimum width for splitting. Defaults to 1.
            min_height (int, optional): Minimum height for splitting. Defaults to 1.
            articulation_type_prob (list[float], optional): If not None, all compartments will be of type "drawer", "door", or "none"/"open" according to the probabilities assigned by this 3-element list/tuple. Defaults to None.
            door_left_prob (float, optional): Cabinet doors will open according to the this probability ("right" == 1-left). "Left" means left-handed handle position:  i.e. https://res.cloudinary.com/ecbarton/image/upload/s--k24LhiF0--/c_fill%2Cf_auto%2Ch_288%2Cq_auto%2Cw_512/v1/blog-assets/Picture1_1.jpg?itok=8b05c9Mw
            force_one_primitive_per_link (bool, optional): Whether to create a single link for the drawer body (w/ multiple visuals/collsions). Or to create one link per visual/collision. This allows to keep information about geometric primitives. Defaults to False.
        """
        # check split probability
        rand = self._rng.random()
        do_split = rand < prob

        # minimum size for splitting
        if width < min_width and height < min_height:
            do_split = False

        if not do_split:  # and size < min
            # decide on drawer or left or right door
            etype = (
                self._articulation_type(width / height)
                if articulation_type_prob is None
                else self._rng.choice(["drawer", "door", "none"], p=articulation_type_prob)
            )

            if etype == "drawer":
                self._add_drawer(
                    parent=self.body_name,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    drawer_depth=self.depth * _DRAWER_DEPTH_PERCENT,
                    single_link=not force_one_primitive_per_link,
                )
            elif etype == "door":
                self._add_door(
                    parent=self.body_name,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    left=self._rng.random() < door_left_prob,
                )
            else:
                # empty compartment
                if self.additional_compartment_floor_height > 0.0:
                    # for naming the geometry
                    num_empty_floors = len(
                        [
                            1
                            for v in self._cabinet.links[0].visuals
                            if v.name.startswith("empty_compartment_floor")
                        ]
                    )
                    # calculate floor position
                    floor_xyz = self._local_to_body(
                        x + width / 2.0,
                        y + height - self.additional_compartment_floor_height / 2.0,
                    )
                    floor_xyz[1] -= self.depth / 2.0
                    self._cabinet.links[0].visuals.append(
                        self._create_box_visual(
                            name=f"empty_compartment_floor_{num_empty_floors}",
                            origin=tra.translation_matrix(floor_xyz),
                            size=(width, self.depth, self.additional_compartment_floor_height),
                        )
                    )
                    self._cabinet.links[0].collisions.append(
                        self._create_box_visual(
                            name=f"empty_compartment_floor_{num_empty_floors}",
                            origin=tra.translation_matrix(floor_xyz),
                            size=(width, self.depth, self.additional_compartment_floor_height),
                        )
                    )

            return

        # decay the split probability
        prob_new = prob * decay

        # calculate wall position
        w_xyz = self._local_to_body(
            x + width / 2.0,
            y + height / 2.0,
        )
        w_xyz[1] -= self.depth / 2.0

        # decide on vertical/horizontal split
        vertical_split = self._rng.random() < 0.5

        # minimum size for splitting vert/horiz
        if width < min_width:
            vertical_split = False
        if height < min_height:
            vertical_split = True

        # wrap call to make sure that we pass all params to the lower level
        def run_split(x, y, width, height, wt, prob):
            self._split(
                x=x,
                y=y,
                width=width,
                height=height,
                wt=wt,
                prob=prob,
                decay=decay,
                min_width=min_width,
                min_height=min_height,
                articulation_type_prob=articulation_type_prob,
                door_left_prob=door_left_prob,
                force_one_primitive_per_link=force_one_primitive_per_link,
            )

        if vertical_split:
            self._add_wall(
                link=self._cabinet.links[0],
                origin=tra.translation_matrix(w_xyz),
                size=(wt, self.depth, height),
                single_link=not force_one_primitive_per_link,
            )

            new_w = width / 2.0 - wt / 2.0
            r_left = (x, y, new_w, height)
            r_right = (x + width / 2.0 + wt / 2.0, y, new_w, height)

            run_split(
                x=r_left[0],
                y=r_left[1],
                width=r_left[2],
                height=r_left[3],
                wt=wt,
                prob=prob_new,
            )
            run_split(
                x=r_right[0],
                y=r_right[1],
                width=r_right[2],
                height=r_right[3],
                wt=wt,
                prob=prob_new,
            )
        else:  # horizontal split
            self._add_wall(
                link=self._cabinet.links[0],
                origin=tra.translation_matrix(w_xyz),
                size=(width, self.depth, wt),
                single_link=not force_one_primitive_per_link,
            )

            new_h = height / 2.0 - wt / 2.0
            r_top = (x, y, width, new_h)
            r_bot = (x, y + height / 2.0 + wt / 2, width, new_h)

            run_split(
                x=r_top[0],
                y=r_top[1],
                width=r_top[2],
                height=r_top[3],
                wt=wt,
                prob=prob_new,
            )
            run_split(
                x=r_bot[0],
                y=r_bot[1],
                width=r_bot[2],
                height=r_bot[3],
                wt=wt,
                prob=prob_new,
            )

    def _local_to_body(self, x, y):
        """Converts local rectangle coordinates into cabinet body coordinates
        The local coordinate frame has x pointing to the right and y to the bottom.

        Args:
            x (float): x-coordinate of local coordinates.
            y (float): y-coordinate of local coordinates.

        Returns:
            list (3,): 3D coordinates in cabinet reference frame.
        """

        bx = self.width / 2.0 - x
        by = self.depth / 2.0
        bz = self.height - y
        return [bx, by, bz]

    def _create_box_visual(self, size, origin, name=None, material=None):
        """Create visual URDF element with box geometry.

        Args:
            size (list): 3D size of box.
            origin (np.ndarray): 4x4 homogenous matrix of box pose.
            name (str, optional): Name of visual element. Defaults to None.
            material (yourdfpy.Material, optional): Material. Defaults to None.

        Returns:
            yourdfpy.Visual: Visual element.
        """
        return yourdfpy.Visual(
            name=name,
            geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=size)),
            origin=origin,
            material=material,
        )

    def _create_mesh_visual(self, filename, scale=None, origin=np.eye(4), name=None, material=None):
        """Create visual URDF element with mesh geometry.

        Args:
            filename (str): Filename of mesh.
            scale (list or float, optional): Scale of mesh geometry. If None, scale equals 1.0. Defaults to None.
            origin (np.ndarray, optional): 4x4 homogenous matrix of mesh pose. Defaults to np.eye(4).
            name (str, optional): Name of visual element. Defaults to None.
            material (yourdfpy.Material, optional): Material. Defaults to None.

        Returns:
            yourdfpy.Visual: Visual element.
        """
        return yourdfpy.Visual(
            name=name,
            geometry=yourdfpy.Geometry(
                mesh=yourdfpy.Mesh(
                    filename=filename,
                    scale=scale,
                )
            ),
            origin=origin,
            material=material,
        )

    def _create_mesh_collision(self, filename, scale=None, origin=np.eye(4), name=None):
        """Create collision URDF element with mesh geometry.

        Args:
            filename (str): Filename of mesh.
            scale (list or float, optional): Scale of mesh geometry. If None, scale equals 1.0. Defaults to None.
            origin (np.ndarray, optional): 4x4 homogenous matrix of mesh pose. Defaults to np.eye(4).
            name (str, optional): Name of collision element. Defaults to None.

        Returns:
            yourdfpy.Collision: Collision element.
        """
        return yourdfpy.Collision(
            name=name,
            geometry=yourdfpy.Geometry(
                mesh=yourdfpy.Mesh(
                    filename=filename,
                    scale=scale,
                )
            ),
            origin=origin,
        )

    def _create_box_collision(self, size, origin, name=None):
        """Create collision URDF element with box geometry.

        Args:
            size (list): 3D size of box.
            origin (np.ndarray): 4x4 homogenous matrix of box pose.
            name (str, optional): Name of collision element. Defaults to None.

        Returns:
            yourdfpy.Collision: Collision element.
        """
        return yourdfpy.Collision(
            name=name,
            geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=size)),
            origin=origin,
        )

    def _add_body(self, name, width, depth, height, thickness=0.02, single_link=True):
        """Generate and add a cabinet body link to the URDF.
        Top and Bottom boards are the same size as width and depth.
        The side boards cover the Bottom and Top boards on the sides.

        Args:
            name (str): Name of the link element.
            width (float): Width of interior of body link.
            depth (float): Depth of interior of body link.
            height (float): Height of interior of body link.
            thickness (float, optional): Wall thickness. Defaults to 0.02.
            single_link (bool, optional): Whether to create a single link for the cabinet body (w/ multiple visuals/collsions). Or to create one link per visual/collision. This allows to keep information about geometric primitives. Defaults to True.
        """
        boxes = [
            {
                "origin": tra.translation_matrix([0, 0, -thickness / 2]),
                "size": (width, depth, thickness),
            },
            {
                "origin": tra.translation_matrix([0, 0, height + thickness / 2]),
                "size": (width, depth, thickness),
            },
            # sideboards
            {
                "origin": tra.translation_matrix([width / 2 + thickness / 2, 0, height / 2]),
                "size": (thickness, depth, height + 2 * thickness),
            },
            {
                "origin": tra.translation_matrix([-width / 2 - thickness / 2, 0, height / 2]),
                "size": (thickness, depth, height + 2 * thickness),
            },
            # backboard
            {
                "origin": tra.translation_matrix([0, -depth / 2.0 + thickness / 2.0, height / 2]),
                "size": (width + 2 * thickness, thickness, height + 2 * thickness),
            },
        ]

        if single_link:
            visuals = []
            collisions = []

            for i, board in enumerate(boxes):
                visuals.append(
                    self._create_box_visual(
                        name=f"{name}_board_{i}",
                        origin=board["origin"],
                        size=board["size"],
                    )
                )
                collisions.append(
                    self._create_box_collision(
                        name=f"{name}_board_{i}",
                        origin=board["origin"],
                        size=board["size"],
                    )
                )

            inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
            link = yourdfpy.Link(
                name=name, inertial=inertial, visuals=visuals, collisions=collisions
            )

            self._cabinet.links.append(link)
        else:
            self._cabinet.links.append(yourdfpy.Link(name=name))

            for i, board in enumerate(boxes):
                inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
                link_name = f"{name}_board_{i}"
                link = yourdfpy.Link(
                    name=link_name,
                    inertial=inertial,
                    visuals=[
                        self._create_box_visual(
                            name=f"{name}_board_{i}",
                            origin=tra.identity_matrix(),
                            size=board["size"],
                        )
                    ],
                    collisions=[
                        self._create_box_collision(
                            name=f"{name}_board_{i}",
                            origin=tra.identity_matrix(),
                            size=board["size"],
                        )
                    ],
                )

                joint = self._create_fixed_joint(
                    # name=f"{name}_fixed_joint_{i}",
                    parent=name,
                    child=link_name,
                    origin=board["origin"],
                )

                self._cabinet.joints.append(joint)
                self._cabinet.links.append(link)

    def _add_door(
        self,
        parent,
        width,
        height,
        x=0.0,
        y=0.0,
        frontboard_thickness=0.019,
        left=False,
        is_top=False,
        connect_handle_with_fixed_joint=False,
    ):
        """Adds a cabinet door with a handle and a revolute joint.

               Args:
                   parent (str): Name of parent link of revolute joint.
                   width (float): Width of door.
                   height (height): Height of door.
                   x (float, optional): Local x-coordinate of door. Defaults to 0.0.
                   y (float, optional): Local y-coordinate of door. Defaults to 0.0.
                   frontboard_thickness (float, optional): Thickness of door. Defaults to 0.019.
                   left (bool, optional): Whether door is left handed or right handed. i.e. https://res.cloudinary.com/ecbarton/image/upload/s--k24LhiF0--/c_fill%2Cf_auto%2Ch_288%2Cq_auto%2Cw_512/v1/blog-assets/Picture1_1.jpg?itok=8b05c9Mw
        Defaults to False.
                   is_top (bool, optional): Currently unused(?). Defaults to False.
        """
        name = "door_" + str(self.num_doors)
        self.num_doors += 1

        # Create additional floor geometry
        if self.additional_compartment_floor_height > 0.0:
            # get parent link
            parent_link = next((x for x in self._cabinet.links if x.name == parent), None)

            floor_xyz = self._local_to_body(
                x + width / 2.0, y + height - self.additional_compartment_floor_height / 2.0
            )
            floor_xyz[1] -= self.depth / 2.0

            parent_link.visuals.append(
                self._create_box_visual(
                    name=f"{name}_floor",
                    origin=tra.translation_matrix(floor_xyz),
                    size=(width, self.depth, self.additional_compartment_floor_height),
                )
            )
            parent_link.collisions.append(
                self._create_box_visual(
                    name=f"{name}_floor",
                    origin=tra.translation_matrix(floor_xyz),
                    size=(width, self.depth, self.additional_compartment_floor_height),
                )
            )

        # Create door link
        offset = width / 2.0 if left else -width / 2.0
        inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
        door_link = yourdfpy.Link(
            name=name,
            inertial=inertial,
            visuals=[
                self._create_box_visual(
                    name=f"{name}_door",
                    origin=tra.translation_matrix([offset, frontboard_thickness / 2, 0]),
                    size=(width, frontboard_thickness, height),
                )
            ],
            collisions=[
                self._create_box_collision(
                    name=f"{name}_door",
                    origin=tra.translation_matrix([offset, frontboard_thickness / 2, 0]),
                    size=(width, frontboard_thickness, height),
                )
            ],
        )

        # Create handle link
        handle_type = self._rng.choice(self.handle_types, p=self.handle_type_prob)
        handle_frame = name + "_handle"

        handle_rotation = [0, 0, 0]
        if handle_type == "box":
            handle_link = yourdfpy.Link(
                name=handle_frame,
                inertial=inertial,
                visuals=[
                    self._create_box_visual(
                        [0.014, 0.1682, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.031, 0, 0]),
                        name=name + "_part_0",
                    ),
                    self._create_box_visual(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, 0.0641, 0]),
                        name=name + "_part_1",
                    ),
                    self._create_box_visual(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, -0.0641, 0]),
                        name=name + "_part_2",
                    ),
                ],
                collisions=[
                    self._create_box_collision(
                        [0.014, 0.1682, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.031, 0, 0]),
                        name=name + "_part_0",
                    ),
                    self._create_box_collision(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, 0.0641, 0]),
                        name=name + "_part_1",
                    ),
                    self._create_box_collision(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, -0.0641, 0]),
                        name=name + "_part_2",
                    ),
                ],
            )
        elif handle_type == "knob":
            handle_link = _create_knob_link(name=handle_frame, **self.knob_kwargs)

        elif handle_type == "procedural":
            handle_link = _create_handle_link(
                name=handle_frame,
                inertial=inertial,
                handle_width=self.handle_width,
                handle_depth=self.handle_depth,
                handle_height=self.handle_height,
                handle_offset=self.handle_offset,
                handle_shape_args=self.handle_shape_args,
            )
            handle_rotation = [0, 0, np.pi]

        # Position the handle w.r.t the door
        handle_dist = RecursivelyPartitionedCabinetAsset._DEFAULT_DOOR_HANDLE_DIST
        offset = np.array((width - handle_dist, 0, 0))
        handle_pos = np.array((0, frontboard_thickness, 0))
        handle_pos = handle_pos + offset if left else handle_pos - offset

        handle_joint_origin = tra.compose_matrix(
            translate=handle_pos,
            angles=handle_rotation,
        )
        joint = self._create_fixed_joint(
            parent=name, child=handle_frame, origin=handle_joint_origin
        )

        # Add links to drawer model
        if connect_handle_with_fixed_joint:
            # This is for adding the handle with a fixed joint
            self._cabinet.links.extend([door_link, handle_link])
            self._cabinet.joints.append(joint)
        else:
            self._cabinet.links.append(door_link)

            assert len(handle_link.visuals) == len(handle_link.collisions)
            for v, c in zip(handle_link.visuals, handle_link.collisions):
                v.origin = handle_joint_origin @ v.origin
                c.origin = handle_joint_origin @ c.origin

                self._cabinet.links[-1].visuals.append(v)
                self._cabinet.links[-1].collisions.append(c)

        pos = [x + width, y + height / 2.0] if left else [x, y + height / 2.0]
        d_xyz = self._local_to_body(*pos)
        axis = np.array([0, 0, 1]) if left else np.array([0, 0, -1])
        self._cabinet.joints.append(
            self._create_revolute_joint(
                parent=parent,
                child=name,
                origin=tra.translation_matrix(d_xyz),
                axis=axis,
            )
        )

    def _create_fixed_joint(self, parent, child, origin):
        """Create a URDF joint element for a fixed joint.

        Args:
            parent (str): Name of parent link.
            child (str): Name of child link.
            origin (np.ndarray): 4x4 homogeneous matrix for joint pose.

        Returns:
            yourdfpy.Joint: Joint element.
        """

        return yourdfpy.Joint(
            name=parent + "_to_" + child,
            type="fixed",
            parent=parent,
            child=child,
            origin=origin,
        )

    def _create_revolute_joint(
        self,
        parent,
        child,
        origin,
        axis,
        damping=None,
        friction=None,
    ):
        """Create a URDF joint element for a revolute joint.

        Args:
            parent (str): Name of parent link.
            child (str): Name of child link.
            origin (np.ndarray): 4x4 homogeneous matrix for joint pose.
            axis (tuple, optional): Joint axis.
            damping (float, optional): Joint damping. Defaults to None.
            friction (float, optional): Joint friction. Defaults to None.

        Returns:
            yourdfpy.Joint: Joint element.
        """
        return yourdfpy.Joint(
            name=parent + "_to_" + child,
            type="revolute",
            parent=parent,
            child=child,
            origin=origin,
            axis=axis,
            limit=yourdfpy.Limit(
                effort=1000.0,
                velocity=0.1,
                lower=0.0,
                upper=np.pi / 2.0,
            ),
            dynamics=yourdfpy.Dynamics(damping=damping, friction=friction),
        )

    def _create_prismatic_joint(
        self,
        parent,
        child,
        origin,
        axis,
        lower=0.0,
        upper=0.4,
        damping=None,
        friction=None,
    ):
        """Create a URDF joint element for a prismatic joint.

        Args:
            parent (str): Name of parent link.
            child (str): Name of child link.
            origin (np.ndarray): 4x4 homogeneous matrix for joint pose.
            axis (tuple, optional): Joint axis.
            lower (float, optional): Lower joint limit. Defaults to 0.0.
            upper (float, optional): Upper joint limit. Defaults to 0.4.
            damping (float, optional): Joint damping. Defaults to None.
            friction (float, optional): Joint friction. Defaults to None.

        Returns:
            yourdfpy.Joint: Joint element.
        """
        return yourdfpy.Joint(
            name=parent + "_to_" + child,
            parent=parent,
            child=child,
            type="prismatic",
            origin=origin,
            axis=axis,
            dynamics=yourdfpy.Dynamics(damping=damping, friction=friction),
            limit=yourdfpy.Limit(effort=1000.0, lower=lower, upper=upper, velocity=1.0),
        )

    def _add_drawer(
        self,
        parent,
        width,
        height,
        x=0.0,
        y=0.0,
        frontboard_thickness=0.019,
        drawer_depth=0.5,
        wall_thickness=0.004,
        single_link=True,
        connect_handle_with_fixed_joint=False,
    ):
        """Add a drawer with a handle and a prismatic joint.

        Args:
            parent (str): Name of parent link of prismatic joint.
            width (float): Width of drawer front.
            height (float): Height of drawer front.
            x (float, optional): Local x-coordinate. Defaults to 0.0.
            y (float, optional): Local y-coordinate. Defaults to 0.0.
            frontboard_thickness (float, optional): Thickness of front board. Defaults to 0.019.
            drawer_depth (float, optional): Depth of drawer part that goes inside cabinet (depth without front board thickness). Defaults to 0.5.
            wall_thickness (float, optional): Thickness of drawer walls. Defaults to 0.004.
            single_link (bool, optional): Whether to create a single link for the drawer body (w/ multiple visuals/collsions). Or to create one link per visual/collision. This allows to keep information about geometric primitives. Defaults to True.
        """
        name = "drawer_" + str(self.num_drawers)
        self.num_drawers += 1

        boxes = [
            # front
            {
                "origin": tra.translation_matrix((0, frontboard_thickness / 2, 0)),
                "size": (width, frontboard_thickness, height),
            },
            # bottom
            {
                "origin": tra.translation_matrix(
                    (
                        0,
                        -(drawer_depth - wall_thickness) / 2,
                        (wall_thickness - height) / 2,
                    )
                ),
                "size": (
                    width - 2 * wall_thickness,
                    drawer_depth - wall_thickness,
                    wall_thickness,
                ),
            },
            # left
            {
                "origin": tra.translation_matrix(
                    (
                        (width - wall_thickness) / 2,
                        (wall_thickness - drawer_depth) / 2,
                        0,
                    )
                ),
                "size": (wall_thickness, drawer_depth - wall_thickness, height),
            },
            # right
            {
                "origin": tra.translation_matrix(
                    (
                        (wall_thickness - width) / 2,
                        (wall_thickness - drawer_depth) / 2,
                        0,
                    )
                ),
                "size": (wall_thickness, drawer_depth - wall_thickness, height),
            },
            # back
            {
                "origin": tra.translation_matrix((0, -drawer_depth + wall_thickness / 2, 0)),
                "size": (width, wall_thickness, height),
            },
        ]

        if single_link:
            visuals = []
            collisions = []

            for i, board in enumerate(boxes):
                visuals.append(
                    self._create_box_visual(
                        name=f"{name}_board_{i}",
                        origin=board["origin"],
                        size=board["size"],
                    )
                )
                collisions.append(
                    self._create_box_collision(
                        name=f"{name}_board_{i}",
                        origin=board["origin"],
                        size=board["size"],
                    )
                )
            inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
            link = yourdfpy.Link(
                name=name,
                inertial=inertial,
                visuals=visuals,
                collisions=collisions,
            )
            self._cabinet.links.append(link)
        else:
            self._cabinet.links.append(yourdfpy.Link(name=name))

            for i, board in enumerate(boxes):
                inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
                link_name = f"{name}_board_{i}"
                link = yourdfpy.Link(
                    name=link_name,
                    inertial=inertial,
                    visuals=[
                        self._create_box_visual(
                            name=f"{name}_board_{i}",
                            origin=tra.identity_matrix(),
                            size=board["size"],
                        )
                    ],
                    collisions=[
                        self._create_box_collision(
                            name=f"{name}_board_{i}",
                            origin=tra.identity_matrix(),
                            size=board["size"],
                        )
                    ],
                )

                joint = self._create_fixed_joint(
                    # name=f"{name}_fixed_joint_{i}",
                    parent=name,
                    child=link_name,
                    origin=board["origin"],
                )

                self._cabinet.joints.append(joint)
                self._cabinet.links.append(link)

        # Create handle link
        handle_type = self._rng.choice(self.handle_types, p=self.handle_type_prob)
        
        handle_frame = name + "_handle"
        handle_rotation = [0.0, 0.0, 0.0]
        if handle_type == "box":
            handle_link = yourdfpy.Link(
                name=handle_frame,
                inertial=inertial,
                visuals=[
                    self._create_box_visual(
                        [0.014, 0.1682, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.031, 0, 0]),
                        name=name + "_part_0",
                    ),
                    self._create_box_visual(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, 0.0641, 0]),
                        name=name + "_part_1",
                    ),
                    self._create_box_visual(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, -0.0641, 0]),
                        name=name + "_part_2",
                    ),
                ],
                collisions=[
                    self._create_box_collision(
                        [0.014, 0.1682, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.031, 0, 0]),
                        name=name + "_part_0",
                    ),
                    self._create_box_collision(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, 0.0641, 0]),
                        name=name + "_part_1",
                    ),
                    self._create_box_collision(
                        [0.024, 0.04, 0.024],
                        tra.euler_matrix(np.pi / 2.0, 0, np.pi / 2.0)
                        @ tra.translation_matrix([0.012, -0.0641, 0]),
                        name=name + "_part_2",
                    ),
                ],
            )
            handle_rotation = (0, np.pi / 2, 0)
        elif handle_type == "knob":
            handle_link = _create_knob_link(name=handle_frame, **self.knob_kwargs)
        elif handle_type == "procedural":
            handle_link = _create_handle_link(
                name=handle_frame,
                inertial=inertial,
                handle_width=self.handle_width,
                handle_depth=self.handle_depth,
                handle_height=self.handle_height,
                handle_offset=self.handle_offset,
                handle_shape_args=self.handle_shape_args,
            )
            handle_rotation = (0, np.pi / 2, np.pi)

        # Position the handle link
        if single_link:
            handle_pos = np.array([0, frontboard_thickness, 0])
            handle_joint_origin = tra.compose_matrix(translate=handle_pos, angles=handle_rotation)
        else:
            handle_pos = np.array(
                [0, frontboard_thickness + drawer_depth - wall_thickness, 0]
            )

            handle_joint_origin = tra.compose_matrix(
                translate=handle_pos,
                angles=handle_rotation,
            )

        joint = self._create_fixed_joint(
            parent=name,
            child=handle_frame,
            origin=handle_joint_origin,
        )

        if connect_handle_with_fixed_joint:
            self._cabinet.links.append(handle_link)
            self._cabinet.joints.append(joint)
        else:
            assert len(handle_link.visuals) == len(handle_link.collisions)
            for v, c in zip(handle_link.visuals, handle_link.collisions):
                v.origin = handle_joint_origin @ v.origin
                c.origin = handle_joint_origin @ c.origin

                self._cabinet.links[-1].visuals.append(v)
                self._cabinet.links[-1].collisions.append(c)

        # create prismatic joint
        d_xyz = self._local_to_body(
            x + width / 2,
            y + height / 2,
        )
        self._cabinet.joints.append(
            self._create_prismatic_joint(
                parent=parent,
                child=name,
                origin=tra.translation_matrix(d_xyz),
                axis=np.array([0, 1, 0]),
                lower=0.0,
                upper=drawer_depth * _DRAWER_DEPTH_PERCENT,
            )
        )

    def _add_wall(self, link, origin, size, single_link=True):
        """Add URDF elements that represent visual and collision geometries for wall.

        Args:
            link (urdfpy.Link): URDF link to which this wall geometry will be added.
            origin (np.ndarray): 4x4 homogenous matrix of wall pose.
            size (list): 3D size of box representing wall.
            single_link (bool, optional): Whether to create a single link for the wall body (w/ multiple visuals/collsions). Or to create one link per visual/collision. This allows to keep information about geometric primitives. Defaults to True.
        """
        if single_link:
            link.visuals.append(
                self._create_box_visual(name=f"{link.name}_wall", size=size, origin=origin)
            )
            link.collisions.append(
                self._create_box_collision(name=f"{link.name}_wall", size=size, origin=origin)
            )
        else:
            new_link = yourdfpy.Link(
                name=f"{link.name}_geometry",
                visuals=[
                    self._create_box_visual(name=f"{link.name}_wall", size=size, origin=origin)
                ],
                collisions=[
                    self._create_box_collision(name=f"{link.name}_wall", size=size, origin=origin)
                ],
            )
            joint = self._create_fixed_joint(
                parent=link.name,
                child=new_link.name,
                origin=tra.identity_matrix(),
            )

            self._cabinet.joints.append(joint)
            self._cabinet.links.append(new_link)


class RangeHoodAsset(TrimeshSceneAsset):
    """A range hood asset."""

    def __init__(
        self,
        width=0.92,
        depth=0.51,
        height=1.0,
        duct_width=0.23,
        duct_depth=0.33,
        duct_offset=(0, 0),
        pyramid_height=0.22,
        control_panel_height=0.04,
        use_primitives=False,
        **kwargs,
    ):
        """A range hood / extractor hood / kitchen hood asset.

        .. image:: /../imgs/range_hood_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the range hood. Defaults to 0.92.
            depth (float, optional): Depth of the range hood. Defaults to 0.51.
            height (float, optional): Height of the range hood. Defaults to 1.0.
            duct_width (float, optional): Width of the duct. Defaults to 0.23.
            duct_depth (float, optional): Depth of the duct. Defaults to 0.33.
            duct_offset (tuple, optional): Offset of the duct in the xy-plane. Defaults to (0, 0).
            pyramid_height (float, optional): Height of the pyramidically shaped blower box. Defaults to 0.22.
            control_panel_height (float, optional): Height of the control panel. Defaults to 0.04.
            use_primitives (bool, optional): Use primitve shapes only. Defaults to False.

        Raises:
            ValueError: If duct_height + pyramid_height exceed the total height.
        """
        fn = RangeHoodAsset.create_primitives if use_primitives else RangeHoodAsset.create_mesh
        super().__init__(
            trimesh.Scene(
                fn(
                    width=width,
                    depth=depth,
                    height=height,
                    duct_width=duct_width,
                    duct_depth=duct_depth,
                    duct_offset=duct_offset,
                    pyramid_height=pyramid_height,
                    control_panel_height=control_panel_height,
                )
            ),
            **kwargs,
        )

    @staticmethod
    def create_primitives(
        width,
        depth,
        height,
        duct_width,
        duct_depth,
        duct_offset,
        pyramid_height,
        control_panel_height,
    ):
        # check validity of arguments
        if (pyramid_height + control_panel_height) > height:
            raise ValueError(
                "RangeHoodAsset's (duct_height + pyramid_height ="
                f" {(pyramid_height + control_panel_height)}) cannot exceed the total height ="
                f" {height}."
            )

        height_0 = control_panel_height
        height_1 = pyramid_height
        width_1 = duct_width
        depth_1 = duct_depth

        height_2 = height - height_0 - height_1
        width_2 = width_1
        depth_2 = depth_1

        wall_offset = (depth - depth_1) / 2.0 + duct_offset[1]

        control_panel_box = trimesh.primitives.Box((width, depth, height_0), transform=tra.translation_matrix((0, 0, height_0 / 2.0)))
        exhaust_pipe = trimesh.primitives.Box((width_2, depth_2,  height_1 + height_2), transform=tra.translation_matrix((0, wall_offset, height_0 + (height_1 + height_2) / 2.0)))

        return (control_panel_box, exhaust_pipe)

    @staticmethod
    def create_mesh(
        width,
        depth,
        height,
        duct_width,
        duct_depth,
        duct_offset,
        pyramid_height,
        control_panel_height,
    ):
        # check validity of arguments
        if (pyramid_height + control_panel_height) > height:
            raise ValueError(
                "RangeHoodAsset's (duct_height + pyramid_height ="
                f" {(pyramid_height + control_panel_height)}) cannot exceed the total height ="
                f" {height}."
            )

        height_0 = control_panel_height
        height_1 = pyramid_height
        width_1 = duct_width
        depth_1 = duct_depth

        height_2 = height - height_0 - height_1
        width_2 = width_1
        depth_2 = depth_1

        wall_offset = (depth - depth_1) / 2.0 + duct_offset[1]

        verts = [
            # control panel part
            [-width / 2.0, -depth / 2.0, 0],
            [+width / 2.0, -depth / 2.0, 0],
            [+width / 2.0, depth / 2.0, 0],
            [-width / 2.0, depth / 2.0, 0],
            [-width / 2.0, -depth / 2.0, height_0],
            [+width / 2.0, -depth / 2.0, height_0],
            [+width / 2.0, depth / 2.0, height_0],
            [-width / 2.0, depth / 2.0, height_0],
            # pyramid part
            [-width_1 / 2.0 + duct_offset[0], -depth_1 / 2.0 + wall_offset, height_0 + height_1],
            [+width_1 / 2.0 + duct_offset[0], -depth_1 / 2.0 + wall_offset, height_0 + height_1],
            [+width_1 / 2.0 + duct_offset[0], +depth_1 / 2.0 + wall_offset, height_0 + height_1],
            [-width_1 / 2.0 + duct_offset[0], +depth_1 / 2.0 + wall_offset, height_0 + height_1],
            # duct part
            [
                -width_2 / 2.0 + duct_offset[0],
                -depth_2 / 2.0 + wall_offset,
                height_0 + height_1 + height_2,
            ],
            [
                +width_2 / 2.0 + duct_offset[0],
                -depth_2 / 2.0 + wall_offset,
                height_0 + height_1 + height_2,
            ],
            [
                +width_2 / 2.0 + duct_offset[0],
                +depth_2 / 2.0 + wall_offset,
                height_0 + height_1 + height_2,
            ],
            [
                -width_2 / 2.0 + duct_offset[0],
                +depth_2 / 2.0 + wall_offset,
                height_0 + height_1 + height_2,
            ],
        ]
        faces = [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [0, 7, 3],
            [0, 4, 7],
            [4, 5, 8],
            [5, 9, 8],
            [5, 10, 9],
            [6, 10, 5],
            [6, 7, 10],
            [7, 11, 10],
            [7, 8, 11],
            [4, 8, 7],
            [8, 9, 13],
            [8, 13, 12],
            [9, 10, 14],
            [9, 14, 13],
            [10, 11, 15],
            [10, 15, 14],
            [8, 15, 11],
            [8, 12, 15],
            [12, 13, 14],
            [14, 15, 12],
        ]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        return mesh
    
    @classmethod
    def random_size_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {}
        params["width"] = kwargs.get("width", rng.uniform(0.92 - 0.1, 0.92 + 0.01))
        params["depth"] = kwargs.get("depth", rng.uniform(0.51 - 0.05, 0.51 + 0.1))
        params["height"] = kwargs.get("height", rng.uniform(1.0 - 0.1, 1.0 + 0.1))

        return params

    @classmethod
    def random_params(cls, seed=None, **kwargs):
        params = {}
        params.update(cls.random_size_params(seed=seed, **kwargs))
        params.update(**kwargs)
        
        return params

    @classmethod
    def random(cls, seed=None, **kwargs):
        params = cls.random_params(seed=seed, **kwargs)
        range_hood = cls(**params)

        return range_hood

class WallCabinetAsset(URDFAsset):
    """A wall cabinet asset."""

    def __init__(
        self,
        width=0.762,
        depth=0.305,
        height=0.762,
        compartment_types=("door_right", "door_left"),
        num_shelves=2,
        frontboard_thickness=0.02,
        wall_thickness=0.01,
        inner_wall_thickness=0.01,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=(-0.25, 0.05),
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """ A wall cabinet. A specialization of the CabinetAsset with two doors.

        .. image:: /../imgs/wall_cabinet_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the wall cabinet. Defaults to 0.762.
            depth (float, optional): Depth of the wall cabinet. Defaults to 0.305.
            height (float, optional): Height of the wall cabinet. Defaults to 0.762.
            compartment_types (tuple, optional): Compartment types. Defaults to ("door_right", "door_left").
            num_shelves (int, optional): Number of interior shelves. Defaults to 2.
            frontboard_thickness (float, optional): Thickness of doors. Defaults to 0.02.
            wall_thickness (float, optional): Thickness of outer walls. Defaults to 0.01.
            inner_wall_thickness (float, optional): Thickness of inner walls. Defaults to 0.01.
            handle_width (float, optional): Handle width. Defaults to 0.1682.
            handle_height (float, optional): Handle height. Defaults to 0.038.
            handle_depth (float, optional): Handle depth. Defaults to 0.024.
            handle_offset (tuple, optional): Handle offset. Defaults to (-0.25, 0.05).
            handle_shape_args (dict, optional): Handle shape parameters. Defaults to None.
            door_shape_args (dict, optional): Door shape parameters. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        compartment_mask = [[i for i, _ in enumerate(compartment_types)]]

        compartment_interior_masks = {i: [[j] for j in range(num_shelves + 1)] for i in range(len(compartment_types))}

        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width,
            depth=depth,
            height=height,
            compartment_mask=np.array(compartment_mask),
            compartment_types=compartment_types,
            compartment_interior_masks=compartment_interior_masks,
            compartment_widths=None,
            compartment_heights=None,
            outer_wall_thickness=wall_thickness,
            inner_wall_thickness=inner_wall_thickness,
            frontboard_thickness=frontboard_thickness,
            frontboard_overlap=0.9,
            handle_width=handle_width,
            handle_height=handle_height,
            handle_depth=handle_depth,
            handle_offset=handle_offset,
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
        )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @classmethod
    def random(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        kwargs["width"] = kwargs.get("width", rng.uniform(0.762 - 0.1, 0.762 + 0.01))
        kwargs["depth"] = kwargs.get("depth", rng.uniform(0.305 - 0.05, 0.305 + 0.1))
        kwargs["height"] = kwargs.get("height", rng.uniform(0.762 - 0.1, 0.762 + 0.1))
        kwargs["num_shelves"] = kwargs.get("num_shelves", rng.integers(1, 4))

        if "handle_shape_args" not in kwargs:
            kwargs["handle_shape_args"] = HandleAsset.random_shape_params(seed=seed)

        if "door_shape_args" not in kwargs:
            kwargs["door_shape_args"] = CabinetDoorAsset.random_shape_params(seed=seed)

        base_cabinet = cls(
            **kwargs,
        )

        return base_cabinet


class KitchenIslandAsset(URDFAsset):
    """A kitchen island asset."""

    def __init__(
        self,
        width=1.22,
        depth=0.64,
        height=0.91,
        depth_storage=0.35,
        depth_side_storage=0.1,
        countertop_thickness=0.03,
        frontboard_thickness=0.02,
        wall_thickness=0.01,
        handle_width=0.1682,
        handle_height=0.038,
        handle_depth=0.024,
        handle_offset=(0.18, 0.05),
        handle_shape_args=None,
        door_shape_args=None,
        **kwargs,
    ):
        """A kitchen island consisting of cabinets, drawers and open shelves below a countertop.

        .. image:: /../imgs/kitchen_island_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the island. Defaults to 1.22.
            depth (float, optional): Depth of the island. Defaults to 0.64.
            height (float, optional): Height of the island. Defaults to 0.91.
            depth_storage (float, optional): Depth of storage cabinets. Defaults to 0.35.
            depth_side_storage (float, optional): Depth of open side compartments. Defaults to 0.1.
            countertop_thickness (float, optional): Thickness of countertop. Defaults to 0.03.
            frontboard_thickness (float, optional): Thickness of cabinet doors. Defaults to 0.02.
            wall_thickness (float, optional): Outer wall thickness of cabinets. Defaults to 0.01.
            handle_width (float, optional): Width of handles. Defaults to 0.1682.
            handle_height (float, optional): Height of handles. Defaults to 0.038.
            handle_depth (float, optional): Depth of handles. Defaults to 0.024.
            handle_offset (tuple[float, float], optional): Offset of handles. Defaults to (0.18, 0.05).
            handle_shape_args (dict, optional): Handle shape parameters. Defaults to None.
            door_shape_args (dict, optional): Door shape parameters. Defaults to None.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.

        Raises:
            ValueError: If depth_storage is bigger than depth.
            ValueError: If depth_side_storage is bigger than width/2.0.
        """
        self._init_default_attributes(**kwargs)

        if depth_storage > depth:
            raise ValueError(
                f"KitchenIslandAsset: depth_storage={depth_storage} needs to be smaller or equal to"
                f" depth={depth}."
            )
        if depth_side_storage * 2.0 > width:
            raise ValueError(
                f"KitchenIslandAsset: depth_side_storage={depth_storage} needs to be smaller or"
                f" equal to width/2={width/2.0}."
            )

        compartment_mask = [[0, 1], [2, 3], [4, 4]]
        compartment_types = ("drawer", "drawer", "door_right", "door_left", "closed")
        compartment_heights = [0.2, 0.7, 0.1]

        width_center_storage = width - 2.0 * depth_side_storage

        # Create front of island
        urdf_model = CabinetAsset._create_yourdfpy_model(
            width=width_center_storage,
            depth=depth_storage,
            height=height - countertop_thickness,
            compartment_mask=compartment_mask,
            compartment_types=compartment_types,
            compartment_heights=compartment_heights,
            compartment_widths=None,
            outer_wall_thickness=wall_thickness,
            inner_wall_thickness=wall_thickness,
            frontboard_thickness=frontboard_thickness,
            frontboard_overlap=0.9,
            handle_width=handle_width,
            handle_height=handle_height,
            handle_depth=handle_depth,
            handle_offset=handle_offset,
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
        )

        # Create sides of island
        for i in range(2):
            urdf_model_side = CabinetAsset._create_yourdfpy_model(
                width=depth_storage,
                depth=depth_side_storage,
                height=height - countertop_thickness,
                compartment_mask=[[0], [1], [2], [3]],
                compartment_types=("open", "open", "open", "closed"),
                compartment_heights=[0.3, 0.3, 0.3, 0.1],
                compartment_widths=None,
                outer_wall_thickness=wall_thickness,
                inner_wall_thickness=wall_thickness,
            )

            # append to existing model
            # and transform while doing it
            # assumes that geometries are not meshes (!)
            if i == 0:
                transform = tra.translation_matrix(
                    [width / 2.0 - depth_side_storage / 2.0, 0, 0]
                ) @ tra.euler_matrix(0, 0, np.pi / 2.0)
            else:
                transform = tra.translation_matrix(
                    [-width / 2.0 + depth_side_storage / 2.0, 0, 0]
                ) @ tra.euler_matrix(0, 0, -np.pi / 2.0)

            for link in urdf_model_side.links:
                for v in link.visuals:
                    v.origin = transform @ v.origin
                    urdf_model.links[0].visuals.append(v)

                for c in link.collisions:
                    c.origin = transform @ c.origin
                    urdf_model.links[0].collisions.append(c)

        # Put countertop on top
        urdf_model.links[0].visuals.append(
            yourdfpy.Visual(
                name="countertop",
                origin=tra.translation_matrix(
                    [0, depth_side_storage, height - countertop_thickness / 2.0]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(size=[width, depth, countertop_thickness])
                ),
            )
        )
        urdf_model.links[0].collisions.append(
            yourdfpy.Collision(
                name="countertop",
                origin=tra.translation_matrix(
                    [0, depth_side_storage, height - countertop_thickness / 2.0]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(size=[width, depth, countertop_thickness])
                ),
            )
        )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))


class CabinetDoorAsset(TrimeshAsset):
    """A cabinet door asset."""

    def __init__(
        self,
        width,
        height,
        depth,
        inner_depth_ratio=0.0,
        outer_depth_ratio=1.0,
        trim_depth_ratio=1.0,
        trim_width_ratio=0.0,
        trim_outer_offset_ratio=0.3,
        knot_0=0.5,
        knot_1=0.6,
        knot_2=0.7,
        knot_3=0.8,
        num_depth_sections=20,
        use_primitives=False,
        **kwargs,
    ):
        """Procedural cabinet door asset based on revolving a B-Spline around its y-axis to create molding.

        .. image:: /../imgs/cabinet_door_asset.png
            :align: center
            :width: 250px
        
        Args:
            width (float): Width of the resulting cabinet door.
            height (float): Height of the resulting cabinet door.
            depth (float): Depth of the resulting cabinet door.
            inner_depth_ratio (float, optional): Defaults to 0.0.
            outer_depth_ratio (float, optional): Defaults to 1.0.
            trim_depth_ratio (float, optional): Defaults to 1.0.
            trim_width_ratio (float, optional): Defaults to 0.0.
            trim_outer_offset_ratio (float, optional): Defaults to 0.3.
            knot_0 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.5.
            knot_1 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.6.
            knot_2 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.7.
            knot_3 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.8.
            num_depth_sections (int, optional): Number of interpolated sections along B-Spline. Defaults to 20.
            use_primitives (bool, optional): Defaults to False.
            **kwargs: Keyword argument passed onto the TrimeshAsset constructor.
        """
        mesh = CabinetDoorAsset._create_door_mesh(
            width=width,
            height=height,
            depth=depth,
            inner_depth_ratio=inner_depth_ratio,
            outer_depth_ratio=outer_depth_ratio,
            trim_depth_ratio=trim_depth_ratio,
            trim_width_ratio=trim_width_ratio,
            trim_outer_offset_ratio=trim_outer_offset_ratio,
            knot_0=knot_0,
            knot_1=knot_1,
            knot_2=knot_2,
            knot_3=knot_3,
            num_depth_sections=num_depth_sections,
            use_primitives=use_primitives,
        )

        super().__init__(
            mesh,
            **kwargs,
        )
    
    @staticmethod
    def _create_urdf_geometries(
        name,
        width,
        height,
        depth,
        geometry_origin=None,
        tmp_mesh_dir="/tmp",
        **door_shape_args
    ):
        visual_geometries = []
        collision_geometries = []

        if door_shape_args is None or len(door_shape_args) == 0 or door_shape_args.get('use_primitives', False):
            trim_width_ratio = door_shape_args.get('trim_width_ratio', 0.0)
            inner_depth_ratio = door_shape_args.get('trim_width_ratio', None)

            if trim_width_ratio == 0 or inner_depth_ratio is None:
                box_primitives = {
                    '_door': ((width, height, depth), geometry_origin),
                }
            else:
                # Add five panels
                side_width = width * trim_width_ratio / 2.0
                side_depth = depth * trim_width_ratio / 2.0
                
                box_primitives = {
                    '_door_left': ((side_width, height, depth), geometry_origin @ tra.translation_matrix(((side_width - width) / 2.0, 0, 0))),
                    '_door_right': ((side_width, height, depth), geometry_origin @ tra.translation_matrix(((-side_width + width) / 2.0, 0, 0))),
                    '_door_top': ((width - 2. * side_width, height, side_depth), geometry_origin @ tra.translation_matrix((0, 0, (side_depth - depth) / 2.0))),
                    '_door_bottom': ((width - 2. * side_width, height, side_depth), geometry_origin @ tra.translation_matrix((0, 0, (-side_depth + depth) / 2.0))),
                    '_door_window': ((width - 2. * side_width, height * inner_depth_ratio, depth - 2.*side_depth    ), geometry_origin),
                }

            for k, v in box_primitives.items():
                visual_geometries.append(yourdfpy.Visual(
                    name=f"{name}{k}",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=v[0])
                    ),
                    origin=v[1],
                ))
                collision_geometries.append(yourdfpy.Collision(
                    name=f"{name}{k}",
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=v[0])
                    ),
                    origin=v[1],
                ))                    
        else:
            door_mesh = CabinetDoorAsset._create_door_mesh(
                width, height, depth, **door_shape_args
            )
            door_mesh_fname = utils.get_random_filename(
                dir=tmp_mesh_dir,
                prefix="cabinet_door_",
                suffix=".obj",
            )
            door_mesh.export(door_mesh_fname)
            panel_geometry_vis = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))
            panel_geometry_coll = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=door_mesh_fname))

            visual_geometries.append(yourdfpy.Visual(
                name=f"{name}_door",
                geometry=panel_geometry_vis,
                origin=geometry_origin,
            ))
            collision_geometries.append(yourdfpy.Collision(
                name=f"{name}_door",
                geometry=panel_geometry_coll,
                origin=geometry_origin,
            ))
        return visual_geometries, collision_geometries

    @staticmethod
    def _create_door_box(
        width,
        depth,
        height,
        inner_depth_ratio=0.0,
        outer_depth_ratio=1.0,
        trim_width_ratio=0.0,
    ):  
        if trim_width_ratio == 0:
            m = trimesh.primitives.Box(
                extents=(width, depth, height),
                transform=tra.euler_matrix(np.pi / 2.0, 0, 0),
                )
        else:
            m1 = BoxWithHoleAsset.create_mesh(
                width=width,
                depth=height,
                height=depth,
                hole_width=(1.0-trim_width_ratio)*width,
                hole_height=(1.0-trim_width_ratio)*depth
            )
            m2 = trimesh.primitives.Box(
                extents=(width*(1.-trim_width_ratio), depth*(1.0-trim_width_ratio), inner_depth_ratio*height),
                transform=tra.euler_matrix(np.pi / 2.0, 0, 0),
            )
            m = (m1 + m2)

        return m
        

    @staticmethod
    def _create_door_mesh(
        width,
        height,
        depth,
        inner_depth_ratio=0.0,
        outer_depth_ratio=1.0,
        trim_depth_ratio=1.0,
        trim_width_ratio=0.0,
        trim_outer_offset_ratio=0.3,
        knot_0=0.5,
        knot_1=0.6,
        knot_2=0.7,
        knot_3=0.8,
        num_depth_sections=20,
        use_primitives=False,
    ):
        """Procedural cabinet door asset based on revolving a B-Spline around its y-axis.

        Args:
            width (float): Width of the resulting door mesh.
            height (float): Height of the resulting door mesh.
            depth (float): Depth of the resulting door mesh.
            inner_depth_ratio (float, optional): Defaults to 0.0.
            outer_depth_ratio (float, optional): Defaults to 1.0.
            trim_depth_ratio (float, optional): Defaults to 1.0.
            trim_width_ratio (float, optional): Defaults to 0.0.
            trim_outer_offset_ratio (float, optional): Defaults to 0.3.
            knot_0 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.5.
            knot_1 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.6.
            knot_2 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.7.
            knot_3 (float, optional): A knot of the B-spline used to interpolate the silhoutte. Defaults to 0.8.
            num_depth_sections (int, optional): Number of interpolated sections along B-Spline. Defaults to 20.

        Raises:
            ValueError: If width != height.
            ValueError: If base_ratio not \\in ]0, 1[.
            ValueError: If knob_ratio not \\in ]0, 1[.
            ValueError: If knot_0 not in \\in ]0, knot_1].
            ValueError: If knot_1 not in \\in [knot_0, 1[.

        Returns:
            trimesh.Trimesh: A mesh representing the door.
        """
        for variable in [
            inner_depth_ratio,
            trim_depth_ratio,
            trim_width_ratio,
            trim_outer_offset_ratio,
            knot_0,
            knot_1,
            knot_2,
            knot_3,
        ]:
            if not (1.0 >= variable >= 0.0):
                raise ValueError(
                    "All *_ratio and knot arguments for CabinetDoorAsset must be between 0 and 1"
                    f" (one is currently: {variable})"
                )

        if outer_depth_ratio is not None and not (1.0 >= outer_depth_ratio >= 0.0):
            raise ValueError(
                "outer_depth_ratio for CabinetDoorAsset must be between 0 and 1 (currently:"
                f" {outer_depth_ratio})"
            )

        for knot_a, knot_b in zip([knot_0, knot_1, knot_2], [knot_1, knot_2, knot_3]):
            if knot_a > knot_b:
                raise ValueError(
                    f"knot_t <= knot_t+1 for CabinetDoorAsset. Currently {knot_a}>{knot_b}"
                )

        tmp_height = 1.4142135623730951  # sqrt(2)
        tmp_depth = 1.0
        half_height = tmp_height / 2.0

        knots = [0, 0, 0, knot_0, knot_1, knot_2, knot_3, 1, 1, 1]
        inner_depth = inner_depth_ratio * tmp_depth
        outer_depth = inner_depth if outer_depth_ratio is None else outer_depth_ratio * tmp_depth
        trim_depth = trim_depth_ratio * tmp_depth
        trim_width = trim_width_ratio * half_height
        trim_outer_offset = trim_outer_offset_ratio * half_height
        coefficients = [
            [0, inner_depth],
            [half_height - trim_outer_offset - trim_width, inner_depth],
            [half_height - trim_outer_offset - trim_width, trim_depth],
            [half_height - trim_outer_offset, trim_depth],
            [half_height - trim_outer_offset, outer_depth],
            [half_height, outer_depth],
            [half_height, 0],
        ]

        bspline = BSpline(t=knots, c=coefficients, k=2)
        x = np.linspace(0, 1, num_depth_sections)

        linestring = np.concatenate([[[0.0, 0.0]], bspline(x)[:, ::]])

        m = trimesh.creation.revolve(
            linestring=linestring[::-1, :],
            sections=4,
        )
        m.apply_transform(tra.euler_matrix(0, 0, np.pi / 4.0))
        m.apply_transform(tra.euler_matrix(np.pi / 2.0, 0, 0))
        m.apply_scale(np.array([width, height, depth]) / m.extents)
        m.apply_translation([0, height / 2.0, 0])

        return m

    @classmethod
    def random_shape_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {}
        params["inner_depth_ratio"] = kwargs.get("inner_depth_ratio", rng.uniform(0.0, 1.0))
        params["outer_depth_ratio"] = kwargs.get("outer_depth_ratio", rng.uniform(0.0, 1.0))
        params["trim_depth_ratio"] = kwargs.get("trim_depth_ratio", rng.uniform(0.0, 1.0))
        params["trim_width_ratio"] = kwargs.get("trim_width_ratio", rng.uniform(0.0, 1.0))
        params["trim_outer_offset_ratio"] = kwargs.get(
            "trim_outer_offset_ratio", rng.uniform(0.0, 1.0)
        )

        # knots = rng.uniform(0, 1.0, size=4)
        params["knot_0"] = kwargs.get("knot_0", rng.uniform(0.0, 0.4))
        params["knot_1"] = kwargs.get("knot_1", rng.uniform(0.4, 0.6))
        params["knot_2"] = kwargs.get("knot_2", rng.uniform(0.6, 0.8))
        params["knot_3"] = kwargs.get("knot_3", rng.uniform(0.8, 1.0))

        params["num_depth_sections"] = kwargs.get("num_depth_sections", rng.integers(10, 25))

        return params

    @classmethod
    def random_size_params(cls, seed=None, **kwargs):
        rng = np.random.default_rng(seed)

        params = {}
        params["width"] = kwargs.get("width", rng.uniform(0.2 - 0.06, 0.2 + 0.06))
        params["depth"] = kwargs.get("depth", rng.uniform(0.2, 0.6))
        params["height"] = kwargs.get("height", rng.uniform(0.02, 0.04))

        return params

    @classmethod
    def random(cls, seed=None, **kwargs):
        params = cls.random_shape_params(seed=seed, **kwargs)
        params.update(cls.random_size_params(seed=seed, **kwargs))

        cabinet_door = cls(**params)

        return cabinet_door


class HandWheelAsset(URDFAsset):
    """A handwheel asset."""

    def __init__(
        self,
        radius=0.05,
        rim_width=0.01,
        num_spokes=3,
        spoke_angle=0.4,
        spoke_width=0.01,
        hub_height=0.02,
        hub_radius=0.005,
        handle_height=0.0,
        handle_radius=None,
        spoke_depth=None,
        num_major_segments=32,
        num_minor_segments=16,
        tmp_mesh_dir="/tmp",
        joint_limit_lower=0.0,
        joint_limit_upper=10.0,
        joint_limit_velocity=100.0,
        joint_limit_effort=1000.0,
        **kwargs,
    ):
        """A procedural handwheel asset as it is used to close and open valves.
        The wheel is a torus with cuboid spokes and a cylindrical hub.
        It has an optional handle sticking out the wheel plane.
        The handwheel has a single revolute degree of freedom.

        .. image:: /../imgs/handwheel_asset.png
            :align: center
            :width: 250px

        Args:
            radius (float): Radius of the wheel. Defaults to 0.05.
            rim_width (float): Width of the wheel rim. Defaults to 0.01.
            num_spokes (int): Number of spokes. Defaults to 3.
            spoke_angle (float): Angle of spokes w.r.t. the plane of the wheel (0 == in the wheel plane). In Radians. Defaults to 0.4.
            spoke_width (float or str): Width of the spokes. If argument has special string value 'filled' the spokes will be as wide as needed to fill out the torus. Note that in this case the num_spokes still matter and can create a smoother appearance. Defaults to 0.01.
            hub_height (float): Height of the wheel hub cylinder. Defaults to 0.02.
            hub_radius (float): Radius of the wheel hub cylinder. Defaults to 0.005.
            handle_height (float): Height of the handle sticking out of the wheel. If zero no handle is created. Defaults to 0.0.
            handle_radius (float, optional): The radius of the cylindrical handle. If None, the radius will be 0.35 * rim_width. Defaults to None.
            spoke_depth (float, optional): Depth of the spoke. If None, will be equal to rim_width. Defaults to None.
            num_major_segments (int, optional): The number of discretized major segments of the wheel torus. Defaults to 32.
            num_minor_segments (int, optional): The number of discretized minor segments of the wheel torus. Defaults to 16.
            tmp_mesh_dir (str, optional): Directory to save the mesh of the wheel. Defaults to '/tmp'.
            joint_limit_lower (float, optional): Lower revolute joint limit in radians. Defaults to 0.0.
            joint_limit_upper (float, optional): Upper revolute joint limit in radians. Defaults to 10.0.
            joint_limit_velocity (float, optional): Joint velocity limit. Defaults to 100.0.
            joint_limit_effort (float, optional): Joint effort limit. Defaults to 1000.0.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        wheel_mesh = HandWheelAsset.create_handwheel_mesh(
            radius=radius,
            rim_width=rim_width,
            num_spokes=num_spokes,
            spoke_angle=spoke_angle,
            spoke_width=spoke_width,
            spoke_depth=spoke_depth,
            hub_height=hub_height,
            hub_radius=hub_radius,
            handle_height=handle_height,
            handle_radius=handle_radius,
            num_minor_segments=num_minor_segments,
            num_major_segments=num_major_segments,
        )
        wheel_mesh_fname = utils.get_random_filename(
            dir=tmp_mesh_dir,
            prefix="hand_wheel_",
            suffix=".obj",
        )
        wheel_mesh.export(wheel_mesh_fname)
        wheel_geometry_vis = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=wheel_mesh_fname))
        wheel_geometry_coll = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=wheel_mesh_fname))

        urdf_model = yourdfpy.Robot(name="HandWheel")
        urdf_model.links.append(yourdfpy.Link(name="shaft"))
        urdf_model.links.append(
            yourdfpy.Link(
                name="wheel",
                visuals=[yourdfpy.Visual(name="wheel", geometry=wheel_geometry_vis)],
                collisions=[yourdfpy.Collision(name="wheel", geometry=wheel_geometry_coll)],
            )
        )

        urdf_model.joints.append(
            yourdfpy.Joint(
                name="wheel_joint",
                type="revolute",
                parent=urdf_model.links[0].name,
                child=urdf_model.links[1].name,
                limit=yourdfpy.Limit(
                    lower=joint_limit_lower,
                    upper=joint_limit_upper,
                    velocity=joint_limit_velocity,
                    effort=joint_limit_effort,
                ),
                axis=np.array([0, 0, 1.0]),
                origin=np.eye(4),
            )
        )

        self._model = yourdfpy.URDF(
            robot=urdf_model,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @staticmethod
    def create_handwheel_mesh(
        radius,
        num_major_segments=32,
        num_minor_segments=20,
        rim_width=0.04,
        num_spokes=10,
        spoke_angle=0.3,
        spoke_width=0.01,
        spoke_depth=None,
        hub_height=0.02,
        hub_radius=0.02,
        handle_height=0.05,
        handle_radius=None,
    ):
        elements = []
        if rim_width > 0:
            wheel = utils.create_torus(
                major_radius=radius,
                minor_radius=rim_width / 2.0,
                num_major_segments=num_major_segments,
                num_minor_segments=num_minor_segments,
            )
            elements = [wheel]

        if spoke_depth is None:
            spoke_depth = rim_width if rim_width > 0 else hub_height

        spoke_length = radius / np.cos(spoke_angle)
        spoke_z = np.sin(spoke_angle) * spoke_length
        spokes = []
        start_alpha = 0.0
        if spoke_width == 'filled':
            spoke_width = 2 * radius * np.sin(np.pi / num_spokes)
            start_alpha = np.pi / num_spokes
            # reduce spoke length by circle discretization error
            spoke_length -=  radius * (1.0 - np.cos(np.pi / num_spokes))
        for alpha in np.linspace(0, np.pi * 2.0, num_spokes, endpoint=False):
            spoke = trimesh.creation.box(
                [spoke_depth, spoke_width, spoke_length],
                transform=tra.translation_matrix((0, 0, -spoke_z))
                @ tra.euler_matrix(0, np.pi / 2.0 - spoke_angle, start_alpha + alpha)
                @ tra.translation_matrix((0, 0, spoke_length / 2.0)),
            )
            spokes.append(spoke)
        elements.extend(spokes)

        # Make sure that the hub aligns with the angled spoke and doesn't protrude
        extra_spoke_depth_z = spoke_depth / (2 * np.sin(np.pi / 2.0 - spoke_angle))
        # check if negative
        extra_hub_z = (hub_height / 2.0) - (hub_radius / np.tan(np.pi / 2.0 - spoke_angle))
        
        hub = trimesh.creation.cylinder(
            height=hub_height, radius=hub_radius, transform=tra.translation_matrix((0, 0, -spoke_z - max(extra_hub_z - extra_spoke_depth_z, 0.0)))
        )
        elements.append(hub)

        if handle_height > 0:
            if handle_radius is None:
                handle_radius = 0.7 * rim_width / 2.0
            handle = trimesh.creation.cylinder(
                height=handle_height,
                radius=handle_radius,
                transform=tra.translation_matrix((radius, 0, handle_height / 2.0)),
            )
            elements.append(handle)

        return trimesh.Scene(elements)


class CNCMachineAsset(URDFAsset):
    """A CNC Machine asset."""

    def __init__(
        self,
        width=5.3,
        depth=3.1,
        height=2.3,
        handle_length=0.66,
        handle_thickness=0.05,
        handle_depth=0.15,
        handle_offset=(-0.1, 0.1),
        handle_shape_args={
            "straight_ratio": 0.95,
            "curvature_ratio": 0.5,
            "num_segments_cross_section": 10,
            "num_segments_curvature": 16,
            "aspect_ratio_cross_section": 1.0,
            "tmp_mesh_dir": "/tmp",
        },
        button_size=0.03,
        button_panel_offset=(0, 0),
        window_size=0.0,
        **kwargs,
    ):
        """A simple asset looking like an industrial CNC machine, useful for industrial manipulation tasks.
        The model consists of a prismatic sliding door, with a parameterizable handle, nine square buttons and two round ones.

        .. image:: /../imgs/cnc_machine_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the model. Defaults to 5.3.
            depth (float, optional): Depth of the model. Defaults to 3.1.
            height (float, optional): Height of the model. Defaults to 2.3.
            handle_length (float, optional): Length of the handle. Defaults to 0.66.
            handle_thickness (float, optional): Thickness of the handle. Defaults to 0.05.
            handle_depth (float, optional): Depth of the Handle. Defaults to 0.15.
            handle_offset (tuple, optional): Offset of the handle relative to the door in the model's XY plane. Defaults to (-0.1, 0.1).
            handle_shape_args (dict, optional): Dictionary of shape parameters for the handle. Defaults to { "straight_ratio": 0.95, "curvature_ratio": 0.5, "num_segments_cross_section": 10, "num_segments_curvature": 16, "aspect_ratio_cross_section": 1.0, "tmp_mesh_dir": "/tmp", }.
            button_size (float, optional): Size of the buttons. If None or 0.0 no buttons will be added. Buttons are articulated. Defaults to 0.03.
            button_panel_offset (tuple, optional): Offset of the button panel relative to the left front part of the machine. Defaults to (0, 0).
            window_size (float, optional): Relative size of the window in the front door. Number between 0.0 (no window) and 1.0 (full window). Defaults to 0.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=self._create_yourdfpy_model(
                width=width,
                depth=depth,
                height=height,
                handle_length=handle_length,
                handle_thickness=handle_thickness,
                handle_depth=handle_depth,
                handle_offset=handle_offset,
                handle_shape_args=handle_shape_args,
                button_size=button_size,
                button_panel_offset=button_panel_offset,
                window_size=window_size,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    def _create_yourdfpy_model(
        self,
        width,
        depth,
        height,
        handle_length,
        handle_thickness,
        handle_depth,
        handle_offset,
        handle_shape_args,
        button_size,
        button_panel_offset,
        window_size,
    ):
        width_ratios = (
            0.2641509433962264,
            0.6415094339622641,
            0.6415094339622641,
            0.09433962264150944,
        )
        depth_ratios = (1.0, 0.7096774193548387, 0.3870967741935484, 0.8387096774193549)
        height_ratios = (1.0, 0.2173913043478261, 0.782608695652174, 1.0)
        model = yourdfpy.Robot(name="CNCMachine")

        body_link = yourdfpy.Link(name="body")
        body_link.visuals.append(
            yourdfpy.Visual(
                name="left_part",
                origin=tra.translation_matrix(
                    [0.13207547 * width, -0.01612903 * depth, 0.15217391 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=[
                            width * width_ratios[0],
                            depth * depth_ratios[0],
                            height * height_ratios[0],
                        ]
                    )
                ),
            )
        )
        body_link.visuals.append(
            yourdfpy.Visual(
                name="bottom_part",
                origin=tra.translation_matrix(
                    [-0.32075472 * width, -0.16129032 * depth, -0.23913043 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=[
                            width * width_ratios[1],
                            depth * depth_ratios[1],
                            height * height_ratios[1],
                        ]
                    )
                ),
            )
        )
        body_link.visuals.append(
            yourdfpy.Visual(
                name="center_back",
                origin=tra.translation_matrix(
                    [-0.32075472 * width, -0.32258065 * depth, 0.26086957 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=[
                            width * width_ratios[2],
                            depth * depth_ratios[2],
                            height * height_ratios[2],
                        ]
                    )
                ),
            )
        )
        body_link.visuals.append(
            yourdfpy.Visual(
                name="right_part",
                origin=tra.translation_matrix(
                    [-0.68867925 * width, -0.09677419 * depth, 0.15217391 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=[
                            width * width_ratios[3],
                            depth * depth_ratios[3],
                            height * height_ratios[3],
                        ]
                    )
                ),
            )
        )
        body_link.visuals.append(
            yourdfpy.Visual(
                name="cover_front",
                origin=tra.translation_matrix(
                    [-0.48113208 * width, 0.17419355 * depth, 0.25217391 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=(0.32075472 * width, 0.01290323 * depth, 0.76521739 * height)
                    )
                ),
            )
        )
        body_link.visuals.append(
            yourdfpy.Visual(
                name="cover_top",
                origin=tra.translation_matrix(
                    [-0.48113208 * width, 0.01935484 * depth, 0.62608696 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(
                        size=[0.32075472 * width, 0.29677419 * depth, 0.0173913 * height]
                    )
                ),
            )
        )

        workpiece_support_radius = 0.05660377358490566 * min(width, depth)
        body_link.visuals.append(
            yourdfpy.Visual(
                name="workpiece_support",
                origin=tra.translation_matrix(
                    [-0.16037736 * width, 0.0 * depth, -0.08695652 * height]
                ),
                geometry=yourdfpy.Geometry(
                    cylinder=yourdfpy.Cylinder(
                        radius=workpiece_support_radius, length=0.08695652173913045 * height
                    )
                ),
            )
        )

        # add display
        if button_size is not None and button_size > 0:
            display_width = 4.2 * button_size
            display_height = 0.06
            total_panel_height = display_height * 1.2 + 4 * (button_size * 1.6) + 8 * button_size

            if total_panel_height > height:
                raise ValueError(
                    f"Button panel doesn't fit height={height}. Either increase height or reduce"
                    f" button_size={button_size}."
                )

            display_left = display_width * 1.2 + button_panel_offset[0]
            display_top = -0.34782609 * height + total_panel_height + button_panel_offset[1]
            body_link.visuals.append(
                yourdfpy.Visual(
                    name=f"display",
                    origin=tra.translation_matrix(
                        [
                            display_left - (button_size * 1.6),
                            0.48387096774193544 * depth + 0.005,
                            display_top,
                        ]
                    ),
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(size=[display_width, 0.01, display_height])
                    ),
                )
            )
            model.links.append(body_link)

            button_links = []
            button_depth = 0.01
            button_big_depth = 0.03
            # Add 3x3 square buttons
            for i in range(3):
                for j in range(3):
                    button_link = yourdfpy.Link(f"button_{i*3 + j}")
                    button_link.visuals.append(
                        yourdfpy.Visual(
                            name=f"button_{i*3 + j}",
                            origin=tra.translation_matrix(
                                [
                                    display_left - j * (button_size * 1.6),
                                    0.48387096774193544 * depth + button_depth / 2.0,
                                    display_top - display_height * 1.2 - i * (button_size * 1.6),
                                ]
                            ),
                            geometry=yourdfpy.Geometry(
                                box=yourdfpy.Box(size=[button_size, button_depth, button_size])
                            ),
                        )
                    )
                    button_links.append(button_link)
            # Add 2x1 round buttons
            for i in range(2):
                button_link = yourdfpy.Link(f"button_big_{i}")
                button_link.visuals.append(
                    yourdfpy.Visual(
                        name=f"button_big_{i}",
                        origin=tra.translation_matrix(
                            [
                                display_left - button_size * 1.6,
                                0.48387096774193544 * depth + button_big_depth / 2.0,
                                display_top
                                - display_height * 1.2
                                - 4 * (button_size * 1.6)
                                - i * 4 * button_size,
                            ]
                        )
                        @ tra.euler_matrix(np.pi / 2.0, 0, 0),
                        geometry=yourdfpy.Geometry(
                            cylinder=yourdfpy.Cylinder(
                                radius=1.3 * button_size, length=button_big_depth
                            )
                        ),
                    )
                )
                button_links.append(button_link)
            for bl in button_links:
                model.links.append(bl)
        else:
            model.links.append(body_link)

        door_size = (0.32075472 * width, 0.01290323 * depth, 0.7826087 * height)
        door_xyz = (-0.16037736 * width, 0.18709677 * depth, 0.26086957 * height)

        # Add another link for door and handle
        door_link = yourdfpy.Link(name="door")
        if window_size > 0 and window_size < 1.0:
            # Create five boxes for front instead of one
            # From left to right, top to bottom
            window_names = [
                "door_front_left",
                "door_front_top",
                "door_front_window",
                "door_front_bottom",
                "door_front_right",
            ]
            window_sizes = [
                (door_size[0] * (1 - window_size) / 2.0, door_size[1], door_size[2]),
                (
                    door_size[0] * window_size,
                    door_size[1],
                    door_size[2] * (1 - window_size) / 2.0,
                ),
                (door_size[0] * window_size, door_size[1] / 2.0, door_size[2] * window_size),
                (
                    door_size[0] * window_size,
                    door_size[1],
                    door_size[2] * (1 - window_size) / 2.0,
                ),
                (door_size[0] * (1 - window_size) / 2.0, door_size[1], door_size[2]),
            ]
            window_translations = [
                (
                    door_xyz[0] + door_size[0] * (window_size / 4.0 + 0.25),
                    door_xyz[1],
                    door_xyz[2],
                ),
                (
                    door_xyz[0],
                    door_xyz[1],
                    door_xyz[2] + door_size[2] * (window_size / 4.0 + 0.25),
                ),
                door_xyz,
                (
                    door_xyz[0],
                    door_xyz[1],
                    door_xyz[2] - door_size[2] * (window_size / 4.0 + 0.25),
                ),
                (
                    door_xyz[0] - door_size[0] * (window_size / 4.0 + 0.25),
                    door_xyz[1],
                    door_xyz[2],
                ),
            ]
            for name, translation, size in zip(window_names, window_translations, window_sizes):
                door_link.visuals.append(
                    yourdfpy.Visual(
                        name=name,
                        origin=tra.translation_matrix(translation),
                        geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=size)),
                    )
                )
        else:
            # No window
            door_link.visuals.append(
                yourdfpy.Visual(
                    name="door_front",
                    origin=tra.translation_matrix(door_xyz),
                    geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=door_size)),
                )
            )
        door_link.visuals.append(
            yourdfpy.Visual(
                name="door_top",
                origin=tra.translation_matrix(
                    [-0.16037736 * width, 0.02580645 * depth, 0.64347826 * height]
                ),
                geometry=yourdfpy.Geometry(
                    box=yourdfpy.Box(size=[door_size[0], 0.30967742 * depth, 0.0173913 * height])
                ),
            )
        )
        model.links.append(door_link)

        if handle_length > door_size[2]:
            raise ValueError(
                f"handle_length={handle_length} is larger than the height of the"
                f" door={door_size[2]}. Either increase height or decrease handle_length."
            )

        handle_link = _create_handle_link(
            name="handle",
            inertial=None,
            handle_width=handle_length,
            handle_depth=handle_thickness,
            handle_height=handle_depth,
            handle_offset=None,  # TODO: Remove, not used
            handle_shape_args=handle_shape_args,
        )
        for v in handle_link.visuals:
            v.origin = v.origin @ tra.compose_matrix(
                translate=(
                    -door_xyz[2] + door_size[2] / 2.0 - handle_length / 2.0 - handle_offset[1],
                    door_xyz[1] + 2 * handle_depth - handle_thickness,
                    door_xyz[0] + door_size[0] / 2.0 - handle_thickness / 2.0 + handle_offset[0],
                ),
                angles=(0, 0, np.pi),
            )
            door_link.visuals.append(v)

        # copy all visuals to collisions
        for l in model.links:
            for v in l.visuals:
                l.collisions.append(
                    yourdfpy.Collision(
                        name=v.name + "_collision",
                        origin=v.origin.copy(),
                        geometry=copy.deepcopy(v.geometry),
                    )
                )

        model.joints.append(
            yourdfpy.Joint(
                name="door_joint",
                type="prismatic",
                parent=body_link.name,
                child=door_link.name,
                origin=np.eye(4),
                axis=np.array([1.0, 0, 0]),
                limit=yourdfpy.Limit(effort=1000.0, velocity=100.0, lower=-1.7, upper=0.0),
            )
        )

        if button_size is not None and button_size > 0:
            # Add button joints
            for bl in button_links:
                model.joints.append(
                    yourdfpy.Joint(
                        name=f"{bl.name}_joint",
                        type="prismatic",
                        parent=body_link.name,
                        child=bl.name,
                        origin=np.eye(4),
                        axis=np.array([0, 1.0, 0]),
                        limit=yourdfpy.Limit(
                            effort=1000.0,
                            velocity=100.0,
                            lower=-button_big_depth if "big" in bl.name else -button_depth,
                            upper=0.0,
                        ),
                    )
                )

        return model

class LeverSwitchAsset(URDFAsset):
    """A lever switch asset."""

    def __init__(self, 
            lever_length,
            lever_width=None,
            lever_depth=None,
            lever='box',
            tip='cylinder',
            tip_size=None,
            tip_width=None,
            base=None,
            base_extents=None,
            joint_limit_lower=-0.7853981633974483,
            joint_limit_upper=0.7853981633974483,
            joint_limit_velocity=100.0,
            joint_limit_effort=1000.0,
            **kwargs
        ):
        """A lever switch with a revolute joint.

        .. image:: /../imgs/lever_switch_asset.png
            :align: center
            :width: 250px

        Args:
            lever_length (float): Length of the lever.
            lever_width (float, optional): Width of the lever (when shaped as a box - diameter in case of a cylindrical shape). If None will be one fifth of the length. Defaults to None.
            lever_depth (float, optional): Depth of the lever (when shaped as a box). If None will be one tenth of the length. Defaults to None.
            lever (str, optional): Shape of the lever. Either 'box' or 'cylinder'. Defaults to 'box'.
            tip (str, optional): Shape of the tip of the lever. Either 'cylinder', 'sphere', or None. Defaults to 'cylinder'.
            tip_size (float, optional): Radius of the tip. If None lever_width will be used. Defaults to None.
            tip_width (float, optional): Width of the tip. Only used if tip == 'cylinder'. If None will be the same as lever_depth. Defaults to None.
            base (bool, optional): If True will create a base shaped like a box in which the lever sits. Defaults to None.
            base_extents (tuple[float], optional): A 3-tuple representing the size of the base box. If None the extents will be based on the lever size. Defaults to None.
            joint_limit_lower (float, optional): Lower joint limits of the revolute switch joint in radians. Defaults to -0.7853981633974483 (-45deg).
            joint_limit_upper (float, optional): Upper joint limits of the revolute switch joint in radians. Defaults to 0.7853981633974483 (+45deg).
            joint_limit_velocity (float, optional): Joint velocity limit. Defaults to 100.0.
            joint_limit_effort (float, optional): Joint effort limit. Defaults to 1000.0.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=LeverSwitchAsset._create_yourdfpy_model(
                lever_length=lever_length,
                lever_width=lever_width,
                lever_depth=lever_depth,
                lever=lever,
                tip=tip,
                tip_size=tip_size,
                tip_width=tip_width,
                base=base,
                base_extents=base_extents,
                joint_limit_lower=joint_limit_lower,
                joint_limit_upper=joint_limit_upper,
                joint_limit_velocity=joint_limit_velocity,
                joint_limit_effort=joint_limit_effort,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )
        self._configuration = np.zeros(len(self._model.actuated_joint_names))

    @staticmethod
    def _create_yourdfpy_model(
            lever_length,
            lever_width=None,
            lever_depth=None,
            lever='box',
            tip='cylinder',
            tip_size=None,
            tip_width=None,
            base=None,
            base_extents=None,
            joint_limit_lower=-0.7853981633974483,
            joint_limit_upper=+0.7853981633974483,
            joint_limit_velocity=100.0,
            joint_limit_effort=1000.0,
        ):
        allowed_lever_values = ('box', 'cylinder')
        if lever not in allowed_lever_values:
            raise ValueError(f"Argument lever={lever} in LeverSwitchAsset needs to be one of ({', '.join(allowed_lever_values)}).")
        allowed_tip_values = ('sphere', 'cylinder', None)
        if tip not in allowed_tip_values:
            raise ValueError(f"Argument tip={tip} in LeverSwitchAsset needs to be one of ({', '.join(allowed_tip_values)}).")

        model = yourdfpy.Robot(name="LeverSwitch")

        base_link = yourdfpy.Link(name="base")
        lever_link = yourdfpy.Link(name="lever")

        # add lever stick
        if lever == 'box':
            lever_size = [
                lever_width if lever_width is not None else lever_length / 5.0, # Just use some meaningful numbers if not user-specified
                lever_depth if lever_depth is not None else lever_length / 10.0,
                lever_length,
            ]
            lever_link.visuals.append(
                yourdfpy.Visual(
                    name="lever",
                    origin=tra.translation_matrix(
                        [0., 0, lever_length / 2.0]
                    ),
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(
                            size=lever_size
                        )
                    ),
                )
            )
            actual_lever_size = lever_size
        elif lever == 'cylinder':
            diameter = lever_width if lever_width is not None else lever_length / 5.0
            lever_link.visuals.append(
                yourdfpy.Visual(
                    name="lever",
                    origin=tra.translation_matrix(
                        [0., 0, lever_length / 2.0]
                    ),
                    geometry=yourdfpy.Geometry(
                        cylinder=yourdfpy.Cylinder(
                            radius=diameter / 2.0,
                            length=lever_length,
                        )
                    ),
                )
            )
            actual_lever_size = [diameter, diameter, lever_length]

        # add base box
        if base and base is not None:
            if base_extents is not None:
                base_size = base_extents
                if base_size[0] is None:
                    base_size[0] = actual_lever_size[0] / 2.0
                if base_size[1] is None:
                    base_size[1] = actual_lever_size[1] * 2.0
                if base_size[2] is None:
                    base_size[2] = actual_lever_size[2] * 2.0
            else:
                base_size = [actual_lever_size[2] / 2.0, actual_lever_size[1] * 2.0, actual_lever_size[0]]
            
            base_link.visuals.append(
                yourdfpy.Visual(
                    name="base",
                    origin=tra.translation_matrix(
                        [0, 0, 0]
                    ),
                    geometry=yourdfpy.Geometry(
                        box=yourdfpy.Box(
                            size=base_size
                        )
                    ),
                )
            )

        # add lever tip
        if tip == 'sphere':
            if tip_size is None:
                tip_size = actual_lever_size[0]
            lever_link.visuals.append(
                yourdfpy.Visual(
                    name="tip",
                    origin=tra.translation_matrix(
                        [0., 0, lever_length]
                    ),
                    geometry=yourdfpy.Geometry(
                        sphere=yourdfpy.Sphere(
                            radius=tip_size / 2.0,
                        )
                    ),
                )
            )
        elif tip == 'cylinder':
            if tip_size is None:
                tip_size = actual_lever_size[0]
            
            lever_link.visuals.append(
                yourdfpy.Visual(
                    name="tip",
                    origin=tra.compose_matrix(angles=(np.pi/2.0, 0, 0), translate=[0., 0, lever_length]),
                    geometry=yourdfpy.Geometry(
                        cylinder=yourdfpy.Cylinder(
                            radius=tip_size / 2.0,
                            length=tip_width if tip_width is not None else actual_lever_size[1],
                        )
                    ),
                )
            )

        model.links.append(base_link)
        model.links.append(lever_link)

        # copy all visuals to collisions
        for l in model.links:
            for v in l.visuals:
                l.collisions.append(
                    yourdfpy.Collision(
                        name=v.name + "_collision",
                        origin=v.origin.copy(),
                        geometry=copy.deepcopy(v.geometry),
                    )
                )
        
        model.joints.append(
            yourdfpy.Joint(
                name="lever_joint",
                type="revolute",
                parent=base_link.name,
                child=lever_link.name,
                origin=np.eye(4),
                axis=np.array([0, 1.0, 0]),
                limit=yourdfpy.Limit(effort=joint_limit_effort, velocity=joint_limit_velocity, lower=joint_limit_lower, upper=joint_limit_upper),
            )
        )

        return model

class SafetySwitchAsset(URDFAsset):
    """A safety switch asset."""

    def __init__(self,
            fuse_box_width,
            fuse_box_depth,
            fuse_box_height,
            lever_length,
            fuse_box_shape_args={
                "inner_depth_ratio": 1.0,
                "outer_depth_ratio": 0.95,
                "num_depth_sections": 40,
            },
            lever_right_of_box=True,
            lever_width=None,
            lever_depth=None,
            lever='box',
            tip='cylinder',
            tip_size=None,
            tip_width=None,
            base=None,
            base_extents=None,
            joint_limit_lower=-0.7853981633974483,
            joint_limit_upper=0.7853981633974483,
            joint_limit_velocity=100.0,
            joint_limit_effort=1000.0,
            **kwargs
            ):
        """A safety switch attached to a fuse box.

        .. image:: /../imgs/safety_switch_asset.png
            :align: center
            :width: 250px
        
        Args:
            fuse_box_width (float): Width of fuse box.
            fuse_box_depth (float): Depth of fuse box.
            fuse_box_height (float): Height of fuse box.
            lever_length (float): Length of lever.
            fuse_box_shape_args (dict, optional): Shape argument for the fuse box. If None a box primitive will be used. See CabinetDoorAsset for details. Defaults to { "inner_depth_ratio": 1.0, "outer_depth_ratio": 0.95, "num_depth_sections": 40, }.
            lever_right_of_box (bool, optional): Whether the lever is attached to the right or left of the fuse box. Defaults to True.
            lever_width (float, optional): Width of the lever (when shaped as a box - diameter in case of a cylindrical shape). If None will be one fifth of the length. Defaults to None.
            lever_depth (float, optional): Depth of the lever (when shaped as a box). If None will be one tenth of the length. Defaults to None.
            lever (str, optional): Shape of the lever. Either 'box' or 'cylinder'. Defaults to 'box'.
            tip (str, optional): Shape of the tip of the lever. Either 'cylinder', 'sphere', or None. Defaults to 'cylinder'.
            tip_size (float, optional): Radius of the tip. If None lever_width will be used. Defaults to None.
            tip_width (float, optional): Width of the tip. Only used if tip == 'cylinder'. If None will be the same as lever_depth. Defaults to None.
            base (bool, optional): If True will create a base shaped like a box in which the lever sits. Defaults to None.
            base_extents (tuple[float], optional): A 3-tuple representing the size of the base box. If None the extents will be based on the lever size. Defaults to None.
            joint_limit_lower (float, optional): Lower joint limits of the revolute switch joint in radians. Defaults to -0.7853981633974483 (-45deg).
            joint_limit_upper (float, optional): Upper joint limits of the revolute switch joint in radians. Defaults to 0.7853981633974483 (+45deg).
            joint_limit_velocity (float, optional): Joint velocity limit. Defaults to 100.0.
            joint_limit_effort (float, optional): Joint effort limit. Defaults to 1000.0.
            **kwargs: Keyword argument passed onto the URDFAsset constructor.
        """
        self._init_default_attributes(**kwargs)

        self._model = yourdfpy.URDF(
            robot=LeverSwitchAsset._create_yourdfpy_model(
                lever_length=lever_length,
                lever_width=lever_width,
                lever_depth=lever_depth,
                lever=lever,
                tip=tip,
                tip_size=tip_size,
                tip_width=tip_width,
                base=base,
                base_extents=base_extents,
                joint_limit_lower=joint_limit_lower,
                joint_limit_upper=joint_limit_upper,
                joint_limit_velocity=joint_limit_velocity,
                joint_limit_effort=joint_limit_effort,
            ),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )

        if fuse_box_shape_args is None:
            box_mesh_vis = yourdfpy.Geometry(box=yourdfpy.Box(size=(fuse_box_width, fuse_box_depth, fuse_box_height)))
            box_mesh_coll = yourdfpy.Geometry(box=yourdfpy.Box(size=(fuse_box_width, fuse_box_depth, fuse_box_height)))
            box_bounds = np.asarray([np.asarray([fuse_box_width, fuse_box_depth, fuse_box_height]) / -2.0, np.asarray([fuse_box_width, fuse_box_depth, fuse_box_height]) / 2.0])
        else:
            box_mesh = CabinetDoorAsset._create_door_mesh(
                width=fuse_box_width,
                height=fuse_box_height,
                depth=fuse_box_depth,
                **fuse_box_shape_args
            )
            box_mesh_fname = utils.get_random_filename(
                dir=kwargs.get('tmp_mesh_dir', '/tmp'),
                prefix="fuse_box_",
                suffix=".obj",
            )
            box_mesh.export(box_mesh_fname)
            box_mesh_vis = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=box_mesh_fname))
            box_mesh_coll = yourdfpy.Geometry(mesh=yourdfpy.Mesh(filename=box_mesh_fname))

            box_bounds = box_mesh.bounds

        left_or_right = 'top' if lever_right_of_box else 'bottom'
        T = utils.get_reference_frame(self._model.scene.bounds, center_mass=None, centroid=None, x='center', y=left_or_right, z='bottom') @ tra.euler_matrix(-np.pi/2.0, 0, -np.pi/2.0) @ utils.homogeneous_inv(utils.get_reference_frame(box_bounds, center_mass=None, centroid=None, x=left_or_right, y='top', z='center'))

        self._model.link_map['base'].visuals.append(
            yourdfpy.Visual(
                name="fuse_box",
                origin=T,
                geometry=box_mesh_vis,
            )
        )
        self._model.link_map['base'].collisions.append(
            yourdfpy.Visual(
                name="fuse_box_coll",
                origin=T,
                geometry=box_mesh_coll,
            )
        )

        self._configuration = np.zeros(len(self._model.actuated_joint_names))


class MugAsset(TrimeshSceneAsset):
    """A mug asset."""

    def __init__(
        self,
        width=0.1,
        height=0.12,
        thickness=0.005,
        num_sections_width=16,
        num_sections_height=16,
        bottom_flatness_radius=0.03,
        bottom_radius_factor=1.25,
        handle_width=0.015,
        handle_depth=0.04,
        handle_height=0.08,
        handle_num_sections_height=16,
        handle_num_segments_cross_section=8,
        handle_aspect_ratio_cross_section=0.5,
        handle_straight_ratio=1.25,
        handle_curvature_ratio=5.0,
        **kwargs,
    ):
        """A mug.

        .. image:: /../imgs/mug_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the mug. Defaults to 0.1.
            height (float, optional): Height of the mug. Defaults to 0.12.
            thickness (float, optional): Thickness of the mug material. Defaults to 0.005.
            num_sections_width (int, optional): Mesh discretization along width. Defaults to 16.
            num_sections_height (int, optional): Mesh discretization along height. Defaults to 16.
            bottom_flatness_radius (float, optional): Area that should be totally flat at the bottom of the mug. Defaults to 0.03.
            bottom_radius_factor (float, optional): Affects the roundness of the bottom edges of the mug. Defaults to 1.25.
            handle_width (float, optional): Width of the mug handle. Defaults to 0.015.
            handle_depth (float, optional): Depth of the mug handle. Defaults to 0.04.
            handle_height (float, optional): Height of the mug handle. Defaults to 0.08.
            handle_num_sections_height (int, optional): Handle mesh discretization along height. Defaults to 16.
            handle_num_segments_cross_section (int, optional): Handle mesh discretization along depth. Defaults to 8.
            handle_aspect_ratio_cross_section (float, optional): Aspect ratio of mug handle cross-section. Same parameter as in HandleAsset. Defaults to 0.5.
            handle_straight_ratio (float, optional): Same parameter as in HandleAsset. Defaults to 1.25.
            handle_curvature_ratio (float, optional): Same parameter as in HandleAsset. Defaults to 5.0.
        """
        s = trimesh.Scene(
                MugAsset.create_body_mesh(
                    width=width,
                    height=height,
                    thickness=thickness,
                    num_sections_width=num_sections_width,
                    num_sections_height=num_sections_height,
                    bottom_flatness_radius=bottom_flatness_radius,
                    bottom_radius_factor=bottom_radius_factor,
                ),
        )
        handle_geometry = MugAsset.create_handle_mesh(
            width=handle_width,
            depth=handle_depth,
            height=handle_height,
            num_sections_height=handle_num_sections_height,
            num_segments_cross_section=handle_num_segments_cross_section,
            aspect_ratio_cross_section=handle_aspect_ratio_cross_section,
            straight_ratio=handle_straight_ratio,
            curvature_ratio=handle_curvature_ratio,
        )
        s.add_geometry(
                handle_geometry,
                node_name='handle',
                geom_name='handle',
                transform=tra.compose_matrix(
                    angles=(0,  np.pi / 2.0, np.pi / 2.0),
                    translate=(s.bounds[1, 0] + handle_geometry.bounds[1, 1] - handle_width * handle_aspect_ratio_cross_section, 0, height - handle_geometry.extents[0] * 0.6)
                ),
        )
        super().__init__(s, **kwargs)

    @staticmethod
    def create_body_mesh(
        width,
        height,
        thickness,
        num_sections_width,
        num_sections_height,
        bottom_flatness_radius,
        bottom_radius_factor,
    ):
        radius = width / 2.0
        control_points = np.array([[0., 0.], [-radius*bottom_radius_factor, 0.], [-radius, height], [-radius + thickness, height], [-radius*bottom_radius_factor + thickness, thickness], [0., thickness]])
        knots = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]

        bspline = BSpline(t=knots, c=control_points, k=2)
        linestring = bspline(np.linspace(0, 1, num_sections_height))
                             
        if bottom_flatness_radius > 0:
            indices = np.logical_and(np.abs(linestring[:, 0]) < bottom_flatness_radius, np.arange(len(linestring)) <= len(linestring) // 2)
            linestring[indices, 1] = 0.0
        
        m = trimesh.creation.revolve(
            linestring=linestring,
            sections=num_sections_width
        )

        return m

    @staticmethod
    def create_handle_mesh(
        width,
        depth,
        height,
        num_sections_height,
        num_segments_cross_section,
        aspect_ratio_cross_section,
        straight_ratio,
        curvature_ratio,
    ):
        
        poly = trimesh.path.polygons.Polygon(
            _ellipse(
                radius=width / 2.0,
                segments=num_segments_cross_section,
                aspect_ratio=aspect_ratio_cross_section,
            )
        )
        
        path = _create_path(
            width=height,
            depth=depth,
            straight_ratio=straight_ratio,
            curvature_ratio=curvature_ratio,
            num_segments=num_sections_height
        )
        
        m = trimesh.creation.sweep_polygon(poly, path, engine="triangle")

        return m


class GlassAsset(TrimeshSceneAsset):
    """A glass asset."""

    def __init__(
        self,
        width=0.07,
        height=0.16,
        thickness=0.003,
        num_sections_width=16,
        num_sections_height=16,
        bottom_flatness_radius=0.02,
        bottom_radius_factor=1.25,
        **kwargs,
    ):
        """A glass.

        .. image:: /../imgs/glass_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the glass. Defaults to 0.07.
            height (float, optional): Height of the glass. Defaults to 0.16.
            thickness (float, optional): Thickness of the glass. Defaults to 0.003.
            num_sections_width (int, optional): Discretization along width. Defaults to 16.
            num_sections_height (int, optional): Discretization along height. Defaults to 16.
            bottom_flatness_radius (float, optional): Area of the bottom of the glass that should be completely flat. Defaults to 0.02.
            bottom_radius_factor (float, optional): Roundness of the bottom edges of the glass. Defaults to 1.25.
        """
        s = trimesh.Scene(
                MugAsset.create_body_mesh(
                    width=width,
                    height=height,
                    thickness=thickness,
                    num_sections_width=num_sections_width,
                    num_sections_height=num_sections_height,
                    bottom_flatness_radius=bottom_flatness_radius,
                    bottom_radius_factor=bottom_radius_factor,
                ),
        )
        super().__init__(s, **kwargs)


class PlateAsset(TrimeshSceneAsset):
    """A plate asset."""

    def __init__(
        self,
        width=0.3,
        height=0.02,
        thickness=0.005,
        num_sections_width=32,
        num_sections_height=8,
        bottom_flatness_radius=0.02,
        bottom_radius_factor=0.8,
        lip_angle=np.pi/4.0,
        **kwargs,
    ):
        """A plate.

        .. image:: /../imgs/plate_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the plate. Defaults to 0.3.
            height (float, optional): Height of the plate. Defaults to 0.02.
            thickness (float, optional): Thickness of the plate. Defaults to 0.005.
            num_sections_width (int, optional): Mesh discretization along circumference. Defaults to 32.
            num_sections_height (int, optional): Mesh discretization along diameter. Defaults to 8.
            bottom_flatness_radius (float, optional): Area of the bottom of the plate that will be completely flat. Defaults to 0.02.
            bottom_radius_factor (float, optional): Roundness of the plate. Defaults to 0.8.
            lip_angle (float, optional): Angle of the outer lip of the plate. Defaults to np.pi/4.0.
        """
        s = trimesh.Scene(
                PlateAsset.create_mesh(
                    width=width,
                    height=height,
                    thickness=thickness,
                    num_sections_width=num_sections_width,
                    num_sections_height=num_sections_height,
                    bottom_flatness_radius=bottom_flatness_radius,
                    bottom_radius_factor=bottom_radius_factor,
                    lip_angle=lip_angle,
                ),
        )
        super().__init__(s, **kwargs)

    @staticmethod
    def create_mesh(
        width,
        height,
        thickness,
        num_sections_width,
        num_sections_height,
        bottom_flatness_radius,
        bottom_radius_factor,
        lip_angle,
    ):
        radius = width / 2.0
        control_points = np.array([
            [0., 0.],
            [-radius*bottom_radius_factor, 0.],
            [-radius, height - np.cos(lip_angle) * thickness],
            [-radius + np.sin(lip_angle) * thickness, height],
            [-radius*bottom_radius_factor + np.sin(lip_angle/2.0) * thickness, thickness],
            [0., thickness],
        ])
        knots = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]

        bspline = BSpline(t=knots, c=control_points, k=2)
        linestring = bspline(np.linspace(0, 1, num_sections_height))

        if bottom_flatness_radius > 0:
            indices = np.logical_and(np.abs(linestring[:, 0]) < bottom_flatness_radius, np.arange(len(linestring)) <= len(linestring) // 2)
            linestring[indices, 1] = 0.0
        
        m = trimesh.creation.revolve(
            linestring=linestring,
            sections=num_sections_width
        )

        return m


class BowlAsset(TrimeshSceneAsset):
    """A bowl asset."""

    def __init__(
        self,
        width=0.15,
        height=0.06,
        thickness=0.005,
        num_sections_width=32,
        num_sections_height=16,
        bottom_flatness_radius=0.05,
        bottom_radius_factor=0.6,
        lip_angle=np.pi/2.0,
        **kwargs,
    ):
        """A bowl.

        .. image:: /../imgs/bowl_asset.png
            :align: center
            :width: 250px

        Args:
            width (float, optional): Width of the bowl. Defaults to 0.15.
            height (float, optional): Height of the bowl. Defaults to 0.06.
            thickness (float, optional): Thickness of the bowl. Defaults to 0.005.
            num_sections_width (int, optional): Discretization along width. Defaults to 32.
            num_sections_height (int, optional): Discretization along height. Defaults to 16.
            bottom_flatness_radius (float, optional): Area at the bottom of the bowl that will be totally flat. Defaults to 0.05.
            bottom_radius_factor (float, optional): Roundness of the bottom edges. Defaults to 0.6.
            lip_angle (float, optional): Angle of the outer lip of the bowl in rad. Defaults to np.pi/2.0.
        """
        s = trimesh.Scene(
                PlateAsset.create_mesh(
                    width=width,
                    height=height,
                    thickness=thickness,
                    num_sections_width=num_sections_width,
                    num_sections_height=num_sections_height,
                    bottom_flatness_radius=bottom_flatness_radius,
                    bottom_radius_factor=bottom_radius_factor,
                    lip_angle=lip_angle,
                ),
        )
        super().__init__(s, **kwargs)