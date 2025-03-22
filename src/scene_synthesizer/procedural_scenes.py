# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Third Party
import numpy as np
import trimesh.transformations as tra

# Local Folder
from . import procedural_assets, utils
from .scene import Scene
from .assets import BoxAsset, LPrismAsset


def _permute_main_kitchen_assets(rng, fridge, oven, sink_cabinet):
    # sequence of kitchen furniture
    not_fridge_items = [oven, sink_cabinet]
    rng.shuffle(not_fridge_items)

    # ensure fridge is on the outside
    order_of_assets = [not_fridge_items[0], fridge]
    rng.shuffle(order_of_assets)
    order_of_assets.insert(1, not_fridge_items[-1])

    return order_of_assets


def _add_asset_next_to(rng, asset, next_to, order_of_assets):
    next_to_idx = order_of_assets.index(next_to)
    asset_idx = rng.choice([next_to_idx, next_to_idx + 1])
    order_of_assets.insert(asset_idx, asset)

    return order_of_assets

def use_primitives_only():
    """Helper function that creates arguments for the procedural kitchen scenes such that they are generated only from shape primitives.

    Example usage:

    .. code-block:: python

        from scene_synthesizer.procedural_scenes import kitchen_single_wall, use_primitives_only
        kitchen_single_wall(**use_primitives_only())


    Returns:
        dict: A dictionary of keyword arguments.
    """
    return {
        'handle_shape_args': None,
        'door_shape_args': None,
        'refrigerator_args': {
            'door_shape_args': None,
            'handle_shape_args': None
        },
        'dishwasher_args': {
            'handle_shape_args': None
        },
        'range_args': {
            'handle_shape_args': None
        },
        'range_hood_args': {
            'use_primitives': True
        }
    }

def kitchen_single_wall(
    seed=None, **kwargs
):
    """Kitchen scene with a ⎮-shaped counter space.

    .. image:: /../imgs/kitchen_single_wall.png
        :align: center
        :width: 250px

    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optinal): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))
    counter_height_without_top = counter_height - counter_thickness

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )
    
    refrigerator_args = kwargs.get("refrigerator_args", {})
    fridge = procedural_assets.RefrigeratorAsset.random(
        seed=rng,
        **refrigerator_args,
    )

    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    base_cabinet = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
    )
    base_cabinet_extents = base_cabinet.get_extents()

    range_args = kwargs.get("range_args", {})
    range_args["height"] = range_args.get("height", counter_height + 0.03)
    range_args["depth"] = range_args.get("depth", counter_depth)
    oven = procedural_assets.RangeAsset.random(
        rng,
        **range_args,
    )

    range_hood_args = kwargs.get("range_hood_args", {})
    range_hood = procedural_assets.RangeHoodAsset.random(rng, **range_hood_args)

    sink_cabinet = procedural_assets.SinkCabinetAsset.random(
        seed=rng,
        height=counter_height,
        depth=counter_depth,
        countertop_thickness=counter_thickness,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
    )

    dishwasher_args = kwargs.get("dishwasher_args", {})
    dishwasher_args["height"] = dishwasher_args.get("height", counter_height_without_top)
    dishwasher_args["depth"] = dishwasher_args.get("depth", counter_depth)
    dishwasher = procedural_assets.DishwasherAsset.random(
        rng,
        **dishwasher_args,
    )
    dishwasher_extents = dishwasher.get_extents()

    wall_cabinet_1 = procedural_assets.WallCabinetAsset.random(
        seed=rng,
        width=dishwasher_extents[0] + sink_cabinet.get_extents()[0] - 0.2,
        depth=counter_depth / 2.0,
        height=wall_cabinet_height,
        compartment_types=rng.choice(["door_right", "door_left"], 3),
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
    )
    wall_cabinet_2 = procedural_assets.WallCabinetAsset.random(
        seed=rng,
        width=base_cabinet_extents[0] - 0.2,
        depth=counter_depth / 2.0,
        height=wall_cabinet_height,
        compartment_types=["door_right", "door_left"],
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
    )

    order_of_assets = _permute_main_kitchen_assets(
        rng=rng,
        fridge=fridge,
        oven=oven,
        sink_cabinet=sink_cabinet,
    )

    # add dishwasher next to sink
    order_of_assets = _add_asset_next_to(
        rng=rng, asset=dishwasher, next_to=sink_cabinet, order_of_assets=order_of_assets
    )

    # add cabinet next to inside of fridge
    fridge_idx = order_of_assets.index(fridge)
    base_cabinet_idx = 1 if fridge_idx == 0 else -1
    order_of_assets.insert(base_cabinet_idx, base_cabinet)

    s = Scene(seed=rng, keep_collision_manager_synchronized=False)
    s.add_object(order_of_assets[0], str(order_of_assets[0]))
    for i, asset in enumerate(order_of_assets[1:]):
        s.add_object(
            asset,
            str(asset),
            connect_parent_id=str(order_of_assets[i]),
            **utils.right_and_aligned_back_bottom(),
        )

    # add range hood above range
    s.add_object(
        range_hood,
        "range_hood",
        connect_parent_id="range",
        connect_parent_anchor=("center", "top", "bottom"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add wall cabinet above dishwasher / sink_cabinet combo
    s.add_object(
        wall_cabinet_1,
        "wall_cabinet_1",
        connect_parent_id=("sink_cabinet", "dishwasher"),
        connect_obj_anchor=("center", "top", "bottom"),
        connect_parent_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )
    # add wall cabinet above base cabinet
    s.add_object(
        wall_cabinet_2,
        "wall_cabinet_2",
        connect_parent_id="base_cabinet",
        connect_obj_anchor=("center", "top", "bottom"),
        connect_parent_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add countertops
    s.add_object(
        BoxAsset(extents=[dishwasher_extents[0], counter_depth, counter_thickness]),
        "countertop_dishwasher",
        connect_parent_id="dishwasher",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        BoxAsset(extents=[base_cabinet_extents[0], counter_depth, counter_thickness]),
        "countertop_base_cabinet",
        connect_parent_id="base_cabinet",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )

    s.keep_collision_manager_synchronized = True

    return s


def kitchen_galley(
    seed=None, **kwargs
):
    """Kitchen scene with a  ⎢⎟-shaped counter space and an alley between.

    .. image:: /../imgs/kitchen_galley.png
        :align: center
        :width: 250px
    
    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optional): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **aisle_width (float, optional): Width of the center aisle.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to all BaseCabinetAsset.
        **base_cabinet_1_args (dict, optional): Dictionary of parameters that are passed to the first BaseCabinetAsset.
        **base_cabinet_2_args (dict, optional): Dictionary of parameters that are passed to the second BaseCabinetAsset.
        **base_cabinet_3_args (dict, optional): Dictionary of parameters that are passed to the third BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))
    counter_height_without_top = counter_height - counter_thickness

    aisle_width = kwargs.get("aisle_width", rng.uniform(1.0, 1.3))

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )

    refrigerator_args = kwargs.get("refrigerator_args", {})
    fridge = procedural_assets.RefrigeratorAsset.random(
        seed=rng,
        **refrigerator_args,
    )

    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    base_cabinet_1_args = kwargs.get("base_cabinet_1_args", {})
    base_cabinet_1 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
        **base_cabinet_1_args,
    )

    base_cabinet_2_args = kwargs.get("base_cabinet_2_args", {})
    base_cabinet_2 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
        **base_cabinet_2_args,
    )

    base_cabinet_3_args = kwargs.get("base_cabinet_3_args", {})
    base_cabinet_3 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
        **base_cabinet_3_args,
    )

    range_args = kwargs.get("range_args", {})
    range_args["height"] = range_args.get("height", counter_height + 0.03)
    range_args["depth"] = range_args.get("depth", counter_depth)
    oven = procedural_assets.RangeAsset.random(
        rng,
        **range_args,
    )

    range_hood_args = kwargs.get("range_hood_args", {})
    range_hood = procedural_assets.RangeHoodAsset.random(rng, width=oven.get_extents()[0] - 0.05, **range_hood_args)

    sink_cabinet = procedural_assets.SinkCabinetAsset.random(
        seed=rng,
        height=counter_height,
        depth=counter_depth,
        countertop_thickness=counter_thickness,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
    )

    dishwasher_args = kwargs.get("dishwasher_args", {})
    dishwasher_args["height"] = dishwasher_args.get("height", counter_height_without_top)
    dishwasher_args["depth"] = dishwasher_args.get("depth", counter_depth)
    dishwasher = procedural_assets.DishwasherAsset.random(
        rng,
        **dishwasher_args,
    )

    order_of_assets = _permute_main_kitchen_assets(
        rng=rng,
        fridge=fridge,
        oven=oven,
        sink_cabinet=sink_cabinet,
    )

    galley_0 = [order_of_assets[0], order_of_assets[2]]
    galley_1 = [order_of_assets[1]]

    # add dishwasher
    for g in [galley_0, galley_1]:
        if sink_cabinet in g:
            g = _add_asset_next_to(
                rng=rng, asset=dishwasher, next_to=sink_cabinet, order_of_assets=g
            )

    # add base cabinet between fridge and x
    galley_0.insert(-1, base_cabinet_1)

    # add cabinets left and right of central element (sink/range) in second galley
    galley_1.insert(0, base_cabinet_2)
    galley_1.append(base_cabinet_3)

    # mirror randomly
    galley_0, galley_1 = rng.permutation(np.asarray([galley_0, galley_1], dtype=object))

    s = Scene(seed=rng)
    last_obj_id = s.add_object(galley_0[0], str(galley_0[0]))
    for asset in galley_0[1:]:
        last_obj_id = s.add_object(
            asset,
            connect_parent_id=last_obj_id,
            **utils.right_and_aligned_back_bottom(),
        )

    # add second line on the opposite side
    last_obj_id = s.add_object(
        galley_1[0],
        connect_parent_id=last_obj_id,
        connect_parent_anchor=("top", "bottom", "bottom"),
        connect_obj_anchor=("bottom", "bottom", "bottom"),
        transform=tra.rotation_matrix(np.pi, [0, 0, 1])
        @ tra.translation_matrix([0, aisle_width, 0]),
    )
    for asset in galley_1[1:]:
        last_obj_id = s.add_object(
            asset,
            connect_parent_id=last_obj_id,
            **utils.right_and_aligned_back_bottom(),
        )

    # add range hood above range
    s.add_object(
        range_hood,
        "range_hood",
        connect_parent_id="range",
        connect_parent_anchor=("center", "top", "bottom"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add wall cabinets in galley_1
    if oven not in galley_1:
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=sum(asset.get_extents()[0] for asset in galley_1),
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 7),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id=("sink_cabinet", "dishwasher"),
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )
    else:
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=base_cabinet_2.get_extents()[0],
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 3),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id="base_cabinet_0",
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=base_cabinet_3.get_extents()[0],
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 3),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id="base_cabinet_1",
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )

    for width, connect_parent_id in zip((base_cabinet_1.get_extents()[0], base_cabinet_2.get_extents()[0], base_cabinet_3.get_extents()[0], sink_cabinet.get_extents()[0] + dishwasher.get_extents()[0]), ("base_cabinet", "base_cabinet_0", "base_cabinet_1", ("sink_cabinet", "dishwasher"))):
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=width,
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 3 if width > 1.2 else 2),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id="base_cabinet",
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )
    else:
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=base_cabinet_1.get_extents()[0]
            + sink_cabinet.get_extents()[0]
            + dishwasher.get_extents()[0],
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 6),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id=("base_cabinet", "sink_cabinet", "dishwasher"),
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )

    # add countertops
    s.add_object(
        BoxAsset(extents=[dishwasher.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_dishwasher",
        connect_parent_id="dishwasher",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    bc_names = ['base_cabinet', 'base_cabinet_0', 'base_cabinet_1']
    for i, bc in enumerate((base_cabinet_1, base_cabinet_2, base_cabinet_3)):
        s.add_object(
            BoxAsset(extents=[bc.get_extents()[0], counter_depth, counter_thickness]),
            f"countertop_{bc_names[i]}",
            connect_parent_id=bc_names[i],
            connect_parent_anchor=("center", "top", "top"),
            connect_obj_anchor=("center", "top", "bottom"),
        )

    return s


def kitchen_l_shaped(
    seed=None, **kwargs
):
    """Kitchen scene with an L-shaped counter space.

    .. image:: /../imgs/kitchen_l_shaped.png
        :align: center
        :width: 250px
    
    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optinal): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **corner_padding (float, optional): Additional spacing where two furniture fronts meet in a corner.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))
    counter_height_without_top = counter_height - counter_thickness

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )

    refrigerator_args = kwargs.get("refrigerator_args", {})
    fridge = procedural_assets.RefrigeratorAsset.random(
        seed=rng,
        **refrigerator_args,
    )

    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    base_cabinet_1 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
    )

    range_args = kwargs.get("range_args", {})
    range_args["height"] = range_args.get("height", counter_height + 0.03)
    range_args["depth"] = range_args.get("depth", counter_depth)
    oven = procedural_assets.RangeAsset.random(
        rng,
        **range_args,
    )

    range_hood_args = kwargs.get("range_hood_args", {})
    range_hood = procedural_assets.RangeHoodAsset.random(rng, width=oven.get_extents()[0] - 0.05, **range_hood_args)

    sink_cabinet = procedural_assets.SinkCabinetAsset.random(
        seed=rng,
        height=counter_height,
        depth=counter_depth,
        countertop_thickness=counter_thickness,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
    )

    dishwasher_args = kwargs.get("dishwasher_args", {})
    dishwasher_args["height"] = dishwasher_args.get("height", counter_height_without_top)
    dishwasher_args["depth"] = dishwasher_args.get("depth", counter_depth)
    dishwasher = procedural_assets.DishwasherAsset.random(
        rng,
        **dishwasher_args,
    )

    order_of_assets = _permute_main_kitchen_assets(
        rng=rng,
        fridge=fridge,
        oven=oven,
        sink_cabinet=sink_cabinet,
    )

    # add dishwasher next to sink
    order_of_assets = _add_asset_next_to(
        rng=rng, asset=dishwasher, next_to=sink_cabinet, order_of_assets=order_of_assets
    )

    # add cabinet next to inside of fridge
    fridge_idx = order_of_assets.index(fridge)
    base_cabinet_idx = 1 if fridge_idx == 0 else -1
    order_of_assets.insert(base_cabinet_idx, base_cabinet_1)

    # create corner of L-shaped layout
    corner_idx = rng.integers(2, 4)
    corner_padding = kwargs.get('corner_padding', 0.0)
    corner = LPrismAsset(name="corner", extents=[counter_depth + corner_padding, counter_depth + corner_padding, counter_height_without_top], recess=corner_padding)
    
    order_of_assets.insert(corner_idx, corner)

    s = Scene(seed=rng)
    last_obj_id = s.add_object(order_of_assets[0])
    for asset in order_of_assets[1 : corner_idx + 1]:
        last_obj_id = s.add_object(
            asset,
            connect_parent_id=last_obj_id,
            **utils.right_and_aligned_back_bottom(),
        )
    # rotate 90 degrees
    last_obj_id = s.add_object(
        asset=order_of_assets[corner_idx + 1],
        connect_parent_id=[last_obj_id],
        connect_parent_anchor=("top", "bottom", "bottom"),
        connect_obj_anchor=("bottom", "top", "bottom"),
        transform=tra.rotation_matrix(-np.pi / 2.0, [0, 0, 1]),
    )
    for asset in order_of_assets[corner_idx + 2 :]:
        last_obj_id = s.add_object(
            asset,
            connect_parent_id=last_obj_id,
            **utils.right_and_aligned_back_bottom(),
        )

    # add range hood above range
    s.add_object(
        range_hood,
        "range_hood",
        connect_parent_id="range",
        connect_parent_anchor=("center", "top", "bottom"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    for asset in [base_cabinet_1, dishwasher, sink_cabinet]:
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=asset.get_extents()[0],
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 2),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            asset=wall_cabinet,
            connect_parent_id=str(asset),
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )

    # add wall cabinets above corner
    wall_cabinet = procedural_assets.WallCabinetAsset.random(
        seed=rng,
        width=corner.get_extents()[0] / 2.0,
        depth=counter_depth / 2.0,
        height=wall_cabinet_height,
        compartment_types=["door_left"],
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
    )
    s.add_object(
        asset=wall_cabinet,
        connect_parent_id=str(corner),
        connect_obj_anchor=("bottom", "top", "bottom"),
        connect_parent_anchor=("bottom", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )
    wall_cabinet = procedural_assets.WallCabinetAsset.random(
        seed=rng,
        width=corner.get_extents()[0] / 2.0,
        depth=counter_depth / 2.0,
        height=wall_cabinet_height,
        compartment_types=["door_right"],
        up=[0, 0, 1],
        front=[-1, 0, 0],
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
    )
    s.add_object(
        asset=wall_cabinet,
        connect_parent_id=str(corner),
        connect_obj_anchor=("top", "bottom", "bottom"),
        connect_parent_anchor=("top", "bottom", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add countertops
    s.add_object(
        BoxAsset(extents=[dishwasher.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_dishwasher",
        connect_parent_id="dishwasher",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        BoxAsset(extents=[base_cabinet_1.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_base_cabinet",
        connect_parent_id="base_cabinet",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        LPrismAsset(extents=[corner.get_extents()[0], corner.get_extents()[1], counter_thickness], recess=corner_padding),
        "countertop_corner",
        connect_parent_id=str(corner),
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )

    return s


def kitchen_u_shaped(
    seed=None, **kwargs
):
    """Kitchen scene with a U-shaped counter space.

    .. image:: /../imgs/kitchen_u_shaped.png
        :align: center
        :width: 250px
    
    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optinal): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **corner_padding (float, optional): Additional spacing where two furniture fronts meet in a corner.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to all BaseCabinetAsset.
        **base_cabinet_1_args (dict, optional): Dictionary of parameters that are passed to the first BaseCabinetAsset.
        **base_cabinet_2_args (dict, optional): Dictionary of parameters that are passed to the second BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))
    counter_height_without_top = counter_height - counter_thickness

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )

    refrigerator_args = kwargs.get("refrigerator_args", {})
    fridge = procedural_assets.RefrigeratorAsset.random(
        seed=rng,
        **refrigerator_args,
    )

    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    base_cabinet_1_args = kwargs.get("base_cabinet_1_args", {})
    base_cabinet_1 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
        **base_cabinet_1_args,
    )

    base_cabinet_2_args = kwargs.get("base_cabinet_2_args", {})
    base_cabinet_2 = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=1.0,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **base_cabinet_args,
        **base_cabinet_2_args,
    )

    range_args = kwargs.get("range_args", {})
    range_args["height"] = range_args.get("height", counter_height + 0.03)
    range_args["depth"] = range_args.get("depth", counter_depth)
    oven = procedural_assets.RangeAsset.random(
        rng,
        **range_args,
    )

    range_hood_args = kwargs.get("range_hood_args", {})
    range_hood = procedural_assets.RangeHoodAsset.random(rng, width=oven.get_extents()[0] - 0.05, **range_hood_args)

    sink_cabinet = procedural_assets.SinkCabinetAsset.random(
        seed=rng,
        height=counter_height,
        depth=counter_depth,
        countertop_thickness=counter_thickness,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
    )

    dishwasher_args = kwargs.get("dishwasher_args", {})
    dishwasher_args["height"] = dishwasher_args.get("height", counter_height_without_top)
    dishwasher_args["depth"] = dishwasher_args.get("depth", counter_depth)
    dishwasher = procedural_assets.DishwasherAsset.random(
        rng,
        **dishwasher_args,
    )

    order_of_assets = _permute_main_kitchen_assets(
        rng=rng,
        fridge=fridge,
        oven=oven,
        sink_cabinet=sink_cabinet,
    )

    # add two box corners between all assets
    corner_padding = kwargs.get('corner_padding', 0.0)
    corner_1 = LPrismAsset(name="corner", extents=[counter_depth + corner_padding, counter_depth + corner_padding, counter_height_without_top], recess=corner_padding)
    corner_2 = LPrismAsset(name="corner", extents=[counter_depth + corner_padding, counter_depth + corner_padding, counter_height_without_top], recess=corner_padding)

    order_of_assets.insert(2, corner_1)
    order_of_assets.insert(1, corner_2)

    # add dishwasher next to sink
    order_of_assets = _add_asset_next_to(
        rng=rng, asset=dishwasher, next_to=sink_cabinet, order_of_assets=order_of_assets
    )

    # add cabinet next to inside of fridge
    fridge_idx = order_of_assets.index(fridge)
    base_cabinet_idx = 1 if fridge_idx == 0 else -1
    order_of_assets.insert(base_cabinet_idx, base_cabinet_1)

    # add cabinet next to oven
    _add_asset_next_to(rng=rng, asset=base_cabinet_2, next_to=oven, order_of_assets=order_of_assets)

    s = Scene(seed=rng)
    last_obj_id = s.add_object(order_of_assets[0])
    for asset in order_of_assets[1:]:
        if "corner" in last_obj_id:
            # create 90 degree corner
            last_obj_id = s.add_object(
                asset=asset,
                connect_parent_id=[last_obj_id],
                connect_parent_anchor=("top", "bottom", "bottom"),
                connect_obj_anchor=("bottom", "top", "bottom"),
                transform=tra.rotation_matrix(-np.pi / 2.0, [0, 0, 1]),
            )
        else:
            last_obj_id = s.add_object(
                asset,
                connect_parent_id=last_obj_id,
                **utils.right_and_aligned_back_bottom(),
            )
    
    # add range hood above range
    s.add_object(
        range_hood,
        "range_hood",
        connect_parent_id="range",
        connect_parent_anchor=("center", "top", "bottom"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add wall cabinets above cabinets
    for asset, obj_id in zip([base_cabinet_1, base_cabinet_2], ["base_cabinet", "base_cabinet_0"]):
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=asset.get_extents()[0],
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=rng.choice(["door_right", "door_left"], 2),
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id=obj_id,
            connect_obj_anchor=("center", "top", "bottom"),
            connect_parent_anchor=("center", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )

    # add wall cabinets above corners
    for asset, obj_id in zip([corner_1, corner_2], ["corner", "corner_0"]):
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=asset.get_extents()[0] / 2.0,
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=["door_left"],
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id=obj_id,
            connect_obj_anchor=("bottom", "top", "bottom"),
            connect_parent_anchor=("bottom", "top", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )
        wall_cabinet = procedural_assets.WallCabinetAsset.random(
            seed=rng,
            width=asset.get_extents()[0] / 2.0,
            depth=counter_depth / 2.0,
            height=wall_cabinet_height,
            compartment_types=["door_right"],
            up=[0, 0, 1],
            front=[-1, 0, 0],
            handle_shape_args=handle_shape_args,
            door_shape_args=door_shape_args,
            **wall_cabinet_args,
        )
        s.add_object(
            wall_cabinet,
            connect_parent_id=obj_id,
            connect_obj_anchor=("top", "bottom", "bottom"),
            connect_parent_anchor=("top", "bottom", "bottom"),
            translation=(0, 0, wall_cabinet_z),
        )

    # add wall cabinet above sink-dishwasher combination
    wall_cabinet = procedural_assets.WallCabinetAsset.random(
        seed=rng,
        width=dishwasher.get_extents()[0] + sink_cabinet.get_extents()[0],
        depth=counter_depth / 2.0,
        height=wall_cabinet_height,
        compartment_types=rng.choice(["door_right", "door_left"], 3),
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
    )
    s.add_object(
        wall_cabinet,
        connect_parent_id=("sink_cabinet", "dishwasher"),
        connect_parent_anchor=("center", "top", "bottom"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, 0, wall_cabinet_z),
    )

    # add countertops
    s.add_object(
        BoxAsset(extents=[dishwasher.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_dishwasher",
        connect_parent_id="dishwasher",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        BoxAsset(extents=[base_cabinet_1.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_base_cabinet",
        connect_parent_id="base_cabinet",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        BoxAsset(extents=[base_cabinet_2.get_extents()[0], counter_depth, counter_thickness]),
        "countertop_base_cabinet_0",
        connect_parent_id="base_cabinet_0",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        LPrismAsset(extents=[corner_1.get_extents()[0], corner_1.get_extents()[1], counter_thickness], recess=corner_padding),
        "countertop_corner_1",
        connect_parent_id="corner",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )
    s.add_object(
        LPrismAsset(extents=[corner_2.get_extents()[0], corner_2.get_extents()[1], counter_thickness], recess=corner_padding),
        "countertop_corner_2",
        connect_parent_id="corner_0",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
    )

    return s


def kitchen_peninsula(
    seed=None, **kwargs
):
    """Kitchen scene with a G-shaped counter space.

    .. image:: /../imgs/kitchen_peninsula.png
        :align: center
        :width: 250px
    
    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optinal): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **corner_padding (float, optional): Additional spacing where two furniture fronts meet in a corner.
        **exit_width (float, optional): Width of the exit.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to all BaseCabinetAsset.
        **base_cabinet_1_args (dict, optional): Dictionary of parameters that are passed to the first BaseCabinetAsset.
        **base_cabinet_2_args (dict, optional): Dictionary of parameters that are passed to the second BaseCabinetAsset.
        **base_cabinet_3_args (dict, optional): Dictionary of parameters that are passed to the third BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))
    counter_height_without_top = counter_height - counter_thickness

    exit_width = kwargs.get("exit_width", rng.uniform(0.7, 0.9))

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )

    refrigerator_args = kwargs.get("refrigerator_args", {})
    dishwasher_args = kwargs.get("dishwasher_args", {})
    range_args = kwargs.get("range_args", {})
    range_hood_args = kwargs.get("range_hood_args", {})
    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    base_cabinet_1_args = kwargs.get("base_cabinet_1_args", {})
    base_cabinet_2_args = kwargs.get("base_cabinet_2_args", {})

    corner_padding = kwargs.get('corner_padding', 0.0)
    s = kitchen_u_shaped(
        seed=rng,
        counter_height=counter_height,
        counter_depth=counter_depth,
        counter_thickness=counter_thickness,
        corner_padding=corner_padding,
        wall_cabinet_z=wall_cabinet_z,
        wall_cabinet_height=wall_cabinet_height,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        refrigerator_args=refrigerator_args,
        dishwasher_args=dishwasher_args,
        range_args=range_args,
        range_hood_args=range_hood_args,
        wall_cabinet_args=wall_cabinet_args,
        base_cabinet_args=base_cabinet_args,
        base_cabinet_1_args=base_cabinet_1_args,
        base_cabinet_2_args=base_cabinet_2_args,
    )

    # this assumes that the order of obj_names is the order of the assets in the spatial layout
    obj_names = list(s.metadata["object_geometry_nodes"].keys())
    last_obj_added = next(
        x
        for x in obj_names[::-1]
        if not x.startswith("wall_cabinet") and not x.startswith("range_hood") and not x.startswith("countertop")
    )

    # add cabinet
    base_cabinet_3_args = kwargs.get("base_cabinet_3_args", {})
    base_cabinet = procedural_assets.BaseCabinetAsset.random(
        seed=rng,
        width=s.scene.extents[1] - 2.0 * counter_depth - exit_width,
        depth=counter_depth,
        height=counter_height_without_top,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        **wall_cabinet_args,
        **base_cabinet_args,
        **base_cabinet_3_args,
    )
    # add corner
    corner = LPrismAsset(name="corner", extents=[counter_depth + corner_padding, counter_depth + corner_padding, counter_height_without_top], recess=corner_padding)
    corner_beginning = False
    if "refrigerator" in last_obj_added:
        # connect to the beginning
        corner_id = s.add_object(
            asset=corner,
            obj_id="corner_3",
            connect_parent_id=obj_names[0],
            **utils.left_and_aligned_back_bottom(),
        )
        s.add_object(
            asset=base_cabinet,
            connect_parent_id=corner_id,
            connect_parent_anchor=("bottom", "bottom", "bottom"),
            connect_obj_anchor=("top", "top", "bottom"),
            transform=tra.rotation_matrix(np.pi / 2.0, [0, 0, 1]),
        )
        corner_beginning = True
    else:
        # connect to the end
        corner_id = s.add_object(
            asset=corner,
            obj_id="corner_3",
            connect_parent_id=last_obj_added,
            **utils.right_and_aligned_back_bottom(),
        )
        s.add_object(
            asset=base_cabinet,
            connect_parent_id=corner_id,
            connect_parent_anchor=("top", "bottom", "bottom"),
            connect_obj_anchor=("bottom", "top", "bottom"),
            transform=tra.rotation_matrix(-np.pi / 2.0, [0, 0, 1]),
        )
    
    counter_overhang = 0.2
    s.add_object(
        BoxAsset(extents=[base_cabinet.get_extents()[0], counter_depth + counter_overhang, counter_thickness]),
        "countertop_base_cabinet_1",
        connect_parent_id="base_cabinet_1",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(0, counter_overhang, 0.0),
    )
    s.add_object(
        LPrismAsset(extents=[corner.get_extents()[0] + counter_overhang, corner.get_extents()[1], counter_thickness], recess=corner_padding),
        "countertop_corner_3",
        connect_parent_id="corner_3",
        connect_parent_anchor=("center", "top", "top"),
        connect_obj_anchor=("center", "top", "bottom"),
        translation=(-counter_overhang / 2.0 if corner_beginning else counter_overhang / 2.0, 0, 0.0),
    )

    return s


def kitchen_island(
    seed=None, **kwargs
):
    """Kitchen scene with an L-shaped counter space and a separate island component.

    .. image:: /../imgs/kitchen_with_island.png
        :align: center
        :width: 250px
    
    Args:
        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        **counter_height (float, optional): Height of counters.
        **counter_depth (float, optinal): Depth of counters.
        **counter_thickness (float, optional): Thickness of counters.
        **corner_padding (float, optional): Additional spacing where two furniture fronts meet in a corner.
        **wall_cabinet_z (float, optional): Z-coordinate of bottom of wall cabinets above ground.
        **wall_cabinet_height (float, optional): Height of cabinets hanging on the wall.
        **handle_shape_args (dict, optional): Dictionary of parameters defining the handle shape.
        **door_shape_args (dict, optional): Dictionary of parameters defining the cabinet door shape.
        **refrigerator_args (dict, optional): Dictionary of parameters that are passed to RefrigeratorAsset.
        **range_args (dict, optional): Dictionary of parameters that are passed to RangeAsset.
        **range_hood_args (dict, optional): Dictionary of parameters that are passed to RangeHoodAsset.
        **dishwasher_args (dict, optional): Dictionary of parameters that are passed to DishwasherAsset.
        **wall_cabinet_args (dict, optional): Dictionary of parameters that are passed to all WallCabinetAsset.
        **base_cabinet_args (dict, optional): Dictionary of parameters that are passed to BaseCabinetAsset.

    Returns:
        scene.Scene: The kitchen scene.
    """
    rng = np.random.default_rng(seed)

    counter_height = kwargs.get("counter_height", rng.uniform(0.7, 0.8))
    counter_depth = kwargs.get("counter_depth", rng.uniform(0.7, 0.8))
    counter_thickness = kwargs.get("counter_thickness", rng.uniform(0.03, 0.05))

    corner_padding = kwargs.get('corner_padding', 0.0)

    wall_cabinet_z = kwargs.get("wall_cabinet_z", rng.uniform(1.25, 1.35))
    wall_cabinet_height = kwargs.get("wall_cabinet_height", rng.uniform(0.7, 0.8))

    handle_shape_args = kwargs.get(
        "handle_shape_args", procedural_assets.HandleAsset.random_shape_params(seed=rng)
    )

    door_shape_args = kwargs.get(
        "door_shape_args", procedural_assets.CabinetDoorAsset.random_shape_params(seed=rng)
    )

    refrigerator_args = kwargs.get("refrigerator_args", {})
    dishwasher_args = kwargs.get("dishwasher_args", {})
    range_args = kwargs.get("range_args", {})
    range_hood_args = kwargs.get("range_hood_args", {})
    wall_cabinet_args = kwargs.get("wall_cabinet_args", {})
    base_cabinet_args = kwargs.get("base_cabinet_args", {})
    s = kitchen_l_shaped(
        seed=rng,
        counter_height=counter_height,
        counter_depth=counter_depth,
        counter_thickness=counter_thickness,
        corner_padding=corner_padding,
        wall_cabinet_z=wall_cabinet_z,
        wall_cabinet_height=wall_cabinet_height,
        handle_shape_args=handle_shape_args,
        door_shape_args=door_shape_args,
        refrigerator_args=refrigerator_args,
        dishwasher_args=dishwasher_args,
        range_args=range_args,
        range_hood_args=range_hood_args,
        wall_cabinet_args=wall_cabinet_args,
        base_cabinet_args=base_cabinet_args,
    )

    # add island
    if s.scene.extents[0] > s.scene.extents[1]:
        kitchen_island = procedural_assets.KitchenIslandAsset(
            width=1.0,
            depth=counter_depth,
            height=counter_height,
            countertop_thickness=counter_thickness,
            up=[0, 0, 1],
            front=[1, 0, 0],
            handle_shape_args=handle_shape_args, # TODO: Fix handle_offset
            door_shape_args=door_shape_args,
        )
    else:
        kitchen_island = procedural_assets.KitchenIslandAsset(
            width=1.0,
            depth=counter_depth,
            height=counter_height,
            countertop_thickness=counter_thickness,
            handle_shape_args=handle_shape_args, # TODO: Fix handle_offset
            door_shape_args=door_shape_args,
        )

    s.add_object(
        asset=kitchen_island,
        connect_parent_id=None,
        connect_parent_anchor=("bottom", "bottom", "bottom"),
        connect_obj_anchor=("bottom", "top", "bottom"),
        transform=tra.rotation_matrix(np.pi / 2.0, [0, 0, 1]),
    )

    return s


def kitchen(seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    
    fn = (
        kitchen_single_wall,
        kitchen_galley,
        kitchen_l_shaped,
        kitchen_u_shaped,
        kitchen_peninsula,
        kitchen_island,
    )

    return rng.choice(fn)(seed=rng, **kwargs)
