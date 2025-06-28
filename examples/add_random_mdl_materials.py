import random

from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.usd_import import get_scene_paths
from scene_synthesizer.exchange.usd_export import add_mdl_material, bind_material_to_prims

# The file name of the final result
usd_filename = "/tmp/kit.usd"
seed = None

random.seed(seed)

DEFAULT_TEXTURE_SCALE = 0.25

url_mdl_material = 'http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/'

# A manually defined dictionary mapping specific materials to types of objects and parts
materials = {
    'sink': [
        ('Base/Stone/Ceramic_Smooth_Fired.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'countertop': [
        ('vMaterials_2/Metal/Copper_Hammered.mdl', 'Copper_Hammered_Shiny', 0.5),
        ('Base/Stone/Marble.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Dark.mdl', 'Granite_Dark', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Light.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Metal/Stainless_Steel_Milled.mdl', 'Stainless_Steel_Milled_Worn', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Slate.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4_Linen.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Ceramic_Tile_12.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Smooth.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Stone_Natural_Black.mdl', 'Stone_Natural_Black_Shiny', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Steel_Grey.mdl', 'Steel_Grey_Bright', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Basaltite.mdl', 'Basaltite_Worn', DEFAULT_TEXTURE_SCALE),
    ],
    'glass': [
        ('Base/Glass/Tinted_Glass_R85.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'tinted glass': [
        ('Base/Glass/Tinted_Glass_R75.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'cabinet': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Ash', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Denim', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Light_Denim', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Bamboo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Cherry.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
    ],
    'rusted metal': [
        ('Base/Metals/RustedMetal.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'glossy black': [
        ('vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl', 'Black', DEFAULT_TEXTURE_SCALE)
    ],
    'handle': [
        ('vMaterials_2/Metal/Silver_Foil.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'appliances': [
        ('vMaterials_2/Metal/Aluminum_Brushed.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'wall': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Pale_Rose', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Lime', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Mint', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Red_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Graphite_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Glazed_Subway', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_Yellow', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Sloppy_Paint_Job', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall_Cracked', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Cappucino', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Antique_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Lime_Green_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Graphite_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Mint_Varied', DEFAULT_TEXTURE_SCALE),
    ],
    'floor': [
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Parquet_Floor.mdl', 'Parquet_Floor', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Concrete/Concrete_Floor_Damage.mdl', 'Concrete_Floor_Damage', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Beech.mdl', 'Wood_Tiles_Beech_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Adobe_Octagon_Dots.mdl', 'Adobe_Octagon_Dots', None),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Mosaic', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terracotta.mdl', 'Terracotta', None),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Square.mdl', 'Ceramic_Tiles_Square_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Paseo.mdl', 'Ceramic_Tiles_Paseo_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
    ],
    'white plastic': [
        ('vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl', 'Plastic_Thick_Translucent', DEFAULT_TEXTURE_SCALE),
    ],
}

# A dictionary mapping scene object/part names to types the general types of objects/parts defined above
# The keys are regular expressions of prim_paths in the USD
geometry2material = {
    "(.*cabinet.*corpus.*|.*cabinet.*door|.*drawer.*board.*|.*cabinet.*closed.*|/world/kitchen_island/.*)|/world/corner.*": "cabinet",
    "(.*refrigerator.*|.*range_hood.*|.*range.*|.*dishwasher.*)": "appliances",
    "/world/range/corpus/heater.*": 'rusted metal',
    "/world/range/corpus/top": 'glossy black',
    ".*/corpus/sink": 'sink',
    ".*countertop.*": "countertop",
    ".*handle.*": "handle",
    "/world/range/.*window": 'tinted glass',
    "(.*glass|.*_window)": 'glass',
    "/world/plate.*": 'sink',
    "(/world/wall/geometry.*|/world/wall_(x|y|_y|_x)/geometry_0)": 'wall',
    "/world/floor/geometry.*": 'floor',
    ".*dishwasher.*basket": 'white plastic',
}

# Generate a random kitchen
kitchen = ps.kitchen(seed=seed)

# Generate UV coordinates for certain primitives
kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')

# Export the scene to a USD stage (in memory, not written to disk yet)
stage = kitchen.export(file_type='usd')

# Attach MDL materials to prims in USD stage
for geom_regex, material_group in geometry2material.items():
    # Find all geometry prims of a particular category
    paths = get_scene_paths(
        stage=stage,
        prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
        scene_path_regex=geom_regex,
    )

    # select random material
    if len(materials[material_group]) == 0:
        print(f"Warning: No materials for {material_group}")
        continue
    
    mtl_url, mtl_name, texture_scale = random.choice(materials[material_group])
    
    # add material to USD stage and bind to geometry prims
    mtl = add_mdl_material(
        stage=stage,
        mtl_url=url_mdl_material + mtl_url,
        mtl_name=mtl_name,
        texture_scale=texture_scale
        )
    bind_material_to_prims(
        stage=stage,
        material=mtl,
        prim_paths=paths
        )

# Export scene to USD file
stage.Export(usd_filename)

