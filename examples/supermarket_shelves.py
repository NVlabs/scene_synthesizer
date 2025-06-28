import itertools
import random
import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import utils
from scene_synthesizer import datasets

# Load dataset HOPE
# Note that this is just a glorified globber and can be replaced
# by a list of file locations on your disk.
#
# To make the following work, do:
# 1. Download the Hope dataset: https://github.com/swtyree/hope-dataset
# 2. Modify your ~/.config/scene-synth.datasets/config.yaml by adding this entry:
#    - name: 'Hope'
#      parser: Dataset
#      root_dir: <root folder of the downloaded dataset>
#      file_globber: "*/google_16k/textured.obj"
data = datasets.load_dataset("Hope")
mesh_files = data.get_filenames()

# Create a shelf asset
num_boards = 5
shelf = pa.ShelfAsset(
    1, 0.3, 1.4, backboard_thickness=0.02, num_boards=num_boards, num_side_columns=0
)

# Create a scene and use share_geometry=True for improved performance
s = synth.Scene(share_geometry=True)

# Add the shelf to origin of the scene
s.add_object(shelf)

# Subsample Hope object filenames, one for each board on the shelf
selected_mesh_fnames = random.sample(mesh_files, num_boards)

# Iterate through all shelf boards
for i in range(0, num_boards):
    # Identify the support surface on shelf i
    s.label_support(f"surface_{i}", geom_ids=f"shelf/board_{i}")
    
    # Use the mesh version in Hope with less vertices
    fname = selected_mesh_fnames[i].replace('.obj', '_simple.obj')

    # Place the object densely on the shelf
    # We use a grid iterator to replicate the layout commonly found
    # in retail shelves. The place_objects method ensures that no
    # collision between objects happen.
    grid_iter = utils.PositionIteratorGrid(0.02, 0.02)
    s.place_objects(
        obj_id_iterator=utils.object_id_generator(f"obj{i}_"),
        obj_asset_iterator=synth.assets.asset_generator(
            itertools.repeat(fname, 100),
            scale=0.01,
            up=(0, 1, 0),
            front=(0, 0, -1),
            origin=("com", "bottom", "com"),
            align=True,
        ),
        obj_position_iterator=grid_iter,
        obj_orientation_iterator=utils.orientation_generator_const(np.eye(4)),
        obj_support_id_iterator=s.support_generator(support_ids=f"surface_{i}"),
    )

# Export the result to USD
s.export("/tmp/supermarket/shelf.usd")
