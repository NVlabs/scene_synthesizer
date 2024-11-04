"""Create scene with table and mugs on it, all in random stable poses.
"""
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa

s = pa.TableAsset(1., 2, 0.7, 0.04, 0.02).scene('table')

s.label_support("table_surface", min_area=0.3)

mug = synth.Asset(
    fname="/home/clemens/code/scene_synthesizer/tests/data/assets/shapenetsem_watertight/Mug/b46a9c35587a38f6e87dab59b7e63362.obj",
    height=0.12,
    origin=("com", "com", "com"),
)

num_mugs = 10
for i in range(num_mugs):
    s.place_object(
        obj_id=f"mug{i}",
        obj_asset=mug,
        support_id="table_surface",
        obj_orientation_iterator=synth.utils.orientation_generator_stable_poses(mug),
    )

s.colorize()
s.show()