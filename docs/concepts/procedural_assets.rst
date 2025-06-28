.. _Procedural Assets:

.. role:: bash(code)
   :language: bash

#####################################
Procedural Assets & Procedural Scenes
#####################################

The submodule :py:mod:`procedural_assets` contains a number of parametric assets such as tables, shelves, and cabinets that can be instantiated and added to a scene.

Furniture and Appliance Assets
------------------------------

.. automodule:: scene_synthesizer.procedural_assets
   :members: BaseCabinetAsset,BinAsset,BoxWithHoleAsset,BowlAsset,CNCMachineAsset,CabinetAsset,CabinetDoorAsset,CubbyShelfAsset,DishRackAsset,DishwasherAsset,GlassAsset,HandleAsset,KitchenIslandAsset,KnobAsset,MicrowaveAsset,MugAsset,RangeAsset,PlateAsset,RangeHoodAsset,RecursivelyPartitionedCabinetAsset,RefrigeratorAsset,ShelfAsset,SinkCabinetAsset,TableAsset,WallCabinetAsset
   :member-order: alphabetical
   :noindex:


Kitchen Scenes
--------------

The submodule :py:mod:procedural_scenes contains six different kitchen types that are procedural generated with the furnitures and appliance assets.

.. automodule:: scene_synthesizer.procedural_scenes
   :members:
   :member-order: alphabetical
   :noindex:

Industrial Assets
-----------------

.. automodule:: scene_synthesizer.procedural_assets
    :member-order: alphabetical
    :members: LeverSwitchAsset, SafetySwitchAsset, HandWheelAsset, CNCMachineAsset
    :noindex:

MotionBenchMaker Assets
------------------------
The MotionBenchMaker paper [MBM2021]_ contains eight nominal scenes used to construct robot motion planning problems:

.. figure:: /../imgs/motionbenchmaker.png
    :align: center

    This is Fig. 6 from [MBM2021]_. It shows eight nominal planning scenes, seven of which can be constructed using the procedural assets in scene_synthesizer. 

The kitchen scene (second-to-last in the figure above) is a URDF and can be loaded as an asset from the `original source <https://github.com/KavrakiLab/motion_bench_maker/blob/main/configs/scenes/kitchen/kitchen.urdf>`_.
In the following we show each procedural asset class that is relevant to the MotionBenchMaker and their parameterization.

.. [MBM2021] Chamzas, Constantinos, Carlos Quintero-Pena, Zachary Kingston, Andreas Orthey, Daniel Rakita,
    Michael Gleicher, Marc Toussaint, and Lydia E. Kavraki. "MotionBenchMaker: A tool to generate and
    benchmark motion planning datasets." IEEE Robotics and Automation Letters 7, no. 2 (2021): 882-889.
    https://github.com/KavrakiLab/motion_bench_maker.

.. automodule:: scene_synthesizer.procedural_assets
   :members: ShelfAsset,TableAsset,TableWithBarsAsset,CageAsset,OpenBoxAsset
   :noindex:

CAD Assets
----------


OpenSCAD
^^^^^^^^
`OpenSCAD <https://openscad.org/>`__ is a free software application for creating solid 3D computer-aided design objects.
It is a script-only based modeller that uses its own description language.
A number of OpenSCAD libraries exists describing anything from electrical to mechanical parts (see https://openscad.org/libraries.html).
To use these parametric models in scene_synthesizer we provide the asset class :py:class:`scene_synthesizer.OpenSCADAsset` which reads `.scad` files.

.. automodule:: scene_synthesizer.assets
   :members: OpenSCADAsset
   :noindex:

First, install `solidpython2 <https://pypi.org/project/solidpython2/>`__ and `openscad <https://openscad.org/downloads.html>`__:

.. code-block:: bash

    python -m pip install solidpython2
    sudo apt-get install openscad

Then download any libraries that contain CAD models you want to use, e.g.:

.. code-block:: bash

    git clone git@github.com:openscad/MCAD.git
    git clone git@github.com:nophead/NopSCADlib.git

We can now use `solidpython2 <https://pypi.org/project/solidpython2/>`_ to create `.scad` files and load them as an asset:

.. code-block:: python

    import solid2
    mcad = solid2.import_scad('MCAD')

    # MCAD contains about 15 separate packages, each included as its own namespace
    print(dir(mcad)) # => ['bearing', 'bitmap', 'boxes', etc...]
    bearing = mcad.bearing.bearing()
    bearing_fname = solid2.scad_render_to_file(bearing, 'bearing.scad')

    # Do the same with NopSCADlib
    nop = solid2.import_scad('NopSCADlib')
    button = nop.vitamins.button.square_button(["button_12mm",  12,  4.0, 0.8, 1.5, 6.8, 4.3, 12.86, 11.44, 8.15, 2.7, 1.4])
    button_fname = solid2.scad_render_to_file(button, 'button.scad')

    # The scad files can then be loaded as assets
    import scene_synthesizer as synth
    bearing_asset = synth.OpenSCADAsset(bearing_fname)
    bearing_asset.scene().export('/tmp/bearing.usd')

    # The scad file can also be used with the generic asset class
    button_asset = synth.Asset(button_fname)
    button_asset.show()

.. figure:: /../imgs/nopscadlib_button.png
    :align: center
    
CadQuery
^^^^^^^^
`CadQuery <https://cadquery.readthedocs.io>`__ is a Python library for building parametric 3D CAD models.
Like OpenSCAD, CadQuery is an open-source, script-based, parametric model generator.
But unlike OpenSCAD the scripts use Python (see `comparison to OpenSCAD <https://cadquery.readthedocs.io/en/latest/intro.html#why-cadquery-instead-of-openscad>`__ for more details).
A number of libraries exist: `cq_warehouse <https://cq-warehouse.readthedocs.io>`__, `cq_gears <https://github.com/meadiode/cq_gears>`__, or 
`cq-electronics <https://cq-electronics.readthedocs.io/>`__.
To use CadQuery models in scene_synthesizer we provide the asset class :py:class:`scene_synthesizer.CQAsset`.

.. automodule:: scene_synthesizer.assets
   :members: CQAsset
   :noindex:

First, install `CadQuery <https://cadquery.readthedocs.io/en/latest/installation.html>`__ and a library such as `cq_warehouse <https://cq-warehouse.readthedocs.io/en/latest/installation.html>`__ [#]_ and `cq_gears <https://github.com/meadiode/cq_gears#installation>`__.
Then use the instantiated CAD models as assets like here:

.. code-block:: python

    import cadquery as cq
    from cq_warehouse import bearing, fastener, thread
    from cq_gears import SpurGear

    skate_board_bearing = bearing.SingleRowDeepGrooveBallBearing(size="M8-22-7", bearing_type="SKT")
    motorcycle_head_bearing = bearing.SingleRowTaperedRollerBearing(size="M20-42-15", bearing_type="SKT")
    bottle_thread = thread.PlasticBottleThread(size='L24SP444')
    spur_gear = SpurGear(module=1.0, teeth_number=19, width=0.1, bore_d=5.0)
    
    all_fasteners = inspect.getmembers(fastener, lambda member: inspect.isclass(member) and member.__module__ == fastener.__name__)

    import scene_synthesizer as synth
    import trimesh.transformations as tra
    s = synth.Scene()
    cnt_y = 0
    for name, fastener_type in all_fasteners:
        if name in ('Nut', 'Screw', 'Washer'):
            continue
        cnt_x = 0
        for t in fastener_type.types():
            size = fastener_type.sizes(t)[0]
            try:
                asset = synth.CQAsset(fastener_type(fastener_type=t, size=size), scale=0.001)
            except TypeError:
                asset = synth.CQAsset(fastener_type(fastener_type=t, size=size, length=10.0), scale=0.001)
            except:
                continue
            s.add_object(asset, f"{name}_{t}_{size}", transform=tra.translation_matrix((cnt_x * 0.02, cnt_y * 0.02, 0)))
            cnt_x = cnt_x + 1
        cnt_y = cnt_y + 1

    s.add_object(synth.CQAsset(spur_gear, scale=0.001), "spur_gear", transform=tra.translation_matrix((-0.04, 0.1, 0)))
    s.add_object(synth.CQAsset(skate_board_bearing, scale=0.001), "skate_board_bearing", transform=tra.translation_matrix((-0.04, 0.14, 0)))
    s.add_object(synth.CQAsset(motorcycle_head_bearing, scale=0.001), "motorcycle_head_bearing", transform=tra.translation_matrix((-0.04, 0.18, 0)))
    s.add_object(synth.CQAsset(bottle_thread, scale=0.001), "bottle_thread", transform=tra.translation_matrix((-0.04, 0.22, 0)))

    s.show()

.. figure:: /../imgs/cadquery_assets.png
    :align: center


.. rubric:: Footnotes

.. [#] In case Python 3.8 is needed there is a version available here: :bash:`python -m pip install git+https://github.com/clemense/cq_warehouse.git@python3.8#egg=cq_warehouse`

