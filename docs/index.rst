#########################################
Scene Synthesizer |version| Documentation
#########################################

A Python package to easily create scenes for robot manipulation tasks.
=======
.. image:: https://joss.theoj.org/papers/10.21105/joss.07561/status.svg
   :target: https://doi.org/10.21105/joss.07561

.. image:: _static/synth_kitchen.png
   :width: 100%

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   GitHub Repository <https://github.com/NVlabs/scene_synthesizer>
   getting_started/statement_of_need.rst
   getting_started/install.rst
   getting_started/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   concepts/assets.rst
   concepts/scenes.rst
   concepts/procedural_assets.rst
   concepts/arranging_objects.rst
   concepts/articulations.rst
   concepts/exporting.rst

..
    Note: when adding new modules, please also update documented_modules.txt.

.. toctree::
    :maxdepth: 1
    :caption: API

    python_api/scene_synthesizer.scene
    python_api/scene_synthesizer.assets
    python_api/scene_synthesizer.procedural_assets
    python_api/scene_synthesizer.procedural_scenes
    python_api/scene_synthesizer.utils
    python_api/scene_synthesizer.exchange.export


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/add_mdl_materials.rst
   examples/isaacsim.rst
   examples/isaaclab_rl.rst
   examples/isaaclab_teleop.rst

Projects Using scene_synthesizer
================================

* `Motion Policy Networks <https://mpinets.github.io/>`__
* `CabiNet: Scaling Neural Collision Detection for Object Rearrangement with Procedural Scene Generation <https://cabinet-object-rearrangement.github.io/>`__
* `Imitating Task and Motion Planning with Visuomotor Transformers <https://mihdalal.github.io/optimus/>`__
* `M2T2: Multi-Task Masked-Transformer for Object-centric Pick and Place <https://m2-t2.github.io/>`__
* `RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics <https://robo-point.github.io/>`__

License
=======

The code is released under the `Apache-2.0 license <https://github.com/NVlabs/scene_synthesizer/blob/main/LICENSE>`__.

How Can I Cite This Library?
============================

..  code-block:: latex

   @article{Eppner2024, 
      title = {scene_synthesizer: A Python Library for Procedural Scene Generation in Robot Manipulation}, 
      author = {Clemens Eppner and Adithyavairavan Murali and Caelan Garrett and Rowland O'Flaherty and Tucker Hermans and Wei Yang and Dieter Fox},
      journal = {Journal of Open Source Software}
      publisher = {The Open Journal}, 
      year = {2024},
   }


.. image:: https://joss.theoj.org/papers/10.21105/joss.07561/status.svg
   :target: https://doi.org/10.21105/joss.07561

