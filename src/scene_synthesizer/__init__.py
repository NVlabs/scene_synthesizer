# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene Synthesizer Package."""

# Local Folder
from .assets import (
    Asset,
    BoxAsset,
    BoxMeshAsset,
    BoxWithHoleAsset,
    LPrismAsset,
    SphereAsset,
    CylinderAsset,
    CapsuleAsset,
    CQAsset,
    MeshAsset,
    OpenSCADAsset,
    PlaneAsset,
    TrimeshAsset,
    TrimeshSceneAsset,
    URDFAsset,
    USDAsset,
    MJCFAsset,
    CQAsset,
    OpenSCADAsset,
)
from .scene import Scene


# NOTE (roflaherty): This is inspired by how matplotlib does creates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version():
    """Return the version string used for __version__."""
    # Standard Library
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        # Third Party
        import setuptools_scm

        # See the `setuptools_scm` documentation for the description of the schemes used below.
        # https://pypi.org/project/setuptools-scm/
        # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
        return setuptools_scm.get_version(
            root=root,
            version_scheme="no-guess-dev",
            local_scheme="dirty-tag",
        )
    else:  # Get the version from the _version.py setuptools_scm file.
        try:
            # Standard Library
            from importlib.metadata import version
        except ModuleNotFoundError:
            # NOTE: `importlib.resources` is part of the standard library in Python 3.9.
            # `importlib_metadata` is the back ported library for older versions of python.
            # Third Party
            from importlib_metadata import version

        return version("scene_synthesizer")


# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version
