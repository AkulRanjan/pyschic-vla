"""Every module in the package imports cleanly (catches broken stubs early)."""

import importlib
import pkgutil

import thoughtzero


def test_all_modules_import() -> None:
    names = [m.name for m in pkgutil.walk_packages(thoughtzero.__path__, prefix="thoughtzero.")]
    assert len(names) > 30
    for name in names:
        importlib.import_module(name)
