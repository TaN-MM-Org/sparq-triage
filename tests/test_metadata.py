"""Version consistency between the installed metadata and the package."""
import pathlib
from importlib.metadata import version

import sparq


def test_version_consistent():
    assert sparq.__version__ == version("sparq-triage")


def test_version_matches_citation_file():
    cff = pathlib.Path(__file__).resolve().parents[1] / "CITATION.cff"
    if not cff.exists():  # installed from a wheel, nothing to check
        return
    lines = [l for l in cff.read_text().splitlines() if l.startswith("version:")]
    assert lines == [f"version: {sparq.__version__}"]


def test_core_import_is_torch_free():
    """The top-level package must import without PyTorch installed, so the
    core twin stays usable with the base dependencies alone."""
    import importlib
    import sys
    saved = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}
    for k in saved:
        del sys.modules[k]
    sys.modules["torch"] = None  # any 'import torch' now fails loudly
    try:
        for mod in ("sparq.physics", "sparq.exact", "sparq.pulsed"):
            importlib.reload(importlib.import_module(mod))
    finally:
        del sys.modules["torch"]
        sys.modules.update(saved)
