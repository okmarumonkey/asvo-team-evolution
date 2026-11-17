"""Team evolution experiments on ASVO."""
from pathlib import Path
import sys

_runtime = Path(__file__).resolve().parents[1] / ".asvo-runtime"
if _runtime.is_dir():
    # ASVO uses both package imports and top-level imports such as NullObservation.
    for _directory in (_runtime, _runtime / "examples/ASVO"):
        if str(_directory) not in sys.path:
            sys.path.insert(0, str(_directory))
