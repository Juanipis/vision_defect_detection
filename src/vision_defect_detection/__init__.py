"""Vision defect detection toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .api import load_default_detector, predict_image
from .inference import DefectDetector

try:  # pragma: no cover - best effort metadata lookup
    __version__ = version("vision-defect-detection")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "DefectDetector", "load_default_detector", "predict_image"]
