from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .inference import DefectDetector

ImageInput = Union[str, Path, bytes, np.ndarray, Image.Image]


def _default_model_dir() -> os.PathLike[str]:
    from importlib import resources

    return resources.files("vision_defect_detection") / "resources" / "models"


@lru_cache(maxsize=1)
def load_default_detector() -> DefectDetector:
    return DefectDetector(_default_model_dir())


def predict_image(
    image: ImageInput,
    *,
    piece_type: str,
    size: str,
    detector: Optional[DefectDetector] = None,
) -> Dict[str, Any]:
    """Procesa una imagen arbitraria y devuelve el dictamen BUENO/MALO con las métricas clave."""

    detector = detector or load_default_detector()
    frame = _to_bgr_array(image)
    features = detector.feature_extractor.extract_from_array(frame)
    prediction = detector.predict_from_features(features, piece_type, size)
    prediction["features"] = features
    prediction["input_shape"] = frame.shape[:2]
    return prediction


def _to_bgr_array(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, (str, Path)):
        array = cv2.imread(str(image), cv2.IMREAD_COLOR)
    elif isinstance(image, bytes):
        data = np.frombuffer(image, dtype=np.uint8)
        array = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif isinstance(image, Image.Image):
        array = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    if array is None:
        raise ValueError("Could not decode image input.")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array
