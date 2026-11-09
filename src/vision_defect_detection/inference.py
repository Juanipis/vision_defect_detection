from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

from .features import FeatureExtractor


class DefectDetector:
    def __init__(self, model_dir: Path | str, feature_extractor: FeatureExtractor | None = None) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.models = self._load_models()

    def _load_models(self) -> Dict[Tuple[str, str], Dict[str, object]]:
        models: Dict[Tuple[str, str], Dict[str, object]] = {}
        for path in self.model_dir.glob("*.joblib"):
            artifact = joblib.load(path)
            key = (artifact["piece_type"].lower(), artifact["size"].upper())
            models[key] = artifact
        if not models:
            raise RuntimeError(f"No model artifacts found in {self.model_dir}")
        return models

    def available_groups(self) -> Dict[str, Dict[str, str]]:
        summary: Dict[str, Dict[str, str]] = {}
        for (piece_type, size), artifact in self.models.items():
            summary.setdefault(piece_type, {})[size] = artifact["model_type"]
        return summary

    def predict_from_image(self, image_path: Path | str, piece_type: str, size: str) -> Dict[str, object]:
        frame_path = Path(image_path)
        features = self.feature_extractor.extract(frame_path, mask_path=None)
        result = self.predict_from_features(features, piece_type, size)
        result["features"] = features
        result["image_path"] = str(frame_path)
        return result

    def predict_from_features(
        self,
        features: Dict[str, float],
        piece_type: str,
        size: str,
    ) -> Dict[str, object]:
        key = (piece_type.lower(), size.upper())
        if key not in self.models:
            raise KeyError(f"No model available for {piece_type}-{size}")
        artifact = self.models[key]
        feature_names = artifact["feature_names"]
        vector = np.array([[features.get(name, 0.0) for name in feature_names]], dtype=np.float32)
        if artifact["model_type"] == "supervised":
            estimator = artifact["estimator"]
            proba = float(estimator.predict_proba(vector)[0, 1])
            is_defective = int(proba >= artifact.get("threshold", 0.5))
            confidence = proba if is_defective else 1 - proba
            score = proba
        else:
            estimator = artifact["estimator"]
            score = float(estimator.decision_function(vector)[0])
            threshold = artifact.get("threshold", 0.0)
            is_defective = int(score < threshold)
            # Map anomaly score to pseudo probability
            confidence = float(1.0 / (1.0 + np.exp(score - threshold)))
        return {
            "piece_type": piece_type,
            "size": size,
            "is_defective": bool(is_defective),
            "label": "MALO" if is_defective else "BUENO",
            "confidence": confidence,
            "score": score,
            "model_type": artifact["model_type"],
        }
