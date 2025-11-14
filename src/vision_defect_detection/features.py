from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class SegmenterConfig:
    blur_kernel: int = 5
    close_kernel: int = 9
    open_kernel: int = 3
    min_area_ratio: float = 0.002


class Segmenter:
    """Simple foreground extractor tailored to the dataset lighting."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.close_kernel, self.config.close_kernel)
        )
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.open_kernel, self.config.open_kernel)
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.config.blur_kernel, self.config.blur_kernel), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        area_ratio = thresh.mean() / 255.0
        if area_ratio > 0.5:  # flipped foreground/background
            thresh = cv2.bitwise_not(thresh)
            area_ratio = 1.0 - area_ratio
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = areas.argmax() + 1
            mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

        # If the component is too small something went wrong; return empty mask.
        h, w = gray.shape
        if (mask > 0).sum() / (h * w) < self.config.min_area_ratio:
            return np.zeros_like(mask)
        return mask


class FeatureExtractor:
    def __init__(self, segmenter: Segmenter | None = None, target_width: int = 640) -> None:
        self.segmenter = segmenter or Segmenter()
        self.target_width = target_width

    def extract(
        self,
        frame_path: Path,
        mask_path: Optional[Path] = None,
        *,
        use_reference_mask: bool = False,
    ) -> Dict[str, float]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {frame_path}")
        reference_mask = None
        if mask_path and mask_path.exists():
            reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return self.extract_from_array(
            frame,
            reference_mask=reference_mask,
            use_reference_mask=use_reference_mask,
        )

    def extract_from_array(
        self,
        frame: np.ndarray,
        *,
        reference_mask: Optional[np.ndarray] = None,
        use_reference_mask: bool = False,
    ) -> Dict[str, float]:
        frame = self._maybe_resize(frame)
        mask = self.segmenter(frame)

        if reference_mask is not None:
            reference_mask = self._maybe_resize(reference_mask)
        if use_reference_mask and reference_mask is not None:
            mask = reference_mask

        features = self._compute_shape_features(frame, mask)
        if reference_mask is not None:
            features["mask_iou"] = self._mask_iou(mask, reference_mask)
        return features

    def _maybe_resize(self, image: np.ndarray) -> np.ndarray:
        if self.target_width is None or self.target_width <= 0:
            return image
        h, w = image.shape[:2]
        if w <= self.target_width:
            return image
        scale = self.target_width / float(w)
        new_size = (self.target_width, int(round(h * scale)))
        interpolation = cv2.INTER_AREA if image.ndim == 3 else cv2.INTER_NEAREST
        return cv2.resize(image, new_size, interpolation=interpolation)

    @staticmethod
    def _mask_iou(pred: np.ndarray, ref: np.ndarray) -> float:
        pred_bin = pred > 0
        ref_bin = ref > 0
        union = np.logical_or(pred_bin, ref_bin).sum()
        if union == 0:
            return 0.0
        inter = np.logical_and(pred_bin, ref_bin).sum()
        return inter / union

    def _compute_shape_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 0).astype(np.uint8)
        area = mask_bin.sum()
        h, w = gray.shape
        total_area = h * w
        if area == 0:
            return {"area_ratio": 0.0}

        features: Dict[str, float] = {
            "area_ratio": area / total_area,
        }

        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return features
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True) or 1e-6
        area_float = float(area)

        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = bw * bh or 1e-6
        features["bbox_aspect_ratio"] = bw / max(bh, 1)
        features["bbox_coverage_ratio"] = area_float / bbox_area

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) or 1e-6
        features["solidity"] = area_float / hull_area
        features["convex_defect_ratio"] = (hull_area - area_float) / hull_area

        features["circularity"] = 4 * np.pi * area_float / (perimeter**2)
        features["perimeter_area_ratio"] = perimeter / area_float

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw > 0 and rh > 0:
            features["minrect_ratio"] = area_float / (rw * rh)

        roi = mask_bin[y : y + bh, x : x + bw]

        # Hole metrics using contour hierarchy
        hole_count = 0
        hole_area = 0.0
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, info in enumerate(hierarchy):
                parent = info[3]
                if parent >= 0:
                    hole_count += 1
                    hole_area += cv2.contourArea(contours[idx])
        features["hole_count"] = float(hole_count)
        features["hole_area_ratio"] = hole_area / area_float if area_float else 0.0

        # Moments based eccentricity
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]
            cov = np.array([[mu20, mu11], [mu11, mu02]])
            eigvals = np.linalg.eigvalsh(cov)
            eigvals.sort()
            if eigvals[1] > 0:
                features["eccentricity"] = float(np.sqrt(1 - eigvals[0] / eigvals[1]))

        masked_gray = cv2.bitwise_and(gray, gray, mask=mask_bin)
        inside_pixels = masked_gray[mask_bin > 0]
        outside_pixels = gray[mask_bin == 0]
        if inside_pixels.size:
            features["intensity_mean"] = float(inside_pixels.mean())
            features["intensity_std"] = float(inside_pixels.std())
        if outside_pixels.size:
            features["background_mean"] = float(outside_pixels.mean())
            features["background_std"] = float(outside_pixels.std())
        if inside_pixels.size and outside_pixels.size:
            features["fg_bg_contrast"] = float(inside_pixels.mean() - outside_pixels.mean())

        # Edge-based features
        edges = cv2.Canny(masked_gray, 50, 150)
        edge_pixels = edges[mask_bin > 0]
        edge_count = int((edge_pixels > 0).sum())
        features["edge_density"] = edge_count / area_float

        lap = cv2.Laplacian(masked_gray, cv2.CV_64F)
        region = lap[mask_bin > 0]
        if region.size:
            features["laplacian_var"] = float(region.var())

        # Gap ratios using morphological closing with different kernels
        features.update(self._gap_features(roi))

        return features

    @staticmethod
    def _gap_features(mask_bin: np.ndarray) -> Dict[str, float]:
        features: Dict[str, float] = {}
        area_float = float(mask_bin.sum()) or 1.0
        for size in (5, 11, 19):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
            gap = max(0, int(closed.sum()) - int(area_float))
            features[f"gap_ratio_{size}"] = gap / area_float
        return features
