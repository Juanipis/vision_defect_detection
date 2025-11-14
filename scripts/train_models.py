#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_defect_detection.data import iter_frame_records
from vision_defect_detection.features import FeatureExtractor
from vision_defect_detection.modeling import TrainingConfig, save_metrics_report, train_group

METADATA_COLUMNS = {
    "piece_type",
    "size",
    "label",
    "is_defective",
    "frame_path",
    "mask_path",
    "sample_id",
    "frame_id",
}
EXCLUDE_FEATURE_PREFIXES = ("mask_",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-piece defect detectors")
    parser.add_argument("--data-root", default="processed", help="Dataset root directory")
    parser.add_argument("--output-dir", default="artifacts/models", help="Directory for model artifacts")
    parser.add_argument("--cache-file", default="artifacts/cache/frame_features.csv")
    parser.add_argument("--limit-per-sample", type=int, default=None, help="Limit frames per sample for quick tests")
    parser.add_argument("--use-reference-mask", action="store_true", help="Extract features using provided masks")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--report-file", default="artifacts/models/report.json")
    return parser.parse_args()


def build_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    records = iter_frame_records(
        args.data_root,
        include_mask=True,
        limit_per_sample=args.limit_per_sample,
    )
    extractor = FeatureExtractor()
    rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        try:
            features = extractor.extract(
                record.frame_path,
                record.mask_path,
                use_reference_mask=args.use_reference_mask,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Failed on {record.frame_path}: {exc}")
            continue
        features.update(
            {
                "piece_type": record.piece_type,
                "size": record.size,
                "label": record.label,
                "is_defective": record.is_defective,
                "frame_path": str(record.frame_path),
                "mask_path": str(record.mask_path) if record.mask_path else None,
                "sample_id": record.sample_id,
                "frame_id": record.frame_id,
            }
        )
        rows.append(features)
        if idx % 500 == 0:
            print(f"Processed {idx} frames...")
    df = pd.DataFrame(rows)
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved feature table with {len(df)} rows to {cache_path}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []
    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDE_FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_cols.append(column)
    return sorted(feature_cols)


def main() -> None:
    args = parse_args()
    df = build_dataframe(args)
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} numeric features: {feature_cols}")

    config = TrainingConfig(
        random_state=args.random_state,
        test_size=args.test_size,
        min_samples=args.min_samples,
        anomaly_contamination=args.contamination,
    )

    metrics = []
    output_dir = Path(args.output_dir)
    for (piece_type, size), group_df in df.groupby(["piece_type", "size"]):
        print(f"Training model for {piece_type}-{size} with {len(group_df)} samples...")
        try:
            artifact = train_group(
                group_df,
                feature_cols,
                piece_type=piece_type,
                size=size,
                output_dir=output_dir,
                config=config,
            )
        except ValueError as exc:
            print(f"[WARN] Skipping {piece_type}-{size}: {exc}")
            continue
        metrics.append(
            {
                "piece_type": piece_type,
                "size": size,
                "model_type": artifact["model_type"],
                "n_features": len(feature_cols),
                "metrics": artifact["metrics"],
            }
        )
    report_path = Path(args.report_file)
    save_metrics_report(metrics, report_path)
    print(f"Saved training report to {report_path}")


if __name__ == "__main__":
    main()
