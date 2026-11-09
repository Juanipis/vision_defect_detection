#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_defect_detection.data import iter_frame_records
from vision_defect_detection.features import FeatureExtractor
from vision_defect_detection.inference import DefectDetector

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa el conjunto etiquetado y genera matrices de confusión")
    parser.add_argument("--data-root", default="processed", help="Directorio raíz del dataset")
    parser.add_argument("--models-dir", default="artifacts/models", help="Directorio con los modelos .joblib")
    parser.add_argument("--cache-file", default="artifacts/cache/frame_features.csv")
    parser.add_argument("--limit-per-sample", type=int, default=None)
    parser.add_argument(
        "--use-reference-mask",
        action="store_true",
        help="Usar la máscara proporcionada en lugar de la segmentación automática",
    )
    parser.add_argument("--figure-path", default="artifacts/models/confusion_matrix.png")
    return parser.parse_args()


def load_feature_table(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = Path(args.cache_file)
    if cache_path.exists():
        print(f"Cargando features desde {cache_path}")
        return pd.read_csv(cache_path)

    print("Cache no encontrada; extrayendo features (puede tardar)...")
    records = iter_frame_records(
        args.data_root,
        include_mask=True,
        limit_per_sample=args.limit_per_sample,
    )
    extractor = FeatureExtractor()
    rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        try:
            feats = extractor.extract(
                record.frame_path,
                record.mask_path,
                use_reference_mask=args.use_reference_mask,
            )
        except Exception as exc:
            print(f"[WARN] Error con {record.frame_path}: {exc}")
            continue
        feats.update(
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
        rows.append(feats)
        if idx % 500 == 0:
            print(f"Procesadas {idx} imágenes...")
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Guardado cache en {cache_path}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []
    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_cols.append(column)
    return sorted(feature_cols)


def evaluate(args: argparse.Namespace) -> None:
    df = load_feature_table(args)
    feature_cols = get_feature_columns(df)
    detector = DefectDetector(args.models_dir)

    evaluations: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        piece = getattr(row, "piece_type")
        size = getattr(row, "size")
        features = {col: getattr(row, col) for col in feature_cols}
        try:
            pred = detector.predict_from_features(features, piece, size)
        except KeyError:
            continue
        evaluations.append(
            {
                "piece_type": piece,
                "size": size,
                "y_true": getattr(row, "is_defective"),
                "y_pred": int(pred["is_defective"]),
            }
        )
    eval_df = pd.DataFrame(evaluations)
    if eval_df.empty:
        raise RuntimeError("No se pudieron generar predicciones; ¿existen modelos y features compatibles?")

    summarize_results(eval_df, args.figure_path)


def summarize_results(eval_df: pd.DataFrame, figure_path: str) -> None:
    labels = [0, 1]
    label_names = ["BUENO", "MALO"]
    cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels)
    print("=== Matriz de confusión global ===")
    print(pd.DataFrame(cm, index=label_names, columns=label_names))
    print("\n=== Métricas globales ===")
    print(classification_report(eval_df["y_true"], eval_df["y_pred"], target_names=label_names, zero_division=0))

    # Per group summary
    print("\n=== Resumen por pieza-tamaño ===")
    for (piece, size), group in eval_df.groupby(["piece_type", "size"]):
        group_cm = confusion_matrix(group["y_true"], group["y_pred"], labels=labels)
        report = classification_report(
            group["y_true"],
            group["y_pred"],
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        acc = report["accuracy"]
        recall_def = report["MALO"]["recall"]
        print(
            f"{piece}-{size} | muestras={len(group)} | acc={acc:.3f} | recall_defectos={recall_def:.3f} | cm={group_cm.tolist()}"
        )

    save_confusion_matrix_figure(cm, label_names, figure_path)


def save_confusion_matrix_figure(matrix: np.ndarray, labels: List[str], figure_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión global")
    fig.tight_layout()
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Figura guardada en {path}")


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
