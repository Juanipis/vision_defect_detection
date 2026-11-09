from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    min_samples: int = 50
    anomaly_contamination: float = 0.05


def train_group(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    piece_type: str,
    size: str,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> Dict[str, object]:
    config = config or TrainingConfig()
    group_key = f"{piece_type}-{size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{group_key}.joblib"

    if len(df) < config.min_samples:
        raise ValueError(f"Not enough samples for {group_key}: {len(df)} < {config.min_samples}")

    unique_labels = df["is_defective"].unique()
    result: Dict[str, object]
    if len(unique_labels) == 1:
        result = _train_unsupervised(df, feature_cols, piece_type, size, config)
    else:
        result = _train_supervised(df, feature_cols, piece_type, size, config)

    joblib.dump(result, artifact_path)
    return result


def _train_supervised(
    df: pd.DataFrame,
    feature_cols: List[str],
    piece_type: str,
    size: str,
    config: TrainingConfig,
) -> Dict[str, object]:
    X = df[feature_cols].values
    y = df["is_defective"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.1,
        max_iter=400,
        random_state=config.random_state,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "classification_report": report,
        "confusion_matrix": conf,
        "roc_auc": float(roc_auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }

    return {
        "piece_type": piece_type,
        "size": size,
        "model_type": "supervised",
        "feature_names": feature_cols,
        "estimator": model,
        "threshold": 0.5,
        "metrics": metrics,
    }


def _train_unsupervised(
    df: pd.DataFrame,
    feature_cols: List[str],
    piece_type: str,
    size: str,
    config: TrainingConfig,
) -> Dict[str, object]:
    X = df[feature_cols].values
    model = IsolationForest(
        contamination=config.anomaly_contamination,
        random_state=config.random_state,
        n_estimators=400,
    )
    model.fit(X)
    scores = model.decision_function(X)
    threshold = np.quantile(scores, config.anomaly_contamination)

    metrics = {
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "threshold": float(threshold),
        "n_train": int(len(X)),
        "positives": 0,
        "negatives": int(len(X)),
    }

    return {
        "piece_type": piece_type,
        "size": size,
        "model_type": "isolation_forest",
        "feature_names": feature_cols,
        "estimator": model,
        "threshold": float(threshold),
        "metrics": metrics,
    }


def save_metrics_report(metrics: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(list(metrics), fh, indent=2)
