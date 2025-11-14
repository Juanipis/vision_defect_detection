from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

LABEL_PATTERN = re.compile(r"(BUENO|MALO).*?(T\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRecord:
    piece_type: str
    size: str
    label: str
    is_defective: int
    frame_path: Path
    mask_path: Optional[Path]
    sample_id: str
    frame_id: str


def iter_frame_records(
    root: Path | str = Path("processed"),
    *,
    include_mask: bool = True,
    limit_per_sample: Optional[int] = None,
) -> List[FrameRecord]:
    """Collect every frame with its metadata.

    Args:
        root: Dataset root directory.
        include_mask: Whether to attach mask paths when present.
        limit_per_sample: Optional number of frames per sample (useful for quick experiments).
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    records: List[FrameRecord] = []
    for piece_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for sample_dir in sorted(d for d in piece_dir.iterdir() if d.is_dir()):
            match = LABEL_PATTERN.search(sample_dir.name)
            if not match:
                continue
            label, size = match.groups()
            label = label.upper()
            size = size.upper()
            is_defective = 1 if label == "MALO" else 0
            frame_dir = sample_dir / "frame"
            mask_dir = sample_dir / "mask"
            if not frame_dir.exists():
                continue
            frame_files = sorted(frame_dir.glob("*.png"))
            if limit_per_sample:
                frame_files = frame_files[:limit_per_sample]
            for frame_path in frame_files:
                frame_id = frame_path.stem
                mask_path = mask_dir / f"{frame_id}.png" if include_mask else None
                if include_mask and not mask_path.exists():
                    mask_path = None
                records.append(
                    FrameRecord(
                        piece_type=piece_dir.name,
                        size=size,
                        label=label,
                        is_defective=is_defective,
                        frame_path=frame_path,
                        mask_path=mask_path,
                        sample_id=sample_dir.name,
                        frame_id=frame_id,
                    )
                )
    return records


def summarize_by_group(records: Iterable[FrameRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for rec in records:
        key = f"{rec.piece_type}-{rec.size}"
        bucket = summary.setdefault(key, {"BUENO": 0, "MALO": 0, "TOTAL": 0})
        if rec.label not in bucket:
            bucket[rec.label] = 0
        bucket[rec.label] += 1
        bucket["TOTAL"] += 1
    return summary
