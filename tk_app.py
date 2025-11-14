#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_defect_detection.inference import DefectDetector


class DefectTkApp:
    def __init__(self, detector: DefectDetector, title: str) -> None:
        self.detector = detector
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("900x600")
        self.file_var = tk.StringVar()
        self.piece_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Aún no se ha analizado ninguna imagen")
        self.status_var = tk.StringVar(value="Seleccione una imagen y el tipo de pieza")
        self.features_var = tk.StringVar(value="-")
        self.available_groups = self.detector.available_groups()
        self._preview_img: ImageTk.PhotoImage | None = None
        self._build_ui()
        self._populate_pieces()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # File selector
        file_frame = ttk.Frame(main)
        file_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(file_frame, text="Imagen:").pack(side="left")
        entry = ttk.Entry(file_frame, textvariable=self.file_var, width=60)
        entry.pack(side="left", padx=6, expand=True, fill="x")
        ttk.Button(file_frame, text="Buscar...", command=self._browse_file).pack(side="left")

        # Piece selection
        selection = ttk.Frame(main)
        selection.pack(fill="x", pady=(0, 10))
        ttk.Label(selection, text="Tipo de pieza:").grid(row=0, column=0, sticky="w")
        self.piece_combo = ttk.Combobox(selection, textvariable=self.piece_var, state="readonly")
        self.piece_combo.grid(row=1, column=0, padx=(0, 12), pady=4, sticky="we")
        self.piece_combo.bind("<<ComboboxSelected>>", lambda _: self._update_sizes())

        ttk.Label(selection, text="Tamaño (T):").grid(row=0, column=1, sticky="w")
        self.size_combo = ttk.Combobox(selection, textvariable=self.size_var, state="readonly")
        self.size_combo.grid(row=1, column=1, pady=4, sticky="we")

        selection.columnconfigure(0, weight=1)
        selection.columnconfigure(1, weight=1)

        ttk.Button(selection, text="Analizar", command=self._analyze).grid(row=1, column=2, padx=(12, 0))

        ttk.Separator(main).pack(fill="x", pady=8)

        # Result area
        result_frame = ttk.Frame(main)
        result_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(result_frame, text="Resultado:").pack(anchor="w")
        self.result_label = tk.Label(
            result_frame,
            textvariable=self.result_var,
            font=("Helvetica", 16, "bold"),
            fg="#555555",
        )
        self.result_label.pack(anchor="w", pady=(4, 0))
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor="w")

        ttk.Separator(main).pack(fill="x", pady=8)

        content = ttk.Frame(main)
        content.pack(fill="both", expand=True)

        # Preview panel
        preview_wrapper = ttk.LabelFrame(content, text="Vista previa")
        preview_wrapper.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.preview_label = ttk.Label(preview_wrapper)
        self.preview_label.pack(fill="both", expand=True)

        # Feature summary
        feature_box = ttk.LabelFrame(content, text="Indicadores clave")
        feature_box.pack(side="left", fill="both", expand=True)
        ttk.Label(feature_box, textvariable=self.features_var, justify="left").pack(anchor="nw", padx=6, pady=6)

    def _populate_pieces(self) -> None:
        pieces = sorted(self.available_groups.keys())
        self.piece_combo["values"] = pieces
        if pieces:
            self.piece_var.set(pieces[0])
            self._update_sizes()

    def _update_sizes(self) -> None:
        piece = self.piece_var.get().lower()
        size_map = self.available_groups.get(piece, {})
        sizes = sorted(size_map.keys())
        self.size_combo["values"] = sizes
        if sizes:
            self.size_var.set(sizes[0])
        else:
            self.size_var.set("")

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagen", "*.png;*.jpg;*.jpeg;*.bmp"), ("Todos", "*.*")],
        )
        if not path:
            return
        self.file_var.set(path)
        self._load_preview(Path(path))

    def _load_preview(self, path: Path) -> None:
        try:
            image = Image.open(path)
            image.thumbnail((400, 400))
            self._preview_img = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._preview_img)
        except Exception as exc:  # pragma: no cover - Tk UI feedback
            messagebox.showerror("Error al cargar", f"No se pudo abrir la imagen:\n{exc}")

    def _analyze(self) -> None:
        image_path = Path(self.file_var.get())
        if not image_path.exists():
            messagebox.showwarning("Imagen requerida", "Seleccione una imagen válida antes de analizar.")
            return
        piece = self.piece_var.get()
        size = self.size_var.get()
        if not piece or not size:
            messagebox.showwarning("Falta información", "Seleccione el tipo de pieza y su tamaño.")
            return
        try:
            result = self.detector.predict_from_image(image_path, piece, size)
        except FileNotFoundError:
            messagebox.showerror("Sin modelo", f"No hay modelo para {piece}-{size}")
            return
        except Exception as exc:  # pragma: no cover - runtime feedback
            messagebox.showerror("Error", f"No se pudo analizar la imagen:\n{exc}")
            return

        label = result["label"]
        confidence = result.get("confidence", 0.0) * 100
        self.result_var.set(f"{label} ({confidence:.1f}% confianza)")
        color = "#1b9c85" if label == "BUENO" else "#c1121f"
        self.result_label.configure(fg=color)
        self.status_var.set(
            f"Modelo: {result['model_type']} | Score: {result.get('score', 0.0):.4f}"
        )
        self._update_features(result.get("features", {}))
        self._load_preview(image_path)

    def _update_features(self, features: Dict[str, float]) -> None:
        if not features:
            self.features_var.set("-")
            return
        summary_keys = [
            "area_ratio",
            "solidity",
            "hole_area_ratio",
            "fg_bg_contrast",
            "gap_ratio_11",
            "edge_density",
        ]
        lines = []
        for key in summary_keys:
            if key in features:
                value = features[key]
                lines.append(f"{key}: {value:.4f}")
        self.features_var.set("\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplicación Tkinter para detección de defectos")
    parser.add_argument("--models-dir", default="artifacts/models", help="Directorio con los modelos entrenados")
    parser.add_argument("--title", default="Detección de defectos", help="Título de la ventana")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = DefectDetector(args.models_dir)
    app = DefectTkApp(detector, title=args.title)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
