from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


class PlateOCRPipeline:
    """Détecte les plaques de maison avec YOLO et lit le texte avec EasyOCR."""

    def __init__(
        self,
        model_path: str | Path,
        languages: Iterable[str] = ("en",),
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.reader = easyocr.Reader(list(languages), gpu=device == "cuda")

    @staticmethod
    def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # On applique un seuil pour mieux isoler les caractères.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _read_text_from_crop(self, crop: np.ndarray) -> List[str]:
        if crop is None or crop.size == 0:
            return []

        processed = self._preprocess_crop(crop)
        results = self.reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            contrast_ths=0.1,
            adjust_contrast=0.7,
        )

        texts: List[str] = []
        for _, text, confidence in results:
            cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "-_ ").strip()
            if cleaned and confidence > 0.2:
                texts.append(cleaned)

        return texts

    @staticmethod
    def _box_to_xyxy(box):
        if not hasattr(box, "xyxy"):
            return None

        raw = box.xyxy
        if hasattr(raw, "tolist"):
            raw = raw.tolist()

        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            return [int(v) for v in arr[:4]]
        return None

    def process_frame(self, image: np.ndarray, conf_threshold: float = 0.25):
        if image is None or image.size == 0:
            return None

        results = self.model(image, conf=conf_threshold, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            xyxy = self._box_to_xyxy(box)
            if xyxy is None:
                continue

            x1, y1, x2, y2 = xyxy
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            texts = self._read_text_from_crop(crop)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop,
                    "texts": texts,
                    "confidence": (
                        float(box.conf[0])
                        if hasattr(box.conf, "__len__")
                        else float(box.conf)
                    ),
                }
            )

        return {
            "detections": detections,
            "annotated": result.plot(),
        }

    def detect_and_read(self, image_path: str | Path, conf_threshold: float = 0.25):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image introuvable ou non lisible : {image_path}")

        processed = self.process_frame(image, conf_threshold=conf_threshold)
        if processed is None:
            return (
                [],
                image_path.parent / "detections" / f"{image_path.stem}_detected.jpg",
            )

        detections = processed["detections"]
        annotated = processed["annotated"]

        output_dir = image_path.parent / "detections"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), annotated)

        return detections, output_path

    def process_video(
        self,
        video_source: str | Path | int,
        output_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        show: bool = True,
        max_frames: int | None = None,
        fps: int = 25,
    ):
        capture = cv2.VideoCapture(
            int(video_source) if isinstance(video_source, int) else str(video_source)
        )
        if not capture.isOpened():
            raise ValueError(f"Impossible d’ouvrir la source vidéo : {video_source}")

        frame_width = 640
        frame_height = 480
        if hasattr(capture, "get"):
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )

        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            processed = self.process_frame(frame, conf_threshold=conf_threshold)
            if processed is not None:
                display_frame = processed["annotated"]
                for detection in processed["detections"]:
                    texts = detection.get("texts", [])
                    if texts:
                        print(f"Texte OCR détecté : {texts}")
            else:
                display_frame = frame

            if writer is not None:
                writer.write(display_frame)

            if show:
                cv2.imshow("Plate Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        return frame_count

    def process_camera(
        self,
        camera_index: int = 0,
        conf_threshold: float = 0.25,
        show: bool = True,
        max_frames: int | None = None,
    ):
        return self.process_video(
            camera_index,
            output_path=None,
            conf_threshold=conf_threshold,
            show=show,
            max_frames=max_frames,
        )

    def detect_and_read_directory(
        self, input_dir: str | Path, conf_threshold: float = 0.25
    ):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Dossier introuvable : {input_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ],
            key=lambda p: p.name.lower(),
        )

        results = []
        for image_path in image_paths:
            detections, output_path = self.detect_and_read(
                image_path, conf_threshold=conf_threshold
            )
            results.append(
                {
                    "image": image_path.name,
                    "detections": detections,
                    "output": output_path,
                }
            )

        return results


def _collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        [
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ],
        key=lambda p: p.name.lower(),
    )


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_model = project_root / "models" / "best.pt"

    parser = argparse.ArgumentParser(
        description="Détection de plaques de maisons avec YOLOv8 + OCR EasyOCR"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Chemin d'une image, d'un dossier, d'une vidéo, ou 'camera' pour un flux caméra",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model),
        help="Chemin vers le modèle YOLOv8 (.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Seuil de confiance YOLO"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        help="Langues EasyOCR, ex. --languages en fr",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Périphérique de calcul",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier vidéo de sortie pour le traitement de vidéo",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de la caméra si input='camera'",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre maximum de frames à traiter (utile pour les vidéos/caméra)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Masque la fenêtre de visualisation en direct",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    pipeline = PlateOCRPipeline(
        args.model, languages=args.languages, device=args.device
    )

    if args.input.lower() == "camera":
        print(f"Ouverture de la caméra {args.camera_index}...")
        pipeline.process_camera(
            camera_index=args.camera_index,
            conf_threshold=args.conf,
            show=not args.no_show,
            max_frames=args.max_frames,
        )
        return

    if input_path.exists() and input_path.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }:
        print(f"Traitement vidéo : {input_path}")
        pipeline.process_video(
            input_path,
            output_path=args.output,
            conf_threshold=args.conf,
            show=not args.no_show,
            max_frames=args.max_frames,
        )
        return

    for image_path in _collect_images(input_path):
        print(f"\nTraitement : {image_path}")
        detections, output_path = pipeline.detect_and_read(
            image_path, conf_threshold=args.conf
        )

        if not detections:
            print("Aucune plaque détectée.")
            continue

        print(f"{len(detections)} plaque(s) détectée(s). Image annotée : {output_path}")
        for idx, detection in enumerate(detections, start=1):
            print(
                f"  Plaque {idx} : bbox={detection['bbox']} confiance={detection['confidence']:.3f}"
            )
            plate_texts = detection["texts"]
            if plate_texts:
                print(f"  Texte OCR : {plate_texts}")
            else:
                print("  Texte OCR : aucun résultat")


if __name__ == "__main__":
    main()
