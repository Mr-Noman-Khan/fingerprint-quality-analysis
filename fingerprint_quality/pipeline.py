"""Fingerprint image quality analysis using classical image processing."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

def read_grayscale(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read a non-empty image: {path}")
    return image

def estimate_foreground(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros(gray.shape, dtype=bool)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    mask = (labels == largest).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0

def enhance_fingerprint(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(enhanced, None, h=5, templateWindowSize=7, searchWindowSize=21)

def compute_quality_metrics(gray: np.ndarray) -> dict[str, Any]:
    h, w = gray.shape[:2]
    foreground = estimate_foreground(gray)
    foreground_pixels = int(foreground.sum())
    result = {
        "width": int(w), "height": int(h),
        "foreground_ratio": round(foreground_pixels / float(h*w), 6),
        "laplacian_variance_sharpness_proxy": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
        "global_intensity_std_contrast_proxy": round(float(gray.std()), 6),
    }
    if foreground_pixels >= max(100, int(0.01*h*w)):
        pixels = gray[foreground]
        roi = gray.copy()
        roi[~foreground] = 0
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        result["foreground_intensity_std_contrast_proxy"] = round(float(pixels.std()), 6)
        result["foreground_gradient_magnitude_mean"] = round(float(grad[foreground].mean()), 6)
        result["segmentation_note"] = "Foreground estimated by Otsu threshold and largest connected component."
    else:
        result["foreground_intensity_std_contrast_proxy"] = None
        result["foreground_gradient_magnitude_mean"] = None
        result["segmentation_note"] = "Foreground estimate uncertain; interpret foreground metrics cautiously."
    result["interpretation"] = "Heuristic indicators only; not an official NFIQ score or biometric matching result."
    return result

def create_report_figure(gray: np.ndarray, enhanced: np.ndarray, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = estimate_foreground(gray)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in zip(axes, [gray, enhanced, mask], ["Original", "CLAHE + denoising", "Estimated foreground mask"]):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def process_image(image_path: Path, output_dir: Path) -> dict[str, Any]:
    gray = read_grayscale(image_path)
    enhanced = enhance_fingerprint(gray)
    mask = estimate_foreground(gray)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    mask_path = output_dir / f"{stem}_foreground_mask.png"
    figure_path = output_dir / f"{stem}_comparison.png"
    if not cv2.imwrite(str(enhanced_path), enhanced):
        raise RuntimeError(f"Could not write {enhanced_path}")
    if not cv2.imwrite(str(mask_path), mask.astype(np.uint8)*255):
        raise RuntimeError(f"Could not write {mask_path}")
    metrics = compute_quality_metrics(gray)
    metrics.update({"input_file": image_path.name, "enhanced_file": enhanced_path.name,
                    "foreground_mask_file": mask_path.name, "comparison_figure": figure_path.name})
    create_report_figure(gray, enhanced, figure_path)
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def discover_images(path: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        if path.suffix.lower() not in supported:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported)
    raise FileNotFoundError(f"Input path does not exist: {path}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Enhance fingerprint images and calculate simple quality indicators.")
    parser.add_argument("--input", default="data/sample_images", help="Image file or folder (default: data/sample_images)")
    parser.add_argument("--output", default="outputs", help="Output folder (default: outputs)")
    args = parser.parse_args()
    try:
        images = discover_images(Path(args.input))
        if not images:
            print(f"No supported images found in {args.input}. Add PNG/JPG/TIFF images and try again.")
            return 0
        rows = []
        for path in images:
            metrics = process_image(path, Path(args.output))
            rows.append(metrics)
            print(f"Processed {path}: sharpness proxy={metrics['laplacian_variance_sharpness_proxy']}, "
                  f"contrast proxy={metrics['global_intensity_std_contrast_proxy']}")
        out = Path(args.output)
        fields = sorted({key for row in rows for key in row})
        with (out / "quality_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {out.resolve()}")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
