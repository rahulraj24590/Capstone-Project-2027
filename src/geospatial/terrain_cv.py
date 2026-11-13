"""
Terrain Computer Vision & Visual Deformation Analysis Module (Phase 8).

Analyzes user-uploaded landscape and slope images for superficial visual patterns:
- Surface tension cracks and fissures (high-gradient dark linear discontinuities).
- Bare soil exposure and debris scar segmentation (vegetative canopy vs earth/rock).
- Surface texture irregularity and hummocky terrain deformation heuristics.
- Annotated visual overlay generation.

IMPORTANT SCIENTIFIC GUARD:
Optical computer vision only detects superficial ground surface features. It cannot
measure subsurface pore water pressure, soil shear strength, or subterranean slip planes.
An image classification result is NOT proof of an active landslide and must NEVER
independently trigger an official warning.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_FORMATS: set[str] = {"JPEG", "JPG", "PNG", "WEBP"}

CV_SCIENTIFIC_DISCLAIMER: str = (
    "EXPERIMENTAL COMPUTER VISION ANALYSIS ONLY: An image classification or visual anomaly "
    "result is NOT proof of an active landslide, slope instability, or imminent failure. "
    "Optical images only reflect superficial ground characteristics and cannot measure "
    "subsurface pore water pressure, shear strength, or subterranean slip planes. "
    "This analysis must NEVER independently trigger an official warning or emergency evacuation."
)


class TerrainCVAnalyzer:
    """Deterministic, pure-Python computer vision analyzer for terrain and slope images."""

    def __init__(self, max_dimension: int = 1024) -> None:
        self.max_dimension = max_dimension

    def validate_image_bytes(self, image_bytes: bytes) -> tuple[bool, str | None, Image.Image | None]:
        """Validate raw image bytes for size, format, and decode safety."""
        if not image_bytes:
            return False, "Empty image payload provided.", None

        if len(image_bytes) > MAX_IMAGE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            return False, f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit of 10.0 MB.", None

        try:
            bio = io.BytesIO(image_bytes)
            img = Image.open(bio)
            fmt = (img.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                return False, f"Unsupported image format '{fmt}'. Allowed formats: {', '.join(sorted(ALLOWED_FORMATS))}.", None

            # Verify integrity by forcing load
            img.verify()

            # Re-open after verify (PIL requirement)
            bio.seek(0)
            img = Image.open(bio)
            img = img.convert("RGB")
            return True, None, img
        except Exception as exc:
            return False, f"Failed to decode image: {str(exc)}", None

    def analyze(self, image_input: bytes | str | Path | Image.Image) -> dict[str, Any]:
        """Perform comprehensive computer vision analysis on a terrain image.

        Args:
            image_input: Raw image bytes, file path, or PIL.Image.

        Returns:
            Dictionary containing detected crack metrics, soil exposure ratios,
            texture heterogeneity, composite visual anomaly score, overlay image,
            and mandatory scientific disclaimer.
        """
        # Resolve PIL Image
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return {
                    "available": False,
                    "error": f"Image file not found: {p}",
                    "disclaimer": CV_SCIENTIFIC_DISCLAIMER,
                }
            with open(p, "rb") as f:
                raw_bytes = f.read()
            valid, err, pil_img = self.validate_image_bytes(raw_bytes)
        elif isinstance(image_input, bytes):
            valid, err, pil_img = self.validate_image_bytes(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            valid, err = True, None
        else:
            return {
                "available": False,
                "error": f"Unsupported input type '{type(image_input).__name__}'. Expected bytes, Path, or PIL.Image.",
                "disclaimer": CV_SCIENTIFIC_DISCLAIMER,
            }

        if not valid or pil_img is None:
            return {
                "available": False,
                "error": err or "Invalid image input.",
                "disclaimer": CV_SCIENTIFIC_DISCLAIMER,
            }

        # Downscale if exceeding max_dimension
        orig_w, orig_h = pil_img.size
        scale_factor = 1.0
        if max(orig_w, orig_h) > self.max_dimension:
            scale_factor = self.max_dimension / float(max(orig_w, orig_h))
            new_w = max(1, int(orig_w * scale_factor))
            new_h = max(1, int(orig_h * scale_factor))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        rgb_arr = np.asarray(pil_img, dtype=np.float32)
        h, w, _ = rgb_arr.shape
        total_pixels = h * w

        # ── 1. Color Space & Surface Segmentation ─────────────────────────
        r = rgb_arr[:, :, 0]
        g = rgb_arr[:, :, 1]
        b = rgb_arr[:, :, 2]

        # Grayscale luminance (ITU-R 601-2)
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Sky / Cloud mask: upper regions with high brightness and cool bluish/white tone
        # Pixels in the upper 40% of the image with high luminance and B >= G
        y_coords = np.arange(h)[:, None]
        sky_candidate = (gray > 195.0) & (b >= (g - 10.0)) & (b >= (r - 15.0))
        sky_mask = sky_candidate & (y_coords < int(h * 0.45))
        sky_pixel_count = int(np.sum(sky_mask))
        sky_ratio = float(sky_pixel_count / max(1, total_pixels))

        ground_mask = ~sky_mask
        ground_pixel_count = int(np.sum(ground_mask))
        if ground_pixel_count == 0:
            ground_mask = np.ones((h, w), dtype=bool)
            ground_pixel_count = total_pixels

        # Vegetation segmentation: excess green index > threshold
        denom = g + r + 1e-6
        exg = (2.0 * g - r - b) / (g + r + b + 1e-6)
        veg_mask = ground_mask & (g > r * 1.04) & (g > b * 1.04) & (exg > 0.03)
        veg_pixel_count = int(np.sum(veg_mask))
        veg_ratio = float(veg_pixel_count / max(1, ground_pixel_count))

        # Bare soil / rock / scar segmentation:
        # Earth tones (R > B * 1.12, R > 45, G < R * 1.15) OR low-saturation rocky scree
        earth_tones = (r > b * 1.10) & (r > 40.0) & (r >= g * 0.85) & ~veg_mask
        scree_tones = (np.abs(r - g) < 22.0) & (np.abs(g - b) < 22.0) & (gray > 40.0) & (gray < 185.0) & ~veg_mask
        soil_mask = ground_mask & (earth_tones | scree_tones)
        soil_pixel_count = int(np.sum(soil_mask))
        soil_ratio = float(soil_pixel_count / max(1, ground_pixel_count))

        if soil_ratio < 0.15:
            exposure_class = "Low Exposure (< 15%)"
        elif soil_ratio <= 0.40:
            exposure_class = "Moderate Exposure (15–40%)"
        else:
            exposure_class = "High Exposure / Scar (> 40%)"

        # ── 2. Surface Crack & Fissure Detection ──────────────────────────
        # Compute horizontal & vertical Sobel gradients
        gx = ndi.sobel(gray, axis=1)
        gy = ndi.sobel(gray, axis=0)
        grad_mag = np.hypot(gx, gy)

        # Local background luminance filter (cracks are locally darker than surrounding soil)
        local_mean = ndi.uniform_filter(gray, size=15)
        dark_contrast = (local_mean - gray) > 10.0

        # Gradient threshold: relative to ground standard deviation
        ground_grads = grad_mag[ground_mask]
        grad_thresh = float(np.mean(ground_grads) + 1.4 * np.std(ground_grads)) if len(ground_grads) > 0 else 40.0

        raw_crack_mask = ground_mask & dark_contrast & (grad_mag > grad_thresh) & ~veg_mask

        # Morphological closing with 3x3 structure to connect thin fissure line segments
        clean_crack_mask = ndi.binary_closing(raw_crack_mask, structure=np.ones((3, 3), dtype=bool))
        labeled_cracks, num_features = ndi.label(clean_crack_mask, structure=np.ones((3, 3), dtype=bool))

        # Filter components by minimum area (>= 10 connected pixels)
        valid_crack_pixels = 0
        candidate_count = 0
        max_crack_length = 0.0

        if num_features > 0:
            sizes = ndi.sum(np.ones_like(clean_crack_mask, dtype=int), labeled_cracks, range(1, num_features + 1))
            slices = ndi.find_objects(labeled_cracks)
            filtered_crack_mask = np.zeros_like(clean_crack_mask, dtype=bool)

            for idx, (sz, slc) in enumerate(zip(sizes, slices), start=1):
                if sz >= 10 and slc is not None:
                    slc_h = slc[0].stop - slc[0].start
                    slc_w = slc[1].stop - slc[1].start
                    diag = float(np.hypot(slc_h, slc_w))

                    candidate_count += 1
                    valid_crack_pixels += int(sz)
                    max_crack_length = max(max_crack_length, diag)
                    filtered_crack_mask[labeled_cracks == idx] = True
            crack_mask = filtered_crack_mask
        else:
            crack_mask = clean_crack_mask

        crack_pixel_ratio = float(valid_crack_pixels / max(1, ground_pixel_count))
        # Bounded crack severity index [0.0, 1.0]
        crack_severity = float(np.clip(crack_pixel_ratio * 40.0 + (candidate_count / 15.0) * 0.4, 0.0, 1.0))

        # ── 3. Texture Irregularity & Deformation Heuristics ───────────────
        # Hummocky slope deformation creates chaotic high-frequency gradient variances
        if soil_pixel_count > 50:
            soil_grads = grad_mag[soil_mask]
            grad_std = float(np.std(soil_grads))
            texture_roughness = float(np.clip(grad_std / 75.0, 0.0, 1.0))
        else:
            texture_roughness = float(np.clip(np.std(ground_grads) / 90.0, 0.0, 1.0)) if len(ground_grads) > 0 else 0.0

        # Build detected indicator items
        indicators: list[str] = []
        if candidate_count > 0:
            indicators.append(f"Surface tension crack candidates detected ({candidate_count} linear segments, max span ~{max_crack_length:.0f} px).")
        if soil_ratio >= 0.40:
            indicators.append(f"High bare soil/rock exposure ({soil_ratio * 100:.1f}%), indicating possible scarp formation, stripping, or recent slope scar.")
        elif soil_ratio >= 0.20:
            indicators.append(f"Moderate bare soil exposure ({soil_ratio * 100:.1f}%) across slope face.")
        if texture_roughness >= 0.55:
            indicators.append(f"High surface texture irregularity ({texture_roughness:.2f}), consistent with stepped or hummocky terrain deformation.")
        if not indicators:
            indicators.append("No prominent superficial fissures, bare scarps, or anomalous texture irregularities detected.")

        # ── 4. Composite Visual Anomaly Score ──────────────────────────────
        composite_score = float(
            np.clip(
                0.45 * crack_severity
                + 0.35 * min(1.0, soil_ratio / 0.55)
                + 0.20 * texture_roughness,
                0.0,
                1.0,
            )
        )

        if composite_score < 0.22:
            anomaly_tier = "Normal / Stable Appearance"
            severity_color = "#22c55e"
        elif composite_score < 0.45:
            anomaly_tier = "Low Visual Anomaly"
            severity_color = "#38bdf8"
        elif composite_score < 0.70:
            anomaly_tier = "Moderate Visual Anomaly"
            severity_color = "#f59e0b"
        else:
            anomaly_tier = "Significant Visual Anomaly"
            severity_color = "#ef4444"

        # ── 5. Visual Feature Overlay Generation ───────────────────────────
        # Create an annotated RGB image highlighting:
        # - Crack candidates in vivid red/magenta [255, 30, 80]
        # - Bare soil exposure tinted with light amber [245, 158, 11] at 30% alpha
        annotated_rgb = np.array(rgb_arr, dtype=np.uint8)

        # Apply translucent amber overlay on bare soil
        amber_layer = np.zeros_like(annotated_rgb)
        amber_layer[soil_mask] = [245, 158, 11]
        blend_mask = soil_mask[:, :, None]
        annotated_rgb = np.where(blend_mask, (0.75 * annotated_rgb + 0.25 * amber_layer).astype(np.uint8), annotated_rgb)

        # Apply bright red mask over detected crack candidates (dilated slightly for visual pop)
        dilated_crack = ndi.binary_dilation(crack_mask, iterations=1)
        annotated_rgb[dilated_crack] = [239, 68, 68]

        overlay_pil = Image.fromarray(annotated_rgb)
        overlay_bio = io.BytesIO()
        overlay_pil.save(overlay_bio, format="PNG")
        overlay_base64 = base64.b64encode(overlay_bio.getvalue()).decode("ascii")

        return {
            "available": True,
            "image_dimensions": {"width": orig_w, "height": orig_h, "analyzed_width": w, "analyzed_height": h},
            "crack_detection": {
                "crack_candidate_count": candidate_count,
                "crack_pixel_ratio": round(crack_pixel_ratio, 5),
                "max_crack_length_px": round(max_crack_length, 1),
                "crack_severity_index": round(crack_severity, 4),
            },
            "surface_composition": {
                "soil_exposure_ratio": round(soil_ratio, 4),
                "soil_exposure_pct": round(soil_ratio * 100.0, 1),
                "vegetation_cover_ratio": round(veg_ratio, 4),
                "vegetation_cover_pct": round(veg_ratio * 100.0, 1),
                "sky_ratio": round(sky_ratio, 4),
                "exposure_classification": exposure_class,
            },
            "texture_and_deformation": {
                "texture_roughness_score": round(texture_roughness, 4),
                "detected_indicators": indicators,
            },
            "composite_assessment": {
                "anomaly_score": round(composite_score, 4),
                "anomaly_tier": anomaly_tier,
                "severity_color": severity_color,
            },
            "overlay_base64": overlay_base64,
            "disclaimer": CV_SCIENTIFIC_DISCLAIMER,
        }


_analyzer_singleton: TerrainCVAnalyzer | None = None


def get_cv_analyzer() -> TerrainCVAnalyzer:
    """Retrieve or initialize the singleton TerrainCVAnalyzer."""
    global _analyzer_singleton
    if _analyzer_singleton is None:
        _analyzer_singleton = TerrainCVAnalyzer()
    return _analyzer_singleton
