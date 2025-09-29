#!/usr/bin/env python3
"""
Batch demo for depth, foreground, and surface normals.

Adds:
  --img_dir (default: ./data)
  --output_dir (default: ./outputs)

For each image in img_dir, saves per-task PNGs:
  <output_dir>/<stem>_<task>_<mode>.png
where:
  task ∈ {depth, foreground, normal}
  mode ∈ {individual, multitask} depending on which models you provide
"""

import argparse
import os
import sys
from typing import Optional, Dict
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "runtime"))

import cv2
import numpy as np
from tqdm import tqdm

from depth_estimator import RelativeDepthEstimator
from multi_task_estimator import MultiTaskEstimator
from soft_foreground_segmenter import SoftForegroundSegmenter
from surface_normal_estimator import SurfaceNormalEstimator
from visualize import (
    visualize_foreground,
    visualize_normal_maps,
    visualize_relative_depth_map,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_images(folder: str):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if is_image_file(os.path.join(folder, f))
    ]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
def _save_raw_json(arr: np.ndarray, out_path: str, *,
                   task: str, mode: str, src_image: Optional[str] = None):
    """
    Save a numpy array to JSON with minimal metadata.
    NOTE: JSON can be large for big images; consider .npy for compact storage.
    """
    # squeeze trailing singleton channel if present (e.g., HxWx1 -> HxW)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    payload = {
        "image": os.path.basename(src_image) if src_image else None,
        "task": task,
        "mode": mode,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)) if arr.size else None,
        "max": float(np.nanmax(arr)) if arr.size else None,
        "data": arr.tolist(),
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)



def save_task_visuals(
    image_bgr: np.ndarray,
    results: Dict[str, np.ndarray],
    out_dir: str,
    stem: str,
    mode_suffix: str,  # "individual" or "multitask"
    *,
    save_raw: bool = False,         # NEW
    src_image_path: Optional[str] = None,  # NEW
):
    """
    Given raw task arrays in `results` (keys: depth, foreground, normal),
    produce visualizations and write one PNG per task:
      <out_dir>/<stem>_<task>_<mode_suffix>.png
    If save_raw is True, also writes JSON:
      <out_dir>/<stem>_<task>_<mode_suffix>.json
    """
    fg_mask = results.get("foreground")

    if "depth" in results:
        depth_vis = visualize_relative_depth_map(image_bgr, results["depth"], fg_mask)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_depth_{mode_suffix}.png"), depth_vis)
        if save_raw:
            _save_raw_json(
                results["depth"],
                os.path.join(out_dir, f"{stem}_depth_{mode_suffix}.json"),
                task="depth",
                mode=mode_suffix,
                src_image=src_image_path,
            )

    if "foreground" in results:
        foreground_vis = visualize_foreground(image_bgr, results["foreground"])
        cv2.imwrite(os.path.join(out_dir, f"{stem}_foreground_{mode_suffix}.png"), foreground_vis)
        if save_raw:
            _save_raw_json(
                results["foreground"],
                os.path.join(out_dir, f"{stem}_foreground_{mode_suffix}.json"),
                task="foreground",
                mode=mode_suffix,
                src_image=src_image_path,
            )

    if "normal" in results:
        normal_vis = visualize_normal_maps(image_bgr, results["normal"], fg_mask)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_normal_{mode_suffix}.png"), normal_vis)
        if save_raw:
            _save_raw_json(
                results["normal"],
                os.path.join(out_dir, f"{stem}_normal_{mode_suffix}.json"),
                task="normal",
                mode=mode_suffix,
                src_image=src_image_path,
            )

def process_with_individual_models(
    image: np.ndarray,
    depth_model: Optional[str] = None,
    foreground_model: Optional[str] = None,
    normal_model: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run individual models and return raw arrays (no visualization)."""
    results: Dict[str, np.ndarray] = {}

    if depth_model:
        depth_estimator = RelativeDepthEstimator(onnx_model=depth_model, is_inverse=True)
        results["depth"] = depth_estimator.estimate_relative_depth(image)

    if foreground_model:
        foreground_segmenter = SoftForegroundSegmenter(onnx_model=foreground_model)
        results["foreground"] = foreground_segmenter.estimate_foreground_segmentation(
            image
        )

    if normal_model:
        normal_estimator = SurfaceNormalEstimator(onnx_model=normal_model)
        results["normal"] = normal_estimator.estimate_normal(image)

    return results


def process_with_multitask_model(image: np.ndarray, onnx_path: str) -> Dict[str, np.ndarray]:
    multitask_estimator = MultiTaskEstimator(onnx_model=onnx_path, is_inverse_depth=False)
    return multitask_estimator.estimate_all_tasks(image)


def main():
    parser = argparse.ArgumentParser(
        description="Batch depth/foreground/normal estimator that saves per-task files."
    )
    # New batch arguments
    parser.add_argument("--img_dir", type=str, default="./data", help="Folder of input images (non-recursive).")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Folder to write outputs.")

    # Optional: still allow single image for convenience (overrides directory if provided)
    parser.add_argument("--image", help="Path to a single input image (optional).")

    # Model options
    parser.add_argument("--multitask-model", type=str, help="Path to multi-task ONNX model.")
    parser.add_argument("--depth-model", help="Path to depth estimation ONNX model.")
    parser.add_argument("--foreground-model", help="Path to foreground segmentation ONNX model.")
    parser.add_argument("--normal-model", help="Path to surface normal estimation ONNX model.")

    # Progress control (NEW)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--raw-result", action="store_true", help="Also save raw arrays to JSON for each produced task.")

    args = parser.parse_args()

    # Validate model availability
    multitask_available = bool(args.multitask_model and os.path.exists(args.multitask_model))
    depth_available = bool(args.depth_model and os.path.exists(args.depth_model))
    foreground_available = bool(args.foreground_model and os.path.exists(args.foreground_model))
    normal_available = bool(args.normal_model and os.path.exists(args.normal_model))
    individual_available = depth_available or foreground_available or normal_available

    if not (multitask_available or individual_available):
        print("Error: Provide at least one existing model path.")
        print("  --multitask-model OR any of --depth-model/--foreground-model/--normal-model")
        sys.exit(1)

    # Gather images
    images = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        images = [args.image]
    else:
        images = list_images(args.img_dir)
        if not images:
            print(f"No images found in: {args.img_dir}")
            sys.exit(1)

    ensure_dir(args.output_dir)

    # Progress bar config (NEW)
    disable_bar = args.no_progress or not sys.stderr.isatty()
    pbar = tqdm(
        images,
        desc="Processing images",
        unit="img",
        dynamic_ncols=True,
        leave=False,
        disable=disable_bar,
    )

    errors = 0

    # Process each image
    for img_path in pbar:
        pbar.set_postfix_str(os.path.basename(img_path))
        image = cv2.imread(img_path)
        if image is None:
            errors += 1
            if disable_bar:
                print(f"Warning: Could not read image: {img_path}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Individual models
        if individual_available:
            results_ind = process_with_individual_models(
                image,
                depth_model=args.depth_model if depth_available else None,
                foreground_model=args.foreground_model if foreground_available else None,
                normal_model=args.normal_model if normal_available else None,
            )
            if results_ind:
                save_task_visuals(
                    image, results_ind, args.output_dir, stem, "individual",
                    save_raw=args.raw_result, src_image_path=img_path  # NEW
                )
        # Multi-task model
        if multitask_available:
            results_multi = process_with_multitask_model(image, args.multitask_model)
            if results_multi:
                save_task_visuals(
                    image, results_multi, args.output_dir, stem, "multitask",
                    save_raw=args.raw_result, src_image_path=img_path  # NEW
                )
    # Make sure the bar finishes on its own line
    if not disable_bar:
        pbar.close()

    summary = f"Done. Outputs saved under: {os.path.abspath(args.output_dir)}"
    if errors:
        summary += f" (skipped {errors} unreadable image{'s' if errors != 1 else ''})"
    print(summary)


if __name__ == "__main__":
    main()
