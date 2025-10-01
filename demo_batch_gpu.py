#!/usr/bin/env python3
"""
High-throughput batch demo for depth, foreground, and surface normals.

Key improvements vs original:
- Reuse model sessions (no per-image re-init)
- Optional warmup runs
- Overlap disk I/O with GPU compute (prefetch + async writes)
- Tunable workers/queue and minimal TTY contention
"""

import argparse
import os
import sys
import json
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from threading import Thread

# ── Environment: keep CPU libs from hogging threads ───────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    names = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in names if is_image_file(os.path.join(folder, f))]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_raw_json(arr: np.ndarray, out_path: str, *,
                   task: str, mode: str, src_image: Optional[str] = None):
    """Compact, formatted JSON (beware of size on large images)."""
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
        json.dump(payload, f, ensure_ascii=False, indent=2)

def save_task_visuals(
    image_bgr: np.ndarray,
    results: Dict[str, np.ndarray],
    out_dir: str,
    stem: str,
    mode_suffix: str,                 # "individual" | "multitask"
    *,
    save_raw: bool = False,
    src_image_path: Optional[str] = None,
):
    """Synchronously create visualizations + optional raw dumps."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Model execution (reuse sessions)
# ──────────────────────────────────────────────────────────────────────────────

def process_with_individual_models(
    image: np.ndarray,
    depth_estimator: Optional[RelativeDepthEstimator] = None,
    foreground_segmenter: Optional[SoftForegroundSegmenter] = None,
    normal_estimator: Optional[SurfaceNormalEstimator] = None,
) -> Dict[str, np.ndarray]:
    """Run individual models (single image) using pre-created estimator instances."""
    results: Dict[str, np.ndarray] = {}
    if depth_estimator is not None:
        results["depth"] = depth_estimator.estimate_relative_depth(image)
    if foreground_segmenter is not None:
        results["foreground"] = foreground_segmenter.estimate_foreground_segmentation(image)
    if normal_estimator is not None:
        results["normal"] = normal_estimator.estimate_normal(image)
    return results

def process_with_multitask_model(
    image: np.ndarray,
    multitask_estimator: Optional[MultiTaskEstimator] = None
) -> Dict[str, np.ndarray]:
    if multitask_estimator is None:
        return {}
    return multitask_estimator.estimate_all_tasks(image)


# ──────────────────────────────────────────────────────────────────────────────
# Prefetch & async write plumbing
# ──────────────────────────────────────────────────────────────────────────────

def prefetch_paths(paths: List[str], capacity: int) -> Queue:
    """
    Producer thread that reads images and yields (path, image) pairs in a Queue.
    Keeps a bounded buffer to overlap CPU I/O with GPU compute.
    """
    q: Queue[Tuple[str, Optional[np.ndarray]]] = Queue(maxsize=max(1, capacity))

    def _producer():
        for p in paths:
            img = cv2.imread(p)
            q.put((p, img))
        q.put((None, None))  # sentinel

    t = Thread(target=_producer, daemon=True)
    t.start()
    return q

def consume_prefetch(q: Queue):
    """Iterator over prefetch queue until sentinel."""
    while True:
        p, img = q.get()
        if p is None:
            break
        yield p, img


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU-optimized batch estimator for depth/foreground/normal."
    )
    parser.add_argument("--img_dir", type=str, default="./data",
                        help="Folder of input images (non-recursive).")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Folder to write outputs.")
    parser.add_argument("--image", type=str, help="Path to a single input image (optional).")

    # Models
    parser.add_argument("--multitask-model", type=str, help="Path to multi-task ONNX model.")
    parser.add_argument("--depth-model", type=str, help="Path to depth estimation ONNX model.")
    parser.add_argument("--foreground-model", type=str, help="Path to foreground segmentation ONNX model.")
    parser.add_argument("--normal-model", type=str, help="Path to surface normal estimation ONNX model.")

    # Throughput knobs
    parser.add_argument("--workers", type=int, default=4,
                        help="Async writers (PNG/JSON). 2-8 is typical.")
    parser.add_argument("--prefetch", type=int, default=16,
                        help="How many decoded images to prefetch.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs per enabled pipeline to stabilize kernels.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--raw-result", action="store_true", help="Also save raw arrays to JSON.")

    args = parser.parse_args()

    # Validate models exist
    multitask_available = bool(args.multitask_model and os.path.exists(args.multitask_model))
    depth_available     = bool(args.depth_model and os.path.exists(args.depth_model))
    foreground_available= bool(args.foreground_model and os.path.exists(args.foreground_model))
    normal_available    = bool(args.normal_model and os.path.exists(args.normal_model))
    individual_available = depth_available or foreground_available or normal_available

    if not (multitask_available or individual_available):
        print("Error: Provide at least one existing model path.")
        print("  --multitask-model OR any of --depth-model/--foreground-model/--normal-model")
        sys.exit(1)

    # Gather images
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

    # Build estimators ONCE (sessions get created once and stick to GPU/CPU memory)
    depth_estimator = None
    foreground_segmenter = None
    normal_estimator = None
    multitask_estimator = None

    if depth_available:
        depth_estimator = RelativeDepthEstimator(onnx_model=args.depth_model, is_inverse=True)
    if foreground_available:
        foreground_segmenter = SoftForegroundSegmenter(onnx_model=args.foreground_model)
    if normal_available:
        normal_estimator = SurfaceNormalEstimator(onnx_model=args.normal_model)
    if multitask_available:
        multitask_estimator = MultiTaskEstimator(onnx_model=args.multitask_model, is_inverse_depth=False)

    # Warmup on first decodable image to kick in kernels / cudnn autotune
    first_ok = None
    for p in images:
        tmp = cv2.imread(p)
        if tmp is not None:
            first_ok = (p, tmp)
            break

    if first_ok is None:
        print("No readable image found.")
        sys.exit(1)

    warm_im = first_ok[1]
    for _ in range(max(0, args.warmup)):
        if individual_available:
            _ = process_with_individual_models(
                warm_im,
                depth_estimator=depth_estimator,
                foreground_segmenter=foreground_segmenter,
                normal_estimator=normal_estimator,
            )
        if multitask_estimator is not None:
            _ = process_with_multitask_model(warm_im, multitask_estimator)

    # Prefetch decoded images
    q = prefetch_paths(images, capacity=args.prefetch)

    # Async writers pool
    writer_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    pending: List[Future] = []

    disable_bar = args.no_progress or not sys.stderr.isatty()
    pbar = tqdm(images, desc="Processing images", unit="img",
                dynamic_ncols=True, leave=False, disable=disable_bar)

    errors = 0
    processed = 0

    for (img_path, image_bgr) in consume_prefetch(q):
        if image_bgr is None:
            errors += 1
            if disable_bar:
                print(f"Warning: Could not read image: {img_path}")
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        if not disable_bar:
            # Avoid over-updating postfix (TTY lock); show filename sparsely
            if processed % 10 == 0:
                pbar.set_postfix_str(stem)

        # Run GPU work (sessions already warm & resident)
        # Individual models
        if individual_available:
            res_ind = process_with_individual_models(
                image_bgr,
                depth_estimator=depth_estimator,
                foreground_segmenter=foreground_segmenter,
                normal_estimator=normal_estimator,
            )
            if res_ind:
                # Offload disk writes to background
                pending.append(
                    writer_pool.submit(
                        save_task_visuals,
                        image_bgr, res_ind, args.output_dir, stem, "individual",
                        save_raw=args.raw_result, src_image_path=img_path
                    )
                )

        # Multi-task model
        if multitask_estimator is not None:
            res_multi = process_with_multitask_model(image_bgr, multitask_estimator)
            if res_multi:
                pending.append(
                    writer_pool.submit(
                        save_task_visuals,
                        image_bgr, res_multi, args.output_dir, stem, "multitask",
                        save_raw=args.raw_result, src_image_path=img_path
                    )
                )

        processed += 1
        pbar.update(1)

        # Keep the number of in-flight write tasks bounded so we don't explode RAM
        # Target bound = 2×workers
        cap = max(2 * max(1, args.workers), 2)
        if len(pending) >= cap:
            done, pending = _drain_some(pending, leave=len(pending) - cap // 2)
            # swallow exceptions early
            for fut in done:
                _ = fut.result()

    # finalize progress bar
    if not disable_bar:
        pbar.close()

    # Wait for outstanding writes
    for fut in pending:
        _ = fut.result()
    writer_pool.shutdown(wait=True)

    summary = f"Done. Outputs saved under: {os.path.abspath(args.output_dir)}"
    if errors:
        summary += f" (skipped {errors} unreadable image{'s' if errors != 1 else ''})"
    print(summary)


def _drain_some(futs: List[Future], leave: int) -> Tuple[List[Future], List[Future]]:
    """Pop earliest futures until `leave` remain; return (completed, remaining)."""
    completed: List[Future] = []
    remaining = futs
    while len(remaining) > leave:
        f = remaining.pop(0)
        if f.done():
            completed.append(f)
        else:
            # Not done yet; push back and stop (keep order)
            remaining.insert(0, f)
            break
    return completed, remaining


if __name__ == "__main__":
    # OpenCV threads can starve GPU feed when >1; set to 1 explicitly.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    main()
