"""Visualization script for SynthHuman dataset.

Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "runtime"))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenEXR support in OpenCV

import cv2
from visualize import (
    create_concatenated_display,
    visualize_foreground,
    visualize_normal_maps,
    visualize_relative_depth_map,
)


def main():
    """Main function to visualize the SynthHuman dataset."""
    parser = argparse.ArgumentParser(description="Visualize SynthHuman dataset")
    parser.add_argument("data_dir", type=Path, help="Path to the SynthHuman dataset directory")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index for samples (default: 0)",
    )
    args = parser.parse_args()

    sample_idx = args.start_idx
    while True:
        sample_name = f"{sample_idx:07d}"
        if not (args.data_dir / f"rgb_{sample_name}.png").exists():
            break

        image = cv2.imread((args.data_dir / f"rgb_{sample_name}.png").as_posix())
        depth_map = cv2.imread(
            (args.data_dir / f"depth_{sample_name}.exr").as_posix(),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
        foreground_mask = cv2.imread(
            (args.data_dir / f"alpha_{sample_name}.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        normal_maps = cv2.imread(
            (args.data_dir / f"normal_{sample_name}.exr").as_posix(),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )

        depth_vis = visualize_relative_depth_map(image, depth_map, foreground_mask, alpha_threshold=127)
        foreground_vis = visualize_foreground(image, foreground_mask)
        normal_vis = visualize_normal_maps(image, normal_maps, foreground_mask)

        result = create_concatenated_display(
            [image, depth_vis, foreground_vis, normal_vis],
            ["RGB", "Depth", "Foreground", "Normals"],
        )

        cv2.putText(
            result,
            sample_name,
            (10, result.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("SynthHuman", result)
        if cv2.waitKey(0) == 27:  # Exit on ESC key
            break

        sample_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
