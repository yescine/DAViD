"""This module provides a Monocular Relative Depth Estimator which estimates the depth map of human in an image."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from pixelwise_estimator import PixelwiseEstimator
from utils import composite_model_output_to_image


class RelativeDepthEstimator(PixelwiseEstimator):
    """Estimates the relative depth map of human in an image."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
        is_inverse: bool = True,
    ):
        """Creates a depth estimator.

        Arguments:
            onnx_model: A path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
            is_inverse: If True, the depth map is inverted (i.e., closer objects have higher values).

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )
        self.is_inverse = is_inverse

    def estimate_relative_depth(self, image: np.ndarray) -> np.ndarray:
        """Predict the relative depth map given input image."""
        depth, metadata = self._estimate_dense_map(image)
        depth = depth[0][0]
        depth_map = composite_model_output_to_image(
            depth, metadata, interp_mode=cv2.INTER_CUBIC
        )

        return depth_map * -1 if self.is_inverse else depth_map
