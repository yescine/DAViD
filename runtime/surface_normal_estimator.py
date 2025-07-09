"""This module provides a Surface Normal Estimator which estimates the surface normal map of human in an image."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from pixelwise_estimator import PixelwiseEstimator
from utils import composite_model_output_to_image


class SurfaceNormalEstimator(PixelwiseEstimator):
    """Estimates the surface normal map of human in an image."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
    ):
        """Creates a surface normal estimator.

        Arguments:
            onnx_model: A path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )

    def estimate_normal(self, image: np.ndarray) -> np.ndarray:
        """Predict the normal map given input image."""
        normal, metadata = self._estimate_dense_map(image)
        normal = normal[0][0]
        normal = np.transpose(normal, (1, 2, 0))

        normal_map = composite_model_output_to_image(
            normal, metadata, interp_mode=cv2.INTER_CUBIC
        )
        normal_map /= np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-8
        return normal_map
