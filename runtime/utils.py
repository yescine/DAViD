"""Utility classes and functions for image processing and ROI operations."""

from typing import Union, Any
import numpy as np
import cv2
from onnxruntime import InferenceSession
import onnx
import json
from pathlib import Path


ONNX_EP = ["CUDAExecutionProvider", "CPUExecutionProvider"]
UINT8_MAX = np.iinfo(np.uint8).max
UINT16_MAX = np.iinfo(np.uint16).max


class ImageFormatError(Exception):
    """Exception raised for invalid image formats."""

    pass


class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""

    pass


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Preprocesses a BGR image for DNN. Turning to float if not already and normalizing to [0, 1].

    Normalization of uint images is done by dividing by brightest possible value (e.g. 255 for uint8).

    Arguments:
        img: The image to preprocess, can be uint8, uint16, float16, float32 or float64.

    Returns:
        The preprocessed image in np.float32 format.

    Raises:
        ImageFormatError: If the image is not three channels or not uint8, uint16, float16, float32 or float64.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ImageFormatError("image must be 3 channels, got shape: {img.shape}")
    if img.dtype not in [np.uint8, np.uint16, np.float16, np.float32, np.float64]:  # noqa: PLR6201
        raise ImageFormatError("image must be uint8 or float16, float32, float64")

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / UINT8_MAX
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / UINT16_MAX
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


def get_metadata(
    onnx_path_or_session: Union[Path, str, InferenceSession],
) -> dict[str, Any]:
    """Gets metadata for an ONNX model filepath or inference session.

    Arguments:
        onnx_path_or_session: The ONNX model file path or ONNX Runtime session.

    Returns:
        The metadata dictionary.

    Raises:
        TypeError: If the input is not a file path or ONNX Runtime session.
    """
    if isinstance(onnx_path_or_session, (str, Path)):
        assert Path(onnx_path_or_session).is_file(), (
            f"onnx model {onnx_path_or_session} not found"
        )
        onnx_model = onnx.load(str(onnx_path_or_session))
        for prop in onnx_model.metadata_props:
            if prop.key == "metadata":
                return json.loads(prop.value)
        return {}

    if isinstance(onnx_path_or_session, InferenceSession):
        custom_metadata_map = onnx_path_or_session.get_modelmeta().custom_metadata_map
        if "metadata" in custom_metadata_map:
            return json.loads(custom_metadata_map["metadata"])
        return {}

    raise TypeError(
        f"{onnx_path_or_session} should be a file path or ONNX Runtime session"
    )


def prepare_image_for_model(
    image: np.ndarray, roi_size: int = 512
) -> tuple[np.ndarray, dict]:
    """Prepare any input image for model inference by resizing to roi_size x roi_size.

    This function takes an image of any size and prepares it for a model that expects
    a square input (e.g., 512x512). It handles aspect ratio preservation by padding
    with replicated border values.

    Args:
        image: Input image of any size
        roi_size: Target size for the model (default 512)

    Returns:
        tuple: (preprocessed_image, metadata_dict)
            - preprocessed_image: Image resized to roi_size x roi_size
            - metadata_dict: Contains information needed to composite back to original size
    """

    # Get original shape
    original_shape = image.shape[:2]  # (height, width)

    # Calculate padding to make the image square
    if original_shape[0] < original_shape[1]:
        pad_h = (original_shape[1] - original_shape[0]) // 2
        pad_w = 0
        pad_h_extra = original_shape[1] - original_shape[0] - pad_h
        pad_w_extra = 0
    elif original_shape[0] > original_shape[1]:
        pad_w = (original_shape[0] - original_shape[1]) // 2
        pad_h = 0
        pad_w_extra = original_shape[0] - original_shape[1] - pad_w
        pad_h_extra = 0
    else:
        pad_h = pad_w = pad_h_extra = pad_w_extra = 0

    # Pad the image to make it square
    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_h,
        bottom=pad_h_extra,
        left=pad_w,
        right=pad_w_extra,
        borderType=cv2.BORDER_REPLICATE,
    )

    square_shape = padded_image.shape[:2]

    while padded_image.shape[1] > roi_size * 3 and padded_image.shape[0] > roi_size * 3:
        padded_image = cv2.pyrDown(padded_image)

    resized_image = cv2.resize(
        padded_image, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR
    )

    metadata = {
        "original_shape": original_shape,
        "square_shape": square_shape,
        "original_padding": (pad_h, pad_w, pad_h_extra, pad_w_extra),
    }

    return resized_image, metadata


def composite_model_output_to_image(
    model_output: np.ndarray, metadata: dict, interp_mode: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """Composite model output back to the original image size.

    Takes the model output (which should be roi_size x roi_size) and composites it
    back to the original image dimensions using the metadata from prepare_image_for_model.

    Args:
        model_output: Output from the model (roi_size x roi_size)
        metadata: Metadata dict returned from prepare_image_for_model
        interp_mode: Interpolation mode for resizing (default INTER_NEAREST for discrete outputs)

    Returns:
        np.ndarray: Output composited to original image size
    """
    pad_h, pad_w, pad_h_extra, pad_w_extra = metadata["original_padding"]

    # Resize the entire model output back to the square shape
    square_shape = metadata["square_shape"]
    resized_to_square = cv2.resize(
        model_output, (square_shape[1], square_shape[0]), interpolation=interp_mode
    )

    # Remove the padding to get back to original dimensions
    if pad_h > 0 or pad_h_extra > 0:
        final_output = resized_to_square[pad_h : square_shape[0] - pad_h_extra, :]
    elif pad_w > 0 or pad_w_extra > 0:
        final_output = resized_to_square[:, pad_w : square_shape[1] - pad_w_extra]
    else:
        final_output = resized_to_square

    return final_output
