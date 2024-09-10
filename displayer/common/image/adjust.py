#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def adjust_contrast_brightness(image, contrast=1.0, brightness=0.0):
    """
    调整对比度和亮度
    Args:
        image:
        contrast:
        brightness:

    Returns:
    """
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return image


def adjust_gamma(image, gamma=1.0):
    """
    调整Gamma值
    Args:
        image:
        gamma:

    Returns:
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    return image


def sharpen_image(image, alpha=1.5):
    """
    Sharpen the input image by the given alpha value.

    Parameters:
    - image: Input image to be sharpened.
    - alpha: Sharpening intensity factor. Default is 1.5.

    Returns:
    - Sharpened image.
    """
    # Ensure alpha is positive
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")

    # Create a kernel for sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + alpha, -1],
                       [0, -1, 0]])

    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


def smooth_image(image, ksize=5):
    """
    Smooth the input image using Gaussian blur.

    Parameters:
    - image: Input image to be smoothed.
    - ksize: Kernel size for the Gaussian blur. Must be an odd number. Default is 5.

    Returns:
    - Smoothed image.
    """
    # Ensure kernel size is odd and greater than 1
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("Kernel size must be an odd number greater than 1.")

    # Apply Gaussian blur to the image
    smoothed_image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return smoothed_image

