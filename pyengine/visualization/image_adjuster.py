import cv2
import numpy as np
from typing import Tuple

# --- Existing Functions (Refactored) ---

def adjust_contrast_brightness(image: np.ndarray,
                               contrast: float = 1.0,
                               brightness: float = 0.0) -> np.ndarray:
    """
    Adjusts the contrast and brightness of an image.
    output = contrast * input + brightness

    Args:
        image: Input image (NumPy array).
        contrast: Contrast control factor (alpha). Should be non-negative.
                  > 1: increase contrast, < 1: decrease contrast.
        brightness: Brightness control factor (beta). Added to pixel values.

    Returns:
        Adjusted image (NumPy array).

    Raises:
        ValueError: If contrast is negative.
    """
    if contrast < 0:
        raise ValueError("Contrast factor (alpha) must be non-negative.")
    # Apply the linear transformation: g(x) = alpha*f(x) + beta
    # convertScaleAbs handles scaling and ensures output is uint8 (0-255)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Adjusts the gamma of an image using a Look-Up Table (LUT).

    Args:
        image: Input image (NumPy array).
        gamma: Gamma correction factor. Should be positive.
               > 1: image darker, < 1: image brighter.

    Returns:
        Gamma-corrected image (NumPy array).

    Raises:
        ValueError: If gamma is non-positive.
    """
    if gamma <= 0:
        raise ValueError("Gamma value must be positive.")

    # Calculate inverse gamma
    invGamma = 1.0 / gamma
    # Build the LUT: Apply gamma correction formula to each possible pixel value (0-255)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply the LUT to the image
    gamma_corrected_image = cv2.LUT(image, table)
    return gamma_corrected_image


def sharpen_image(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Sharpens the input image using a Laplacian-based kernel.

    Args:
        image: Input image (NumPy array).
        strength: Sharpening intensity factor (controls the weight of the Laplacian).
                  Should be non-negative. Higher values mean more sharpening. Default is 0.5.

    Returns:
        Sharpened image (NumPy array).

    Raises:
        ValueError: If strength is negative.
    """
    if strength < 0:
        raise ValueError("Sharpening strength must be non-negative.")

    # Define a standard Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1,  4, -1],
                                 [0, -1, 0]])
    # Calculate the Laplacian of the image
    laplacian = cv2.filter2D(image, cv2.CV_64F, laplacian_kernel)

    # Add the scaled Laplacian back to the original image
    # Convert image to float for addition, then clip and convert back to uint8
    sharpened_image_float = image.astype(np.float64) + strength * laplacian
    # Clip values to the valid range [0, 255]
    sharpened_image_float = np.clip(sharpened_image_float, 0, 255)
    # Convert back to uint8
    sharpened_image = sharpened_image_float.astype(np.uint8)

    return sharpened_image

    # --- Alternative Sharpening Kernel (Original User Version interpretation) ---
    # kernel_strength = 5.0 + strength * 4.0 # Scale alpha differently if needed
    # kernel = np.array([[0, -1, 0],
    #                    [-1, kernel_strength, -1],
    #                    [0, -1, 0]])
    # # Apply the kernel to the image, -1 means output depth is same as input
    # sharpened_image = cv2.filter2D(image, -1, kernel)
    # return sharpened_image


def smooth_image(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Smoothes the input image using Gaussian blur.

    Args:
        image: Input image (NumPy array).
        ksize: Kernel size for the Gaussian blur (width and height).
               Must be a positive odd number. Default is 5.

    Returns:
        Smoothed image (NumPy array).

    Raises:
        ValueError: If kernel size is not a positive odd number.
    """
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    # Apply Gaussian blur
    # (ksize, ksize) is the kernel size. 0 means sigma is calculated from kernel size.
    smoothed_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return smoothed_image

# --- Other Common Utility Functions ---

def resize_image(image: np.ndarray,
                 width: int = None,
                 height: int = None,
                 inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resizes an image to the specified width and/or height.

    Args:
        image: Input image (NumPy array).
        width: Target width. If None, calculate based on height ratio.
        height: Target height. If None, calculate based on width ratio.
        inter: Interpolation method (e.g., cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC).
               cv2.INTER_AREA is often good for shrinking, cv2.INTER_LINEAR for general purpose,
               cv2.INTER_CUBIC for enlarging (slower).

    Returns:
        Resized image (NumPy array).

    Raises:
        ValueError: If both width and height are None.
    """
    if width is None and height is None:
        raise ValueError("Either width or height must be provided for resizing.")

    (h, w) = image.shape[:2]

    if width is None:
        # Calculate width based on height ratio
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    elif height is None:
        # Calculate height based on width ratio
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        # Both width and height are provided
        dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=inter)
    return resized_image


def rotate_image(image: np.ndarray, angle: float, center: Tuple[int, int] = None, scale: float = 1.0) -> np.ndarray:
    """
    Rotates an image by a given angle around a center point.

    Args:
        image: Input image (NumPy array).
        angle: Rotation angle in degrees (positive values mean counter-clockwise).
        center: Rotation center coordinates (x, y). If None, use the image center.
        scale: Optional scaling factor during rotation.

    Returns:
        Rotated image (NumPy array). The output image size might be the same
        as the input, potentially cropping rotated corners, or larger if adjusted.
        This implementation keeps the original image size.
    """
    (h, w) = image.shape[:2]

    # If center is not specified, use the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Perform the rotation using affine warp
    # The third argument is the output image size (dsize)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to grayscale.

    Args:
        image: Input BGR image (NumPy array).

    Returns:
        Grayscale image (NumPy array). Returns original if already grayscale.
    """
    if len(image.shape) == 2: # Already grayscale
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3: # BGR
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    else:
        raise ValueError("Input image must be BGR or Grayscale.")


def apply_binary_threshold(image: np.ndarray, threshold_value: int = 127, max_value: int = 255, threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
    """
    Applies fixed-level thresholding to a grayscale image.

    Args:
        image: Input grayscale image (NumPy array).
        threshold_value: Threshold value used to classify pixel values.
        max_value: Value assigned to pixels exceeding the threshold (in binary types).
        threshold_type: Thresholding type (e.g., cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,
                        cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV).

    Returns:
        Thresholded image (NumPy array).
    """
    if len(image.shape) != 2:
         # Convert to grayscale if it's not already
         image_gray = convert_to_grayscale(image)
         print("Warning: Input image for thresholding was not grayscale. Converting.")
    else:
        image_gray = image

    # Apply thresholding
    _, thresholded_image = cv2.threshold(image_gray, threshold_value, max_value, threshold_type)
    return thresholded_image


def detect_edges_canny(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """
    Detects edges in an image using the Canny edge detector.

    Args:
        image: Input image (NumPy array, preferably grayscale).
        threshold1: First threshold for the hysteresis procedure.
        threshold2: Second threshold for the hysteresis procedure.

    Returns:
        Edge map (binary image, uint8 NumPy array).
    """
    if len(image.shape) != 2:
        # Canny works best on grayscale images
        image_gray = convert_to_grayscale(image)
    else:
        image_gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, threshold1, threshold2)
    return edges


def apply_median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Applies a median filter to the image, effective for salt-and-pepper noise.

    Args:
        image: Input image (NumPy array).
        ksize: Aperture linear size. Must be odd and greater than 1. Default is 5.

    Returns:
        Image with median filtering applied (NumPy array).

    Raises:
        ValueError: If kernel size is not odd or is less than or equal to 1.
    """
    if ksize <= 1 or ksize % 2 == 0:
        raise ValueError("Kernel size (ksize) must be odd and greater than 1.")

    median_blurred_image = cv2.medianBlur(image, ksize)
    return median_blurred_image


# --- Example Usage ---
if __name__ == '__main__':
    # Load an example image (replace with your image path)
    try:
        # Create a dummy gradient image if no file is available
        img_h, img_w = 300, 400
        original_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        original_image[:, :, 0] = np.tile(np.linspace(0, 255, img_w), (img_h, 1)) # Blue gradient
        original_image[:, :, 1] = np.tile(np.linspace(0, 255, img_h), (img_w, 1)).T # Green gradient
        print("Created a dummy gradient image for demonstration.")
        # Alternatively, load from file:
        # original_image = cv2.imread("path/to/your/image.jpg")
        # if original_image is None:
        #    raise FileNotFoundError("Image not found or unable to load.")

        print("Original Image Shape:", original_image.shape)

        # 1. Adjust Contrast/Brightness
        contrast_img = adjust_contrast_brightness(original_image, contrast=1.5, brightness=10)
        cv2.imshow("Contrast/Brightness Adjusted", contrast_img)
        cv2.waitKey(0)

        # 2. Adjust Gamma
        gamma_img = adjust_gamma(original_image, gamma=0.5) # Make brighter
        cv2.imshow("Gamma Adjusted (Brighter)", gamma_img)
        cv2.waitKey(0)
        gamma_img_dark = adjust_gamma(original_image, gamma=1.5) # Make darker
        cv2.imshow("Gamma Adjusted (Darker)", gamma_img_dark)
        cv2.waitKey(0)

        # 3. Sharpen Image
        sharpened_img = sharpen_image(original_image, strength=0.8)
        cv2.imshow("Sharpened", sharpened_img)
        cv2.waitKey(0)

        # 4. Smooth Image
        smoothed_img = smooth_image(original_image, ksize=9)
        cv2.imshow("Smoothed (Gaussian)", smoothed_img)
        cv2.waitKey(0)

        # 5. Resize Image
        resized_img = resize_image(original_image, width=200) # Resize by width
        cv2.imshow("Resized (Width=200)", resized_img)
        cv2.waitKey(0)

        # 6. Rotate Image
        rotated_img = rotate_image(original_image, angle=45)
        cv2.imshow("Rotated 45 Degrees", rotated_img)
        cv2.waitKey(0)

        # 7. Grayscale Conversion
        gray_img = convert_to_grayscale(original_image)
        cv2.imshow("Grayscale", gray_img)
        cv2.waitKey(0)

        # 8. Binary Thresholding
        threshold_img = apply_binary_threshold(gray_img, threshold_value=100)
        cv2.imshow("Binary Threshold (100)", threshold_img)
        cv2.waitKey(0)

        # 9. Canny Edge Detection
        edges_img = detect_edges_canny(gray_img, threshold1=50, threshold2=150)
        cv2.imshow("Canny Edges", edges_img)
        cv2.waitKey(0)

        # 10. Median Blur
        # First, add some salt-and-pepper noise for demonstration
        noisy_image = original_image.copy()
        noise_level = 0.02
        s_vs_p = 0.5
        # Salt mode
        num_salt = np.ceil(noise_level * noisy_image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_image.shape]
        noisy_image[tuple(coords)] = (255, 255, 255)
        # Pepper mode
        num_pepper = np.ceil(noise_level * noisy_image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_image.shape]
        noisy_image[tuple(coords)] = (0, 0, 0)
        cv2.imshow("Noisy Image", noisy_image)
        cv2.waitKey(0)

        median_blur_img = apply_median_blur(noisy_image, ksize=5)
        cv2.imshow("Median Blurred (Noise Reduction)", median_blur_img)
        cv2.waitKey(0)

        print("All examples shown. Closing windows.")
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Input Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")