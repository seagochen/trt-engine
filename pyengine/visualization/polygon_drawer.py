import cv2
import numpy as np

from pyengine.visualization.color_utils import hex_to_bgr


def fill_area(image: np.ndarray, area: list, color: str, transparency: float) -> np.ndarray:
    """
    If the provided area represents a closed shape, fill that polygon on the image with
    the specified color and transparency.

    Parameters:
      image (np.ndarray): The image on which to draw (assumed to be in BGR format).
      area (list): A list of (x, y) coordinates defining the polygon.
      color (str): Color of the area as a hex string (e.g. "#FF00AA").
      transparency (float): Transparency level (0 to 1), where 0 is fully transparent and 1 is fully opaque.

    Returns:
      np.ndarray: The image with the filled area if the shape is closed; otherwise, the original image.
    """
    # Check if the area has enough points to form a polygon.
    if not area or len(area) < 3:
        print("At least 3 points are required to form a polygon.")
        return image

    # # Check if the shape is closed (first point equals the last point).
    # if area[0] != area[-1]:
    #     print("The area is not a closed shape; no fill will be applied.")
    #     return image

    # 如果 image 是只读的，则复制一份可写的
    if not image.flags.writeable:
        image = image.copy()

    # Convert the points to a NumPy array in the format required by cv2.fillPoly.
    pts = np.array(area, dtype=np.int32).reshape((-1, 1, 2))
    bgr_color = hex_to_bgr(color)

    # Create an overlay of the image to draw the polygon with transparency.
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], bgr_color)

    # Blend the overlay with the original image using the specified transparency.
    cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

    return image


# Example usage:
if __name__ == '__main__':

    def test():
        # Load your image using OpenCV (ensure the path is correct)
        image = cv2.imread("/opt/images/apples.png")

        # Define the area as a list of (x, y) points.
        area = [(50, 50), (150, 50), (150, 150), (50, 150), (50, 50)]
        area_color = "#000000"
        transparency = 0.3  # 0 is fully transparent, 1 is fully opaque

        # Fill the alert area on the image.
        filled_image = fill_area(image, area, area_color, transparency)

        # Display or save the resulting image.
        cv2.imshow("Filled Image", filled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Run test
    test()
