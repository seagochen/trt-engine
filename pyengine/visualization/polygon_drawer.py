import cv2
import numpy as np

from pyengine.visualization.color_utils import hex_to_bgr
# from color_utils import hex_to_bgr


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


def fill_grid_area(
        image: np.ndarray,
        area: list,
        color: str,
        transparency: float,
        grid_rows: int,
        grid_cols: int,
        perspective: bool = True,
        grid_line_color: str = "#FFFFFF",
        grid_transparency: float = 1.0  # <--- 【新增】网格线透明度参数
) -> np.ndarray:
    """
    Fills a polygonal area with a specified color and transparency, and overlays a grid.
    The grid can be drawn with or without a perspective effect.

    Parameters:
      image (np.ndarray): The image on which to draw (BGR format).
      area (list): A list of (x, y) coordinates defining the polygon.
      color (str): Fill color as a hex string (e.g., "#FF00AA").
      transparency (float): Transparency level (0 to 1).
      grid_rows (int): Number of rows in the grid.
      grid_cols (int): Number of columns in the grid.
      perspective (bool): If True, applies a perspective transform to the grid.
      grid_line_color (str): Color of the grid lines as a hex string.
      grid_transparency (float): Transparency of the grid lines (0 to 1).
    Returns:
      np.ndarray: The image with the filled and gridded area.
    """
    filled_image = fill_area(image, area, color, transparency)

    if not area or len(area) < 3:
        return filled_image

    pts = np.array(area, dtype=np.int32)
    grid_bgr_color = hex_to_bgr(grid_line_color)

    grid_overlay = np.zeros_like(filled_image, dtype=np.uint8)

    if perspective:
        if len(area) != 4:
            print("Perspective grid requires exactly 4 points for the area. Aborting grid drawing.")
            return filled_image

        src_rect_size = 100
        src_pts = np.float32([[0, 0], [src_rect_size, 0], [src_rect_size, src_rect_size], [0, src_rect_size]])
        dst_pts = np.float32(area)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        for i in range(1, grid_rows):
            y = i * src_rect_size / grid_rows
            p1_src = np.float32([[[0, y]]])
            p2_src = np.float32([[[src_rect_size, y]]])
            p1_dst = cv2.perspectiveTransform(p1_src, matrix)[0][0]
            p2_dst = cv2.perspectiveTransform(p2_src, matrix)[0][0]
            cv2.line(grid_overlay, tuple(p1_dst.astype(int)), tuple(p2_dst.astype(int)), grid_bgr_color, 1)

        for i in range(1, grid_cols):
            x = i * src_rect_size / grid_cols
            p1_src = np.float32([[[x, 0]]])
            p2_src = np.float32([[[x, src_rect_size]]])
            p1_dst = cv2.perspectiveTransform(p1_src, matrix)[0][0]
            p2_dst = cv2.perspectiveTransform(p2_src, matrix)[0][0]
            cv2.line(grid_overlay, tuple(p1_dst.astype(int)), tuple(p2_dst.astype(int)), grid_bgr_color, 1)

    else:
        x, y, w, h = cv2.boundingRect(pts)

        for i in range(1, grid_rows):
            line_y = int(y + (i * h / grid_rows))
            cv2.line(grid_overlay, (x, line_y), (x + w, line_y), grid_bgr_color, 1)

        for i in range(1, grid_cols):
            line_x = int(x + (i * w / grid_cols))
            cv2.line(grid_overlay, (line_x, y), (line_x, y + h), grid_bgr_color, 1)

    mask = np.zeros(filled_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    clipped_grid = cv2.bitwise_and(grid_overlay, grid_overlay, mask=mask)

    # 【修改】使用 addWeighted 来混合网格，以实现透明效果
    final_image = cv2.addWeighted(filled_image, 1.0, clipped_grid, grid_transparency, 0)

    return final_image


# Example usage:
if __name__ == '__main__':

    def test_fill():
        # Load your image using OpenCV (ensure the path is correct)
        image = cv2.imread("/opt/images/apples.png")
        if image is None:
            print("Error: Could not load image for test_fill.")
            return

        # Define the area as a list of (x, y) points.
        area = [(50, 50), (150, 50), (150, 150), (50, 150), (50, 50)]
        area_color = "#0000FF" # Red
        transparency = 0.4

        # Fill the alert area on the image.
        filled_image = fill_area(image.copy(), area, area_color, transparency)

        # Display or save the resulting image.
        cv2.imshow("Filled Image", filled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_grid():
        # Load your image using OpenCV
        image = cv2.imread("/opt/images/apples.png")
        if image is None:
            print("Error: Could not load image for test_grid.")
            return

        # Define a quadrilateral area for the grid
        # Points: Top-left, Top-right, Bottom-right, Bottom-left
        grid_area = [(200, 100), (450, 120), (500, 300), (150, 250)]
        grid_color = "#00FF00"  # Green
        transparency = 0.5
        rows = 5
        cols = 5

        # --- Test 1: Grid with perspective ---
        perspective_image = fill_grid_area(
            image.copy(), grid_area, grid_color, transparency,
            grid_rows=rows, grid_cols=cols, perspective=True
        )

        # --- Test 2: Grid without perspective (uniform) ---
        non_perspective_image = fill_grid_area(
            image.copy(), grid_area, grid_color, transparency,
            grid_rows=rows, grid_cols=cols, perspective=False
        )
        
        # Display the results
        cv2.imshow("Perspective Grid", perspective_image)
        cv2.imshow("Non-Perspective Grid", non_perspective_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Run tests
    # test_fill()
    test_grid()
