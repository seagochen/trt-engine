#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

def calculate_average_color(image, bbox):
    """Calculate the average color within a bounding box."""
    cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.average(np.average(cropped_image, axis=0), axis=0)

def decide_text_color(average_color):
    """Decide whether to use black or white text based on the average color."""
    return (0, 0, 0) if np.mean(average_color) > 127 else (255, 255, 255)

def draw_text_with_opposite_color(frame: any,
                                  text: str,
                                  left_top: tuple,
                                  font_scale=1,
                                  thickness=2):
    """Draw text with color contrasting the average color of the background."""
    # Define font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size and average color of background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    average_color = calculate_average_color(frame, (left_top[0], left_top[1] - text_height, text_width, text_height))

    # Choose text color based on background
    text_color = decide_text_color(average_color)

    # Draw text on OpenCV frame
    cv2.putText(frame, text, (left_top[0], left_top[1]), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    return frame

def draw_text_with_background(frame: any,
                              text: str,
                              left_top: tuple,
                              font_scale=1,
                              thickness=2,
                              background_color=(0, 0, 0),
                              background_alpha=0.5,
                              background_padding=5):
    """Draw text with a background for better contrast."""
    # Define font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Define background rectangle
    background_rect = (left_top[0] - background_padding,
                       left_top[1] - text_height - baseline - background_padding,
                       left_top[0] + text_width + background_padding,
                       left_top[1])

    # Draw background rectangle on OpenCV frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (background_rect[0], background_rect[1]),
                  (background_rect[2], background_rect[3]), background_color, -1)
    cv2.addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame)

    # Draw text on OpenCV frame
    cv2.putText(frame, text, (left_top[0], left_top[1] - baseline), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return frame

if __name__ == "__main__":

    # Load the image
    frame = cv2.imread("images/spectrum.jpg")
    if frame is None:
        print("Error: Image file not found or unable to load.")
        exit()

    # Draw text with automatic opposite color decision
    frame = draw_text_with_opposite_color(frame, "Hello, World!", (50, 100), font_scale=1, thickness=2)

    # Draw text with background
    frame = draw_text_with_background(frame, "Hello, OpenCV!", (50, 200), font_scale=1, thickness=2, background_alpha=0.5)

    # Resize the frame
    frame = cv2.resize(frame, (600, 600))

    # Show the frame with the added text
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    # Release resources
    cv2.destroyAllWindows()
