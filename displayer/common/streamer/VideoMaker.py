#!/usr/bin/env python
# coding: utf-8

import cv2
import datetime


class VideoMaker:
    def __init__(self, cap, filename=None, fps=None, width=None, height=None):

        if not cap or not cap.isOpened():
            raise ValueError('Invalid video capture object')
        else:
            self.cap = cap

        # Save the filename
        if filename is None:
            self.filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            self.filename = filename

        # If no FPS is provided, get it from the camera
        if not fps:
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        else:
            self.fps = fps

        # If no width is provided, get it from the camera
        if not width:
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            self.width = width

        # If no height is provided, get it from the camera
        if not height:
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.height = height

        # Initialize the VideoWriter
        self.video = self.initialize_video_writer()

    def initialize_video_writer(self):
        # Attempt to initialize VideoWriter with MP4 codec
        video = cv2.VideoWriter(self.filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (self.width, self.height))
        if video.isOpened():
            return video

        # Fallback to AVI codec
        self.filename += '.avi'
        video = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'XVID'),
                                self.fps,
                                (self.width, self.height))
        if video.isOpened():
            return video

        raise ValueError('Failed to initialize video writer with both MP4 and AVI codecs')

    def add_frame(self, frame):
        # Get the size of the frame
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Check if the frame size is the same as the video size
        if frame_width != self.width or frame_height != self.height:
            # Resize the frame to match the video dimensions
            frame = cv2.resize(frame, (self.width, self.height))

        # Write the frame to the video
        self.video.write(frame)

    def release(self):
        print('Closing video file')
        self.video.release()


if __name__ == "__main__":
    # Use VideoCapture to open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError('Invalid video capture object')

    # Get the camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    # Use VideoMaker, specifying keyword arguments explicitly
    video_maker = VideoMaker(cap, fps=fps, width=width, height=height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_maker.add_frame(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoMaker and the camera
    video_maker.release()
    cap.release()
    cv2.destroyAllWindows()
