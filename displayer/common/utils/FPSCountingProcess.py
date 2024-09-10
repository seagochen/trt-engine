#!/usr/bin/env python
# coding: utf-8

import multiprocessing
import time

class FPSCountingProcess(multiprocessing.Process):

    def __init__(self):
        super(FPSCountingProcess, self).__init__()
        self.__last_time = multiprocessing.Value('d', time.time())
        self.__frame_count = multiprocessing.Value('i', 0)
        self.__last_fps = multiprocessing.Value('i', 0)

    def increment_frame_count(self):
        with self.__frame_count.get_lock():
            self.__frame_count.value += 1

    def get_fps(self):
        with self.__last_fps.get_lock():
            elapsed_time = time.time() - self.__last_time.value
            if elapsed_time > 1:
                self.__last_fps.value = self.__frame_count.value
                self.__frame_count.value = 0
                self.__last_time.value = time.time()
        
        return self.__last_fps.value
    
    def stop(self):
        self.terminate()
        self.join()


if __name__ == "__main__":

    import cv2

    process = FPSCountingProcess()
    process.start()

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        process.increment_frame_count()
        fps = process.get_fps()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera
    cap.release()
    cv2.destroyAllWindows()

