import threading # Use threading instead of multiprocessing
import time

class FPSCounter: # No longer inherits from Process

    def __init__(self):
        # Standard attributes instead of multiprocessing.Value
        self.__last_time = time.time()
        self.__frame_count = 0
        self.__last_fps = 0
        # A single lock to protect all shared attributes
        self.__lock = threading.Lock()

    def increment_frame_count(self):
        # Acquire lock to safely modify frame count
        with self.__lock:
            self.__frame_count += 1

    def get_fps(self) -> int:
        # Acquire lock to safely read/modify attributes
        with self.__lock:
            current_time = time.time()
            elapsed_time = current_time - self.__last_time
            # Calculate FPS every second
            if elapsed_time > 1.0:
                self.__last_fps = self.__frame_count # Calculate FPS
                self.__frame_count = 0           # Reset frame count
                self.__last_time = current_time  # Reset timer
            # Return the most recently calculated FPS value
            return self.__last_fps

    # The stop() method is no longer needed as there's no process to stop.


if __name__ == "__main__":

    import cv2

    # Just instantiate the class directly
    counter = FPSCounter()
    # No need for counter.start()

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Starting frame capture loop...")
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Process frame - call methods directly on the instance
        counter.increment_frame_count()
        fps = counter.get_fps()

        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release camera
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()

    # No need for counter.stop()
    print("Done.")