import time
import cv2

from pyengine.io.streamer.stream_reader import StreamReader
from pyengine.algorithms.tracker.deepsort import DeepSORTTracker
from pyengine.algorithms.tracker.sort import SORTTracker
from pyengine.algorithms.tracker.tracker import UnifiedTrack
from pyengine.inference.c_pipeline.pose_pipeline_v1 import PosePipeline
from pyengine.inference.unified_structs.pipeline_converter import convert_pipeline_v1_to_skeletons
from pyengine.utils.logger import logger
from pyengine.visualization.generic_inference_drawer import GenericInferenceDrawer
from pyengine.io.streamer.video_maker import VideoMaker

if __name__ == "__main__":
    # Define your paths
    LIBRARY_PATH = "/home/user/projects/TrtEngineToolkits/build/lib/libjetson.so"
    YOLO_POSE_ENGINE = "/opt/models/yolov8s-pose.engine"
    EFFICIENTNET_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"  # DeepSORT *needs* features
    SCHEMA_FILE = "./configs/schema.json"
    VIDEO_PATH = "./onsite/MKG_ch24_20240319175959_20240319180516.mp4"

    pipeline = None
    tracker = None
    drawer = None
    # Replace cap with stream_reader
    stream_reader = None
    maker = None

    try:
        # Initialize C++ pipeline
        pipeline = PosePipeline(LIBRARY_PATH)
        pipeline.register_models()
        pipeline.create_pipeline(
            yolo_engine_path=YOLO_POSE_ENGINE,
            efficient_engine_path=EFFICIENTNET_ENGINE,
            yolo_max_batch=1,
            efficient_max_batch=32,
            yolo_cls_thresh=0.5,
            yolo_iou_thresh=0.7
        )
        logger.info("Main", "C++ PosePipeline initialized.")

        # --- MODIFICATION START ---
        # Initialize StreamReader instead of cv2.VideoCapture
        # Use the desired output resolution and FPS
        stream_reader = StreamReader(url=VIDEO_PATH, width=1280, height=720, fps=20)
        if not stream_reader.is_connected():
            raise IOError(f"Could not open video stream: {VIDEO_PATH}")
        logger.info("Main", f"StreamReader initialized for: {VIDEO_PATH}")
        # --- MODIFICATION END ---

        # Initialize Drawer
        drawer = GenericInferenceDrawer(SCHEMA_FILE)
        logger.info("Main", f"GenericInferenceDrawer initialized with schema: {SCHEMA_FILE}")

        # Initialize tracker
        tracker = DeepSORTTracker(max_age=100, min_hits=3, iou_threshold=0.7, reid_threshold=0.2)
        # tracker = SORTTracker(max_age=100, min_hits=3, iou_threshold=0.7)
        logger.info("Main", "Tracker initialized.")

        # --- MODIFICATION START ---
        # Initialize VideoMaker for output
        # Pass the underlying cap object from the stream_reader
        maker = VideoMaker(stream_reader.cap, "deepsort_tracking_output", width=1280, height=720, fps=20,
                           append_date=False)
        logger.info("Main", f"VideoMaker initialized for output: {maker.generated_filename()}")

        # Get video properties from the stream_reader instance
        frame_width = stream_reader.width
        frame_height = stream_reader.height
        fps = stream_reader.fps
        logger.info("Main", f"Video Resolution: {frame_width}x{frame_height}, FPS: {fps}")
        # --- MODIFICATION END ---

        frame_idx = 0
        start_time = time.time()

        # --- MODIFICATION START ---
        # Loop while the stream is connected
        while stream_reader.is_connected():
            # Read frame from the StreamReader
            frame = stream_reader.read_frame()

            # If frame is None, it might be due to FPS limiting or end of stream
            if frame is None:
                continue  # Skip this loop iteration
            # --- MODIFICATION END ---

            frame_idx += 1
            # The pipeline requires a 640x640 frame
            frame_for_pipeline = cv2.resize(frame, (640, 640))
            raw_pipeline_results = pipeline.process_batched_images([frame_for_pipeline], 1.0)

            current_frame_skeletons = []
            if raw_pipeline_results:
                all_skeletons_from_pipeline = convert_pipeline_v1_to_skeletons(raw_pipeline_results)
                if all_skeletons_from_pipeline:
                    current_frame_skeletons = all_skeletons_from_pipeline[0]

            tracked_objects = tracker.update(current_frame_skeletons)

            display_frame = frame.copy()
            # The original shape for drawing is now the StreamReader's output size
            original_shape = (frame_height, frame_width)

            epsilon = 15.0
            for skeleton_det in current_frame_skeletons:
                matched_track_id = -1
                for track_id, tracked_rect in tracked_objects.items():
                    if (abs(skeleton_det.rect.x1 - tracked_rect.x1) < epsilon and
                            abs(skeleton_det.rect.y1 - tracked_rect.y1) < epsilon and
                            abs(skeleton_det.rect.x2 - tracked_rect.x2) < epsilon and
                            abs(skeleton_det.rect.y2 - tracked_rect.y2) < epsilon):
                        matched_track_id = track_id
                        break
                skeleton_det.track_id = matched_track_id

            display_frame = drawer.draw_skeletons_batch(
                display_frame,
                current_frame_skeletons,
                original_shape,
                enable_track_id=True,
                label_names=['person', 'person'],
                enable_pts_names=False,
                enable_skeleton=False
            )

            elapsed_time = time.time() - start_time
            current_fps = frame_idx / elapsed_time
            cv2.putText(display_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # This resize is now redundant if StreamReader is set to 1280x720, but it's harmless.
            display_frame = cv2.resize(display_frame, (1280, 720))
            cv2.imshow("Tracking Demo", display_frame)

            if maker:
                maker.add_frame(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Main", "Quit key 'q' pressed. Exiting.")
                break

        logger.info("Main", "End of video stream.")

    except Exception as e:
        logger.error_trace("Main", f"An error occurred: {e}")

    finally:
        # Release resources
        # --- MODIFICATION START ---
        if stream_reader:
            stream_reader.close_camera_stream()
            logger.info("Main", "StreamReader released.")
        # --- MODIFICATION END ---

        if pipeline:
            pipeline.destroy_pipeline()
            logger.info("Main", "Pipeline destroyed.")

        if maker:
            maker.release()
            logger.info("Main", "VideoMaker released.")

        # --- 2. ADD THIS EVALUATION LOG ---
        # Each new track increments the `_next_id`, so its final value is the total count.
        total_trackers_created = UnifiedTrack._next_id
        logger.info("Evaluation", f"A total of {total_trackers_created} unique trackers were created.")
        # ------------------------------------

        cv2.destroyAllWindows()
        logger.info("Main", "Program finished.")