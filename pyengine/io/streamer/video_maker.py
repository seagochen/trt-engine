import cv2
import datetime
import os


class VideoMaker:

    def __init__(self, cap, output_trunk_name=None, fps=None, width=None, height=None, append_date=True):
        if not cap or not cap.isOpened():
            raise ValueError('Invalid video capture object')
        else:
            self.cap = cap

        # 获取当前日期时间的字符串
        date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_trunk_name is None:
            self.filename_base = date_str
        elif append_date:
            self.filename_base = f"{output_trunk_name}_{date_str}"
        else:
            self.filename_base = output_trunk_name

        self.fps = int(cap.get(cv2.CAP_PROP_FPS)) if (fps is None or fps == -1) else fps
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if not width else width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if not height else height

        # 自动选择合适编码器
        self.video, self.codec_used, self.filename = self.auto_initialize_writer()

    def generated_filename(self):
        return self.filename

    def auto_initialize_writer(self):
            """
            自动尝试不同的编码器来初始化VideoWriter。
            如果尝试失败，会删除创建的空文件。
            """
            # 编码器尝试顺序(你可以按平台需求排序)
            codec_list = [
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('H264', '.mp4'),
                ('DIVX', '.avi'),
                ('AVC1', '.mp4'),
            ]

            for codec, ext in codec_list:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                filename = self.filename_base + ext
                video = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
                print(f"⚠️ Testing Codec {codec}...")
                
                if video.isOpened():
                    print(f"✅ Using codec: {codec}, saved as: {filename}")
                    return video, codec, filename
                else:
                    print(f"❌ Codec failed: {codec}")
                    # 新增代码：如果编码器初始化失败，则释放对象并删除已创建的空文件
                    video.release()
                    if os.path.exists(filename):
                        os.remove(filename)
                        print(f"🗑️ Removed empty file: {filename}")

            raise ValueError("❌ Failed to initialize VideoWriter with any known codec.")

    def add_frame(self, frame):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.video.write(frame)

    def release(self):
        self.video.release()
        if os.path.exists(self.filename):
            print(f"✅ Video saved as {self.filename}")
        else:
            print("❌ Failed to save video")


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError('Invalid video capture object')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    video_maker = VideoMaker(cap, fps=fps, width=width, height=height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_maker.add_frame(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_maker.release()
    cap.release()
    cv2.destroyAllWindows()
