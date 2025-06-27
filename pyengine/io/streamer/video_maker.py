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
        # 根据是否提供 filename_trunk 决定文件名的格式
        if output_trunk_name is None:
            self.filename = date_str
        elif append_date:
            self.filename = f"{output_trunk_name}_{date_str}"
        else:
            self.filename = output_trunk_name

        # 如果没有提供 FPS，则从摄像头中获取
        if not fps:
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        else:
            self.fps = fps

        # 如果没有提供宽度，则从摄像头中获取
        if not width:
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            self.width = width

        # 如果没有提供高度，则从摄像头中获取
        if not height:
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.height = height

        # 初始化 VideoWriter
        self.video = self.initialize_video_writer()

    def generated_filename(self):
        # 返回生成的视频文件名
        return self.filename

    def initialize_video_writer(self):
        # 尝试使用 MP4 编码器初始化 VideoWriter
        video = cv2.VideoWriter(self.filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (self.width, self.height))
        if video.isOpened():
            self.filename += '.mp4'
            return video

        # 如果 MP4 编码器失败，则回退到 AVI 编码器
        video = cv2.VideoWriter(self.filename + '.avi', cv2.VideoWriter_fourcc(*'XVID'),
                                self.fps,
                                (self.width, self.height))
        if video.isOpened():
            self.filename += '.avi'
            return video

        raise ValueError('Failed to initialize video writer with both MP4 and AVI codecs')

    def add_frame(self, frame):
        # 获取当前帧的尺寸
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # 检查当前帧尺寸是否与视频尺寸一致
        if frame_width != self.width or frame_height != self.height:
            # 如果不一致，则调整尺寸
            frame = cv2.resize(frame, (self.width, self.height))

        # 将帧写入视频
        self.video.write(frame)

    def release(self):
        # 如果文件顺利创建，打印文件名，否则打印错误信息
        if os.path.exists(self.filename):
            print(f'Video saved as {self.filename}')
        else:
            print('Failed to save video')

        # 释放 VideoWriter 和 VideoCapture 对象
        self.video.release()


if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError('Invalid video capture object')

    # 获取摄像头属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    # 使用 VideoMaker，注意可以传入 base_filename  参数（或者不传）
    # 例如：video_maker = VideoMaker(cap, base_filename ="myvideo", fps=fps, width=width, height=height)
    video_maker = VideoMaker(cap, fps=fps, width=width, height=height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_maker.add_frame(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_maker.release()
    cap.release()
    cv2.destroyAllWindows()
