class VideoPlugin:
    def start(self, bus) -> None: ...
    def stop(self) -> None: ...


# video_plugin_manager.py
class VideoPluginManager:
    def __init__(self, bus):
        self.bus = bus
        self._plugins = []
        self._started = False

    def register(self, plugin):
        self._plugins.append(plugin)
        if self._started:
            plugin.start(self.bus)

    def start(self):
        if self._started: return
        self._started = True
        for p in self._plugins:
            p.start(self.bus)

    def stop(self):
        for p in self._plugins:
            p.stop()
        self._started = False


"""
from video_bus import VideoBus
from video_plugin_manager import VideoPluginManager
from video_plugins import VideoRecorderPlugin, VideoInferencePlugin
import cv2

# 1) 启动视频总线(交给它管理 cap)
bus = VideoBus().start(url="rtsp://admin:xxx@192.168.1.73", width=-1, height=-1, fps=-1)

# 2) 注册插件
pm = VideoPluginManager(bus)
pm.register(VideoRecorderPlugin(filename_trunk="cam1_record", append_date=True))

def dummy_infer(frame):
    # 在这里接你真正的 AI 推理(yolo/pose 等)；演示里返回个占位
    return {"mean": float(frame.mean())}

def on_result(frame, result):
    # 比如：画个文本+显示；或发 MQTT；或落库
    txt = f"mean={result['mean']:.1f}"
    vis = frame.copy()
    cv2.putText(vis, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("infer", vis)
    cv2.waitKey(1)

pm.register(VideoInferencePlugin(infer_fn=dummy_infer, on_result=on_result, throttle_sec=0.2))

# 3) 启动所有插件
pm.start()

# 4) 主线程也可以独立显示(可选)
try:
    while True:
        # 只是保持程序运行；显示逻辑在 on_result 里
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
finally:
    pm.stop()
    bus.stop()
    cv2.destroyAllWindows()
"""