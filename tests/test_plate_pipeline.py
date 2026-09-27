import numpy as np

from src.plate_ocr.detect_and_ocr import PlateOCRPipeline


class FakeBox:
    def __init__(self, conf=0.9):
        self.xyxy = np.array([[[0, 0, 50, 20]]], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)


class FakeResult:
    def __init__(self, frame):
        self.boxes = [FakeBox()]
        self._frame = frame

    def plot(self):
        return self._frame.copy()


class FakeYOLO:
    def __call__(self, frame, conf, verbose=False):
        return [FakeResult(frame)]


class FakeReader:
    def readtext(self, image, **kwargs):
        return [((0, 0), "ABC123", 0.99)]


def test_process_frame_detects_and_reads_plate():
    pipeline = PlateOCRPipeline.__new__(PlateOCRPipeline)
    pipeline.model = FakeYOLO()
    pipeline.reader = FakeReader()
    pipeline._read_text_from_crop = lambda crop: ["ABC123"]

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    detection = pipeline.process_frame(frame, conf_threshold=0.25)

    assert detection is not None
    assert detection["detections"][0]["texts"] == ["ABC123"]
    assert detection["detections"][0]["bbox"] == [0, 0, 50, 20]


def test_process_video_uses_capture_loop(monkeypatch):
    pipeline = PlateOCRPipeline.__new__(PlateOCRPipeline)
    pipeline.model = FakeYOLO()
    pipeline.reader = FakeReader()
    pipeline._read_text_from_crop = lambda crop: ["ABC123"]

    class FakeCapture:
        def __init__(self):
            self.frames = [
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.zeros((100, 100, 3), dtype=np.uint8),
            ]
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            self.index = len(self.frames)

    class FakeWriter:
        def __init__(self):
            self.written = 0

        def write(self, frame):
            self.written += 1

        def release(self):
            pass

    capture = FakeCapture()
    writer = FakeWriter()
    monkeypatch.setattr("cv2.VideoCapture", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr("cv2.VideoWriter", lambda *_args, **_kwargs: writer)

    pipeline.process_video(
        "demo.mp4", output_path="demo_out.mp4", show=False, max_frames=2
    )

    assert writer.written == 2
