"""Streaming voice-activity detection that cuts the audio into utterances."""
import collections
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime
import soxr
from faster_whisper.utils import get_assets_path

SAMPLE_RATE = 16000
FRAME = 512                      # 32 ms at 16 kHz (Silero VAD frame size)
FRAME_SEC = FRAME / SAMPLE_RATE
CONTEXT = 64


@dataclass
class Utterance:
    source: str
    audio: np.ndarray            # float32 mono 16 kHz
    start_wall: float            # time.time() when speech began
    ended_mono: float            # time.monotonic() when the utterance was closed


class StreamingVAD:
    """Frame-by-frame Silero VAD (the onnx file bundled with faster-whisper)."""

    def __init__(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame):
        x = np.concatenate([self.context, frame])[None, :].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class Segmenter(threading.Thread):
    def __init__(self, source, in_queue, in_rate, out_queue, level_cb=None,
                 start_threshold=0.5, end_threshold=0.35, min_silence=0.7,
                 max_utterance=12.0, min_utterance=0.5, preroll=0.3):
        super().__init__(daemon=True)
        self.source = source
        self.in_queue = in_queue
        self.in_rate = in_rate
        self.out_queue = out_queue
        self.level_cb = level_cb
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.silence_frames_to_end = int(min_silence / FRAME_SEC)
        self.max_frames = int(max_utterance / FRAME_SEC)
        self.min_samples = int(min_utterance * SAMPLE_RATE)
        self._preroll = collections.deque(maxlen=int(preroll / FRAME_SEC))
        self._stop_event = threading.Event()
        self._reset()

    def _reset(self):
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames = []
        self.start_wall = 0.0

    def stop(self):
        self._stop_event.set()

    def run(self):
        vad = StreamingVAD()
        resampler = soxr.ResampleStream(self.in_rate, SAMPLE_RATE, 1, dtype="float32")
        buf = np.zeros(0, dtype=np.float32)
        last_data = time.monotonic()
        last_level = 0.0
        while not self._stop_event.is_set():
            try:
                chunk = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                # Loopback delivers nothing while the output is silent: close the utterance anyway.
                if self.in_speech and time.monotonic() - last_data > 0.5:
                    self._finish()
                continue
            last_data = time.monotonic()
            if self.level_cb and last_data - last_level > 0.1:
                self.level_cb(self.source, float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0)
                last_level = last_data
            buf = np.concatenate([buf, resampler.resample_chunk(chunk)])
            while len(buf) >= FRAME:
                frame, buf = buf[:FRAME], buf[FRAME:]
                self._process(frame, vad(frame))
        if self.in_speech:
            self._finish()

    def _process(self, frame, prob):
        if not self.in_speech:
            self._preroll.append(frame)
            self.speech_run = self.speech_run + 1 if prob >= self.start_threshold else 0
            if self.speech_run >= 2:
                self.in_speech = True
                self.frames = list(self._preroll)
                self._preroll.clear()
                self.silence_run = 0
                self.start_wall = time.time() - len(self.frames) * FRAME_SEC
            return
        self.frames.append(frame)
        self.silence_run = self.silence_run + 1 if prob < self.end_threshold else 0
        if self.silence_run >= self.silence_frames_to_end or len(self.frames) >= self.max_frames:
            self._finish()

    def _finish(self):
        # Drop most of the trailing silence, keep ~200 ms.
        keep = len(self.frames) - max(0, self.silence_run - int(0.2 / FRAME_SEC))
        audio = np.concatenate(self.frames[:keep]) if keep > 0 else np.zeros(0, np.float32)
        if len(audio) >= self.min_samples:
            self.out_queue.put(Utterance(self.source, audio, self.start_wall, time.monotonic()))
        self._reset()
