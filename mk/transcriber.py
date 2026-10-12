"""Local English speech-to-text with faster-whisper (CPU, int8)."""
import os
import queue
import sys
import threading
import time

from faster_whisper import WhisperModel

# Packaged exe: models/, data/ and meetings/ live next to MeetingKeeper.exe.
APP_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
           else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(APP_DIR, "models")
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")


def local_whisper_models():
    """Whisper models shipped in models/whisper/<name> (loaded offline, no download)."""
    if not os.path.isdir(WHISPER_DIR):
        return []
    return sorted(d for d in os.listdir(WHISPER_DIR) if os.path.isfile(os.path.join(WHISPER_DIR, d, "model.bin")))

# Whisper tends to invent these on short noise/silence.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "you", "you.", ".", "bye.", "bye!", "so", "okay.",
}


class Transcriber(threading.Thread):
    def __init__(self, in_queue, on_result, on_status, model_name="small.en", cpu_threads=4,
                 speaker_tracker=None, others_source="others"):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.on_result = on_result
        self.on_status = on_status
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self.tracker = speaker_tracker
        self.others_source = others_source
        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._on_done = None

    def stop(self):
        self._stop_event.set()

    def finish(self, on_done):
        """Process everything already queued, then exit and call on_done()."""
        self._on_done = on_done
        self._finish_event.set()

    def run(self):
        local = os.path.join(WHISPER_DIR, self.model_name)
        if os.path.isfile(os.path.join(local, "model.bin")):
            self.on_status(f"음성인식 모델 로딩 중: {self.model_name}")
            source = local
        else:
            self.on_status(f"음성인식 모델 다운로드/로딩 중: {self.model_name} (처음 한 번 시간이 걸립니다)")
            source = self.model_name
        model = WhisperModel(source, device="cpu", compute_type="int8",
                             cpu_threads=self.cpu_threads, download_root=WHISPER_DIR)
        embedder = None
        if self.tracker:
            from mk.speakers import SpeakerEmbedder
            embedder = SpeakerEmbedder()
        self.on_status(f"준비 완료 — 모델 {self.model_name}, CPU 스레드 {self.cpu_threads}")
        while not self._stop_event.is_set():
            try:
                utt = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                if self._finish_event.is_set():
                    break
                continue
            t0 = time.monotonic()
            segments, _ = model.transcribe(
                utt.audio, language="en", beam_size=1, vad_filter=False,
                condition_on_previous_text=False, without_timestamps=True,
            )
            segments = list(segments)
            proc = time.monotonic() - t0
            if not segments:
                continue
            if all(s.no_speech_prob > 0.6 and s.avg_logprob < -1.0 for s in segments):
                continue
            text = " ".join(s.text.strip() for s in segments).strip()
            duration = len(utt.audio) / 16000
            if not text or (duration < 2.0 and text.lower() in HALLUCINATIONS):
                continue
            speaker, emb = None, None
            if embedder and utt.source == self.others_source:
                emb = embedder(utt.audio)
                speaker = self.tracker.assign(emb, duration)
            self.on_result({
                "source": utt.source,
                "text": text,
                "start_wall": utt.start_wall,
                "duration": duration,
                "process_sec": proc,
                "delay_sec": time.monotonic() - utt.ended_mono,
                "backlog": self.in_queue.qsize(),
                "speaker": speaker,
                "embedding": emb,
            })
        if self._on_done:
            self._on_done()
