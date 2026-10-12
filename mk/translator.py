"""Local English → Korean translation (Gemma 3 4B via llama.cpp, or NLLB via CTranslate2)."""
import os
import queue
import threading
import time

from mk.transcriber import MODELS_DIR

ENGINES = {
    "gemma": "Gemma 3 4B (정확)",
    "nllb": "NLLB 1.3B (빠름)",
}

# Each utterance is translated on its own: a 4B model tends to blend earlier lines into the output
# when given surrounding context.
SYSTEM_PROMPT = (
    "You are a live interpreter for a business meeting held in English. Translate the utterance inside "
    "<utterance> into natural, concise Korean as spoken in a Korean company meeting (polite 존댓말). "
    "Keep business jargon natural (deck=발표자료, bandwidth=여력, circle back=다시 논의). "
    "Keep people's names and product names in their original spelling. "
    "Output only the Korean translation, with no notes or parentheses."
)


class _Gemma:
    def __init__(self, threads):
        from llama_cpp import Llama
        path = os.path.join(MODELS_DIR, "llm", "gemma-3-4b-it-q4_k_m.gguf")
        self.llm = Llama(model_path=path, n_ctx=2048, n_threads=threads, verbose=False)

    def __call__(self, text):
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"<utterance>{text}</utterance>"}],
            temperature=0, max_tokens=200,
        )
        return r["choices"][0]["message"]["content"].strip()


class _Nllb:
    def __init__(self, threads):
        import ctranslate2
        import sentencepiece as spm
        path = os.path.join(MODELS_DIR, "nllb-1.3b-ct2")
        self.tr = ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=threads)
        # sentencepiece can't open non-ASCII paths on Windows (e.g. a Korean user folder): pass the bytes.
        with open(os.path.join(path, "sentencepiece.bpe.model"), "rb") as f:
            self.sp = spm.SentencePieceProcessor(model_proto=f.read())

    def __call__(self, text):
        src = ["eng_Latn"] + self.sp.encode(text, out_type=str) + ["</s>"]
        r = self.tr.translate_batch([src], target_prefix=[["kor_Hang"]], beam_size=2, max_decoding_length=256)
        return self.sp.decode(r[0].hypotheses[0][1:])


class Translator(threading.Thread):
    """Consumes (line_id, english_text) and reports (line_id, korean_text, seconds)."""

    def __init__(self, on_result, on_status, engine="gemma", threads=4):
        super().__init__(daemon=True)
        self.in_queue = queue.Queue()
        self.on_result = on_result
        self.on_status = on_status
        self.engine = engine
        self.threads = threads
        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._on_done = None

    def submit(self, line_id, text):
        self.in_queue.put((line_id, text))

    def stop(self):
        self._stop_event.set()

    def finish(self, on_done):
        """Process everything already queued, then exit and call on_done()."""
        self._on_done = on_done
        self._finish_event.set()

    def run(self):
        self.on_status(f"번역 모델 로딩 중: {ENGINES[self.engine]}")
        translate = _Gemma(self.threads) if self.engine == "gemma" else _Nllb(self.threads)
        self.on_status(f"번역 준비 완료 — {ENGINES[self.engine]}")
        while not self._stop_event.is_set():
            try:
                line_id, text = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                if self._finish_event.is_set():
                    break
                continue
            t0 = time.monotonic()
            try:
                ko = translate(text)
            except Exception as e:
                ko = f"(번역 실패: {e})"
            self.on_result(line_id, ko, time.monotonic() - t0)
        if self._on_done:
            self._on_done()
