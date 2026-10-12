"""Korean meeting summary with the local Gemma 3 4B model.

Long meetings are summarised in sections first (map) and then merged (reduce), so the prompt stays
small enough for a laptop CPU.
"""
import os
import threading
import time

from mk.transcriber import MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "llm", "gemma-3-4b-it-q4_k_m.gguf")
CHUNK_CHARS = 6000          # ~1.5k tokens of English transcript per section
NOTES_MAX_CHARS = 9000      # Korean section notes that still fit the 8k context with the final answer

FORMAT = """## 한 줄 요약
(회의 전체를 한 문장으로)

## 주요 논의 내용
- (주제별로 핵심만)

## 결정 사항
- (확정된 것만. 없으면 "없음")

## 액션 아이템
- 담당자: 할 일 (기한이 언급됐으면 기한)

## 미결 사항 / 다음 논의
- (결론이 안 난 것, 다음에 다룰 것. 없으면 "없음")"""

SECTION_PROMPT = (
    "You take notes for a Korean reader on one section of an English business meeting transcript. "
    "The transcript comes from speech recognition, so expect small errors. Write in Korean. Use the "
    "speaker names exactly as written. Record only what was actually said about the meeting's work; "
    "skip greetings, small talk, jokes, equipment trouble and anything off-topic. Never invent "
    "decisions, owners or deadlines. Write short bullet points under these headings: "
    "논의, 결정, 할 일, 미결. A 할 일 is a concrete task someone agreed to do or was asked to do "
    "(write '담당자: 할 일'); a job title or role is not a task. Write '없음' under a heading with nothing."
)

FINAL_PROMPT = (
    "You write the final Korean minutes of a business meeting held in English. Write in Korean "
    "(concise 개조식, e.g. '~함', '~하기로 함'). Use the speaker names exactly as given. Use only "
    "information in the input; never invent decisions, owners or deadlines. Leave out small talk and "
    "off-topic chatter. 결정 사항 are things the group agreed on. 액션 아이템 are concrete tasks someone "
    "agreed to do or was asked to do; a job title or role is not an action item. Follow this markdown "
    "format exactly and output nothing else:\n\n" + FORMAT
)


def _chunks(text, size=CHUNK_CHARS):
    lines, cur, out = text.splitlines(), [], []
    n = 0
    for ln in lines:
        if cur and n + len(ln) > size:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        out.append("\n".join(cur))
    return out


class Summarizer(threading.Thread):
    """Runs in the background. on_progress(str), on_done(summary_markdown | None, error | None)."""

    def __init__(self, transcript, participants, on_progress, on_done, threads=None):
        super().__init__(daemon=True)
        self.transcript = transcript
        self.participants = participants
        self.on_progress = on_progress
        self.on_done = on_done
        self.threads = threads or max(4, (os.cpu_count() or 8) // 2)

    def _ask(self, llm, system, user, max_tokens):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0, max_tokens=max_tokens,
        )
        return r["choices"][0]["message"]["content"].strip()

    def run(self):
        try:
            from llama_cpp import Llama
            t0 = time.monotonic()
            self.on_progress("요약 모델 로딩 중…")
            llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_threads=self.threads, verbose=False)
            who = ", ".join(self.participants)
            parts = _chunks(self.transcript)
            if len(parts) == 1:
                self.on_progress("요약 작성 중…")
                notes = f"Participants: {who}\n\nTranscript:\n{parts[0]}"
            else:
                section_notes = []
                for i, part in enumerate(parts, 1):
                    self.on_progress(f"구간 요약 중 ({i}/{len(parts)})… {time.monotonic() - t0:.0f}초 경과")
                    section_notes.append(f"[구간 {i}]\n" + self._ask(
                        llm, SECTION_PROMPT, f"Participants: {who}\n\nTranscript section:\n{part}", 400))
                # Very long meetings: merge groups of section notes until they fit the final prompt.
                while sum(len(n) for n in section_notes) > NOTES_MAX_CHARS and len(section_notes) > 1:
                    self.on_progress(f"구간 요약 정리 중… {time.monotonic() - t0:.0f}초 경과")
                    section_notes = [
                        self._ask(llm, SECTION_PROMPT,
                                  f"Participants: {who}\n\nMerge these consecutive meeting notes:\n\n"
                                  + "\n\n".join(section_notes[i:i + 4]), 500)
                        for i in range(0, len(section_notes), 4)
                    ]
                self.on_progress(f"최종 요약 작성 중… {time.monotonic() - t0:.0f}초 경과")
                notes = (f"Participants: {who}\n\nNotes from each section of the meeting, in order:\n\n"
                         + "\n\n".join(section_notes))
            summary = self._ask(llm, FINAL_PROMPT, notes, 1200)
            self.on_done(summary, None)
        except Exception as e:
            self.on_done(None, str(e))
