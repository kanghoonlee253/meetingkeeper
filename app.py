"""MeetingKeeper — 실시간 영어 자막 + 한글 번역 + 화자 구분 + 전문/요약 저장."""
import html
import math
import os
import queue
import sys
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from mk.audio import SOURCE_ME, SOURCE_OTHERS, AudioCapture
from mk.meeting import DEFAULT_OUTPUT_DIR, Meeting, load_settings, save_settings
from mk.segmenter import Segmenter
from mk.speakers import NameBook, SpeakerTracker
from mk.summarizer import Summarizer
from mk.transcriber import APP_DIR, Transcriber, local_whisper_models
from mk.translator import ENGINES, Translator

MODELS = local_whisper_models() or ["small.en", "distil-small.en", "distil-medium.en", "medium.en"]
ME_COLOR = "#2e7d32"
OTHERS_COLOR = "#1565c0"
SPEAKER_COLORS = ["#1565c0", "#c62828", "#6a1b9a", "#ef6c00", "#00838f", "#ad1457", "#5d4037", "#283593"]


class Bridge(QObject):
    result = Signal(dict)
    translated = Signal(int, str, float)
    status = Signal(str)
    level = Signal(str, float)
    transcribe_done = Signal()
    translate_done = Signal()
    summary_progress = Signal(str)
    summary_done = Signal(object, object)


class LineCard(QFrame):
    """One utterance: speaker + time, English, Korean (filled in when translation arrives)."""

    def __init__(self, r, on_speaker_click):
        super().__init__()
        self.source = r["source"]
        self.cid = r.get("speaker")
        self.embedding = r.get("embedding")
        self.start_wall = r["start_wall"]
        self.delay = r["delay_sec"]
        self.en_text = r["text"]
        self.ko_text = None
        self.speaker_label = ""
        self.header = QLabel()
        self.header.linkActivated.connect(lambda _: on_speaker_click(self))
        self.en = QLabel(r["text"])
        self.en.setWordWrap(True)
        self.en.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.en.setStyleSheet("font-size:15px;")
        self.ko = QLabel("번역 중…")
        self.ko.setWordWrap(True)
        self.ko.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.ko.setStyleSheet("font-size:15px;color:#aaa;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(2)
        lay.addWidget(self.header)
        lay.addWidget(self.en)
        lay.addWidget(self.ko)

    def set_speaker(self, label, color, clickable):
        self.speaker_label = label
        t = time.strftime("%H:%M:%S", time.localtime(self.start_wall))
        name = f"<b style='color:{color}'>[{html.escape(label)}]</b>"
        if clickable:
            name = f"<a href='spk' style='text-decoration:none'>{name} <span style='color:#999'>▾</span></a>"
        self.header.setText(f"{name} <span style='color:#999;font-size:12px'>{t} · 자막 {self.delay:.1f}s</span>")

    def set_korean(self, text):
        self.ko_text = text
        self.ko.setText(text)
        self.ko.setStyleSheet("font-size:15px;color:#444;")


class NameDialog(QDialog):
    """이름 사전: 내 이름, 자주 만나는 사람 이름, 저장된 목소리 관리."""

    def __init__(self, book, parent=None):
        super().__init__(parent)
        self.book = book
        self.setWindowTitle("이름 사전")
        self.resize(360, 420)
        self.me = QLineEdit(book.me)
        self.list = QListWidget()
        add = QPushButton("추가")
        add.clicked.connect(self.add)
        remove = QPushButton("삭제")
        remove.clicked.connect(self.remove)
        reset = QPushButton("목소리 초기화")
        reset.setToolTip("저장된 목소리를 지워서 다음 회의부터 다시 학습합니다")
        reset.clicked.connect(self.reset_voice)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)

        me_row = QHBoxLayout()
        me_row.addWidget(QLabel("내 이름"))
        me_row.addWidget(self.me, 1)
        btns = QHBoxLayout()
        for b in (add, remove, reset):
            btns.addWidget(b)
        btns.addStretch()
        btns.addWidget(close)
        hint = QLabel("🎤 = 목소리 저장됨. 회의 중 이름을 지정하면 회의가 끝날 때 목소리를 기억해서 "
                      "다음 회의부터 자동으로 이름이 붙습니다.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#777")
        lay = QVBoxLayout(self)
        lay.addLayout(me_row)
        lay.addWidget(self.list, 1)
        lay.addWidget(hint)
        lay.addLayout(btns)
        self.refresh()

    def refresh(self):
        self.list.clear()
        for name in self.book.names():
            mark = "🎤 " if self.book.entries[name]["voice"] else "     "
            item = QListWidgetItem(mark + name)
            item.setData(Qt.UserRole, name)
            self.list.addItem(item)

    def _selected(self):
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def add(self):
        name, ok = QInputDialog.getText(self, "이름 추가", "이름:")
        if ok and name.strip():
            self.book.add(name.strip())
            self.refresh()

    def remove(self):
        name = self._selected()
        if name and QMessageBox.question(self, "삭제", f"'{name}'을(를) 삭제할까요?") == QMessageBox.Yes:
            self.book.remove(name)
            self.refresh()

    def reset_voice(self):
        name = self._selected()
        if name:
            self.book.clear_voice(name)
            self.refresh()

    def accept(self):
        me = self.me.text().strip()
        if me and me != self.book.me:
            self.book.me = me
            self.book.save()
        super().accept()


class SaveDialog(QDialog):
    def __init__(self, title, has_summary, parent=None):
        super().__init__(parent)
        self.setWindowTitle("회의 저장")
        self.title = QLineEdit(title)
        self.title.setPlaceholderText("예: 주간 마케팅 회의")
        self.summarize = QCheckBox("한글 요약 만들기 (회의 길이에 따라 몇 분~십여 분 걸립니다)")
        self.summarize.setChecked(not has_summary)
        if has_summary:
            self.summarize.setText("한글 요약 다시 만들기 (이름을 고쳤다면 체크)")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("회의 제목"))
        lay.addWidget(self.title)
        lay.addWidget(self.summarize)
        lay.addWidget(buttons)
        self.resize(420, 140)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeetingKeeper")
        self.resize(780, 700)
        self.bridge = Bridge()
        self.bridge.result.connect(self.add_line)
        self.bridge.translated.connect(self.set_translation)
        self.bridge.status.connect(self.set_status)
        self.bridge.level.connect(self.set_level)
        self.bridge.transcribe_done.connect(self._on_transcribe_done)
        self.bridge.translate_done.connect(self._on_translate_done)
        self.bridge.summary_progress.connect(self.set_status)
        self.bridge.summary_done.connect(self._on_summary_done)

        self.settings = load_settings()
        self.meeting = None
        self.summarizer = None
        self._draining = False
        self._dirty = False
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(5000)

        self.book = NameBook()
        self.tracker = None
        self.capture = None
        self.segmenters = []
        self.transcriber = None
        self.translator = None
        self.cards = {}            # line id -> LineCard
        self._next_id = 0
        self.delays = []
        self.ko_delays = []

        self.model_box = QComboBox()
        self.model_box.addItems(MODELS)
        self.engine_box = QComboBox()
        for key, label in ENGINES.items():
            self.engine_box.addItem(label, key)
        self.laptop_mode = QCheckBox("CPU 절약")
        self.laptop_mode.setToolTip("음성인식·번역에 CPU 4스레드씩만 사용 (팀즈와 함께 쓸 때 권장). "
                                    "끄면 더 빠르지만 팀즈가 버벅일 수 있습니다.")
        self.laptop_mode.setChecked(True)
        self.on_top = QCheckBox("항상 위")
        self.on_top.toggled.connect(self.toggle_on_top)
        names_btn = QPushButton("이름 사전")
        names_btn.clicked.connect(self.open_names)
        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.toggle)

        top = QHBoxLayout()
        top.addWidget(QLabel("음성인식"))
        top.addWidget(self.model_box)
        top.addWidget(QLabel("번역"))
        top.addWidget(self.engine_box)
        top.addWidget(self.laptop_mode)
        top.addWidget(self.on_top)
        top.addStretch()
        top.addWidget(names_btn)
        top.addWidget(self.start_btn)

        self.loop_box = QComboBox()
        self.mic_box = QComboBox()
        refresh = QPushButton("새로고침")
        refresh.clicked.connect(self.load_devices)
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("상대 소리(출력)"))
        dev_row.addWidget(self.loop_box, 1)
        dev_row.addWidget(QLabel("내 마이크"))
        dev_row.addWidget(self.mic_box, 1)
        dev_row.addWidget(refresh)

        self.meters = {}
        meters = QVBoxLayout()
        for src, name, color in ((SOURCE_OTHERS, "상대", OTHERS_COLOR), (SOURCE_ME, "나", ME_COLOR)):
            row = QHBoxLayout()
            lab = QLabel(f"<b style='color:{color}'>{name}</b>")
            lab.setFixedWidth(40)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            row.addWidget(lab)
            row.addWidget(bar, 1)
            meters.addLayout(row)
            self.meters[src] = bar

        self.lines = QVBoxLayout()
        self.lines.setAlignment(Qt.AlignTop)
        holder = QWidget()
        holder.setLayout(self.lines)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(holder)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._autoscroll)
        self._stick_bottom = True
        self.scroll.verticalScrollBar().valueChanged.connect(self._track_scroll)

        self.status = QLabel("대기 중 — 상대 화자 이름 옆 ▾ 를 누르면 이름을 지정할 수 있습니다")
        self.stats = QLabel("")
        self.stats.setStyleSheet("color:#777")

        self.save_btn = QPushButton("회의 저장 (전문 + 요약)")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_meeting)
        open_btn = QPushButton("회의 폴더 열기")
        open_btn.clicked.connect(self.open_folder)
        where_btn = QPushButton("저장 위치…")
        where_btn.clicked.connect(self.choose_output_dir)
        bottom = QHBoxLayout()
        bottom.addWidget(self.save_btn)
        bottom.addStretch()
        bottom.addWidget(open_btn)
        bottom.addWidget(where_btn)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(dev_row)
        layout.addLayout(meters)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.stats)
        layout.addLayout(bottom)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.load_devices()

    # --- devices / options -------------------------------------------------
    def load_devices(self):
        cap = AudioCapture()
        try:
            loopbacks, mics, def_loop, def_mic = cap.list_devices()
        finally:
            cap.close()
        for box, devices, default in ((self.loop_box, loopbacks, def_loop), (self.mic_box, mics, def_mic)):
            box.clear()
            for d in devices:
                box.addItem(d["name"].replace(" [Loopback]", ""), d["index"])
            pos = box.findData(default)
            if pos >= 0:
                box.setCurrentIndex(pos)

    def toggle_on_top(self, on):
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        self.show()

    def _set_controls_enabled(self, on):
        for w in (self.model_box, self.engine_box, self.loop_box, self.mic_box, self.laptop_mode):
            w.setEnabled(on)

    def open_names(self):
        NameDialog(self.book, self).exec()
        self.refresh_speakers()

    # --- start / stop ------------------------------------------------------
    def toggle(self):
        if self.capture:
            self.stop()
        else:
            self.start()

    def start(self):
        if self._draining or self.summarizer:
            self.set_status("이전 회의 처리가 끝난 뒤에 시작할 수 있습니다")
            return
        threads = 4 if self.laptop_mode.isChecked() else max(2, ((os.cpu_count() or 4) - 2) // 2)
        self.finalize_meeting()
        self.clear_lines()
        self.tracker = SpeakerTracker(self.book)
        self.meeting = Meeting(self.output_dir())
        self.save_btn.setEnabled(False)
        utt_queue = queue.Queue()
        self.transcriber = Transcriber(
            utt_queue, self.bridge.result.emit, self.bridge.status.emit,
            model_name=self.model_box.currentText(), cpu_threads=threads,
            speaker_tracker=self.tracker, others_source=SOURCE_OTHERS,
        )
        self.translator = Translator(
            self.bridge.translated.emit, self.bridge.status.emit,
            engine=self.engine_box.currentData(), threads=threads,
        )
        self.transcriber.start()
        self.translator.start()
        try:
            self.capture = AudioCapture()
            rates = self.capture.start(self.loop_box.currentData(), self.mic_box.currentData())
        except Exception as e:
            self.set_status(f"오디오 장치 오류: {e}")
            if self.capture:
                self.capture.close()
            self.capture = None
            self.transcriber.stop()
            self.translator.stop()
            return
        for src, rate in rates.items():
            seg = Segmenter(src, self.capture.queues[src], rate, utt_queue, self.bridge.level.emit)
            seg.start()
            self.segmenters.append(seg)
        self.start_btn.setText("정지")
        self._set_controls_enabled(False)

    def stop(self, graceful=True):
        for seg in self.segmenters:
            seg.stop()
        for seg in self.segmenters:
            seg.join(timeout=1)
        self.segmenters.clear()
        if self.capture:
            self.capture.close()
            self.capture = None
        for bar in self.meters.values():
            bar.setValue(0)
        self.start_btn.setText("시작")
        self._set_controls_enabled(True)
        if self.meeting and self.meeting.ended is None:
            self.meeting.ended = time.time()
        if graceful and self.transcriber:
            # Finish the sentences still queued: transcription first, then translation.
            self._draining = True
            self.start_btn.setEnabled(False)
            self.set_status("남은 문장 처리 중…")
            self.transcriber.finish(self.bridge.transcribe_done.emit)
            return
        for worker in (self.transcriber, self.translator):
            if worker:
                worker.stop()
        self.transcriber = self.translator = None

    def _on_transcribe_done(self):
        self.transcriber = None
        if self.translator:
            self.set_status("남은 번역 처리 중…")
            self.translator.finish(self.bridge.translate_done.emit)
        else:
            self._on_translate_done()

    def _on_translate_done(self):
        self.translator = None
        self._draining = False
        self.start_btn.setEnabled(True)
        self.autosave()
        self.save_btn.setEnabled(bool(self.cards))
        self.set_status("회의 종료 — 화자 이름을 확인한 뒤 [회의 저장]을 누르세요 (전문은 이미 자동 저장됨)")

    def finalize_meeting(self):
        """Once per meeting (next start / app close): remember the voices of named speakers.
        Done late so names fixed after stopping are learned too."""
        if not self.tracker:
            return []
        learned = []
        for name, centroid, count in self.tracker.named_centroids():
            if count >= 2:
                self.book.learn_voice(name, centroid, count)
                learned.append(name)
        self.tracker = None
        return learned

    def clear_lines(self):
        for card in self.cards.values():
            card.deleteLater()
        self.cards.clear()
        self.delays, self.ko_delays = [], []
        self.stats.setText("")

    # --- speakers ----------------------------------------------------------
    def _style_card(self, card):
        if card.source == SOURCE_ME:
            card.set_speaker(self.book.me, ME_COLOR, False)
        elif card.cid is None or not self.tracker:
            card.set_speaker("상대", OTHERS_COLOR, False)
        else:
            cid = self.tracker.resolve(card.cid)
            card.set_speaker(self.tracker.label(cid), SPEAKER_COLORS[cid % len(SPEAKER_COLORS)], True)

    def refresh_speakers(self):
        for card in self.cards.values():
            self._style_card(card)
        self._dirty = True

    def speaker_menu(self, card):
        if card.source != SOURCE_OTHERS or card.cid is None or not self.tracker:
            return
        t = self.tracker
        cid = t.resolve(card.cid)
        menu = QMenu(self)
        menu.addSection(f"'{t.label(cid)}'의 모든 발언에 이름 지정")
        for name in self.book.names():
            act = menu.addAction(name)
            act.triggered.connect(lambda _=False, n=name: self.name_speaker(cid, n))
        menu.addAction("✎ 새 이름 입력…").triggered.connect(lambda: self.name_speaker_new(cid))
        menu.addSeparator()
        sub = menu.addMenu("이 문장만 다른 화자로")
        for other in t.live_ids():
            if other != cid:
                sub.addAction(t.label(other)).triggered.connect(
                    lambda _=False, o=other: self.move_card(card, o))
        sub.addAction("새 화자").triggered.connect(lambda: self.move_card(card, t.new_cluster()))
        menu.exec(QCursor.pos())

    def name_speaker(self, cid, name):
        self.book.add(name)
        self.tracker.set_name(cid, name)
        self.refresh_speakers()

    def name_speaker_new(self, cid):
        name, ok = QInputDialog.getText(self, "새 이름", "이 화자의 이름:")
        if ok and name.strip():
            self.name_speaker(cid, name.strip())

    def move_card(self, card, dst):
        self.tracker.move_embedding(card.embedding, card.cid, dst)
        card.cid = dst
        self._style_card(card)
        self._dirty = True

    # --- updates from worker threads --------------------------------------
    def set_status(self, text):
        self.status.setText(text)

    def set_level(self, src, rms):
        v = 0 if rms <= 1e-4 else int(max(0, min(100, (math.log10(rms) + 3) / 2.5 * 100)))
        self.meters[src].setValue(v)

    def add_line(self, r):
        card = LineCard(r, self.speaker_menu)
        card.ready_mono = time.monotonic()
        self._style_card(card)
        line_id = self._next_id
        self._next_id += 1
        self.cards[line_id] = card
        self.lines.addWidget(card)
        self.delays.append(r["delay_sec"])
        if self.translator:
            self.translator.submit(line_id, r["text"])
        self._dirty = True
        self._update_stats()

    def set_translation(self, line_id, ko, seconds):
        card = self.cards.get(line_id)
        if card is None:
            return
        card.set_korean(ko)
        self._dirty = True
        # total time from end of speech to Korean on screen
        self.ko_delays.append(card.delay + (time.monotonic() - card.ready_mono))
        self._update_stats()

    def _update_stats(self):
        parts = [f"문장 {len(self.delays)}개"]
        if self.delays:
            parts.append(f"영어 자막 평균 {sum(self.delays) / len(self.delays):.1f}s")
        if self.ko_delays:
            parts.append(f"한글 번역 평균 {sum(self.ko_delays) / len(self.ko_delays):.1f}s "
                         f"(최대 {max(self.ko_delays):.1f}s)")
        pending = len(self.delays) - len(self.ko_delays)
        if pending > 1:
            parts.append(f"번역 대기 {pending}개")
        if self.tracker:
            parts.append(f"상대 화자 {len(self.tracker.live_ids())}명")
        self.stats.setText(" · ".join(parts))

    # --- saving ------------------------------------------------------------
    def output_dir(self):
        return self.settings.get("output_dir") or DEFAULT_OUTPUT_DIR

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "회의록 저장 위치", self.output_dir())
        if d:
            self.settings["output_dir"] = d
            save_settings(self.settings)
            self.set_status(f"저장 위치: {d} (다음 회의부터 적용)")

    def open_folder(self):
        folder = self.meeting.folder if self.meeting and os.path.isdir(self.meeting.folder) else self.output_dir()
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)

    def _sync_meeting(self):
        self.meeting.lines = [
            {"time": c.start_wall, "speaker": c.speaker_label, "me": c.source == SOURCE_ME,
             "en": c.en_text, "ko": c.ko_text}
            for c in self.cards.values()
        ]

    def autosave(self):
        if not self.meeting or not self._dirty or not self.cards:
            return
        self._sync_meeting()
        try:
            self.meeting.autosave()
            self._dirty = False
        except OSError as e:
            self.set_status(f"자동 저장 실패: {e}")

    def save_meeting(self):
        if not self.meeting or not self.cards:
            return
        dlg = SaveDialog(self.meeting.title, bool(self.meeting.summary), self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._sync_meeting()
        self.meeting.set_title(dlg.title.text())
        if not dlg.summarize.isChecked():
            self._write_final()
            return
        self.save_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.summarizer = Summarizer(
            self.meeting.english_transcript(), self.meeting.participants(),
            self.bridge.summary_progress.emit, self.bridge.summary_done.emit,
        )
        self.summarizer.start()

    def _on_summary_done(self, summary, error):
        self.summarizer = None
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        if error:
            self.set_status(f"요약 실패: {error} — 전문만 저장합니다")
        else:
            self.meeting.summary = summary
        self._write_final()

    def _write_final(self):
        try:
            folder = self.meeting.save_final()
        except OSError as e:
            self.set_status(f"저장 실패: {e}")
            return
        self._dirty = False
        self.set_status(f"저장 완료: {folder}")
        os.startfile(folder)

    # --- scrolling -----------------------------------------------------------
    def _track_scroll(self, value):
        bar = self.scroll.verticalScrollBar()
        self._stick_bottom = value >= bar.maximum() - 30

    def _autoscroll(self, _min, maximum):
        if self._stick_bottom:
            QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(maximum))

    def closeEvent(self, event):
        if self.summarizer and QMessageBox.question(
                self, "종료", "요약을 만드는 중입니다. 그래도 종료할까요? (전문은 저장되어 있습니다)") != QMessageBox.Yes:
            event.ignore()
            return
        self.stop(graceful=False)
        self.autosave()
        self.finalize_meeting()
        super().closeEvent(event)


def _log_errors():
    """Uncaught errors go to data/error.log (the packaged exe has no console)."""
    import traceback
    log = os.path.join(APP_DIR, "data", "error.log")

    def hook(exc_type, exc, tb):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(time.strftime("\n[%Y-%m-%d %H:%M:%S]\n") + "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook
    import threading
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def selftest():
    """MeetingKeeper.exe --selftest : loads every model once and writes data/selftest.txt."""
    import numpy as np
    out = []

    def step(name, fn):
        t0 = time.monotonic()
        try:
            res = fn()
            out.append(f"OK   {name} ({time.monotonic() - t0:.1f}s) {res or ''}")
        except Exception as e:
            out.append(f"FAIL {name}: {e!r}")

    def audio():
        cap = AudioCapture()
        try:
            loops, mics, _, _ = cap.list_devices()
            return f"출력 {len(loops)}개, 마이크 {len(mics)}개"
        finally:
            cap.close()

    def whisper():
        from faster_whisper import WhisperModel
        from mk.transcriber import WHISPER_DIR
        m = WhisperModel(os.path.join(WHISPER_DIR, MODELS[0]), device="cpu", compute_type="int8", cpu_threads=4)
        list(m.transcribe(np.zeros(16000, np.float32), language="en")[0])
        return MODELS[0]

    def vad():
        from mk.segmenter import StreamingVAD
        StreamingVAD()(np.zeros(512, np.float32))

    def speaker():
        from mk.speakers import SpeakerEmbedder
        return f"dim {len(SpeakerEmbedder()(np.random.randn(32000).astype(np.float32) * 0.1))}"

    def translate(engine):
        from mk import translator
        t = translator._Gemma(4) if engine == "gemma" else translator._Nllb(4)
        return t("We need to finalize the budget by Friday.")

    def word():
        import tempfile
        m = Meeting(tempfile.mkdtemp())
        m.lines = [{"time": time.time(), "speaker": "Test", "me": False, "en": "Hello.", "ko": "안녕하세요."}]
        m.summary = "## 한 줄 요약\n테스트"
        return os.path.basename(m.save_final())

    step("오디오 장치", audio)
    step("음성 구간 감지(VAD)", vad)
    step("음성인식(Whisper)", whisper)
    step("화자 구분", speaker)
    step("번역 Gemma", lambda: translate("gemma"))
    step("번역 NLLB", lambda: translate("nllb"))
    step("회의록 저장(Word)", word)
    path = os.path.join(APP_DIR, "data", "selftest.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n") + f"폴더: {APP_DIR}\n\n" + "\n".join(out) + "\n")
    return 0 if all(line.startswith("OK") for line in out) else 1


def main():
    _log_errors()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
