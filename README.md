# MeetingKeeper

영어 회의(팀즈 등)를 **실시간 영어 자막 + 한글 번역**으로 보여주고, 끝나면 **회의록(전문 + 한글 요약)** 을 저장하는 Windows 프로그램.
모든 처리는 로컬에서만 이루어집니다 (인터넷·API·비용 없음, 음성은 저장하지 않음).

## 기능
- 이어폰으로 나오는 소리(WASAPI 루프백) = 상대방, 마이크 = 나 — 동시에 캡처
- 실시간 영어 자막 (faster-whisper `small.en`, CPU int8)
- 한글 번역 (Gemma 3 4B / NLLB 1.3B, llama.cpp · CTranslate2)
- 상대 화자 자동 구분 (TitaNet-small 음성 임베딩 + 온라인 군집화), 이름 지정·이름 사전·목소리 기억
- 회의 종료 후 `YYYYMMDD_NN_회의록.docx`, `YYYYMMDD_NN_요약.txt` 저장 (한글 요약: Gemma 3 4B, 구간 요약 → 통합)

## 개발 환경 설정
Python 3.11 (64-bit) 기준.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv\Scripts\python.exe setup_models.py
.venv\Scripts\python.exe app.py
```

### (선택) NLLB 빠른 번역 엔진
별도 가상환경에서 변환합니다 (torch 필요, 한 번만).
```powershell
py -3.11 -m venv convenv
convenv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
convenv\Scripts\python.exe -m pip install "transformers<5" ctranslate2 sentencepiece protobuf
convenv\Scripts\ct2-transformers-converter.exe --model facebook/nllb-200-distilled-1.3B --quantization int8 --output_dir models\nllb-1.3b-ct2 --copy_files sentencepiece.bpe.model
```

## 배포용 빌드
```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```
`dist\MeetingKeeper.zip` (exe + 모델, 약 4.5GB) 이 만들어집니다. 사용법은 그 안의 `사용설명서.txt`.
점검: `MeetingKeeper.exe --selftest` → `data\selftest.txt`

## 구조
| 파일 | 역할 |
|---|---|
| `app.py` | 화면 (PySide6), 시작/정지/저장 흐름, 자체 점검 |
| `mk/audio.py` | 루프백 + 마이크 캡처 |
| `mk/segmenter.py` | Silero VAD 로 발화 단위 분할 |
| `mk/transcriber.py` | 음성인식 + 화자 배정 |
| `mk/speakers.py` | 화자 임베딩, 온라인 군집화, 이름 사전 |
| `mk/translator.py` | 영→한 번역 |
| `mk/summarizer.py` | 한글 요약 |
| `mk/meeting.py` | 자동 저장, 회의록 파일 |

## 모델 라이선스
Gemma 3 (Gemma Terms of Use), Whisper (MIT), TitaNet (CC-BY-4.0), NLLB-200 (CC-BY-NC-4.0, 비상업용).

---

Copyright (c) 2026 이강훈. All rights reserved.
