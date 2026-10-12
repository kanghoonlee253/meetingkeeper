"""Downloads the models MeetingKeeper needs into models/ (run once after cloning).

    .venv\\Scripts\\python.exe setup_models.py

NLLB (the optional fast translation engine) needs a one-off conversion with torch/transformers;
see README.md. The app works without it (Gemma is the default engine).
"""
import os
import urllib.request

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def fetch(url, dest):
    if os.path.exists(dest):
        print("있음:", dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("다운로드:", url)
    urllib.request.urlretrieve(url, dest + ".part")
    os.replace(dest + ".part", dest)


def main():
    print("1/3 음성인식 Whisper small.en")
    snapshot_download("Systran/faster-whisper-small.en", local_dir=os.path.join(ROOT, "whisper", "small.en"),
                      allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"])

    print("2/3 번역·요약 Gemma 3 4B (약 2.5GB)")
    llm = os.path.join(ROOT, "llm")
    if not os.path.exists(os.path.join(llm, "gemma-3-4b-it-q4_k_m.gguf")):
        p = hf_hub_download("bartowski/google_gemma-3-4b-it-GGUF", "google_gemma-3-4b-it-Q4_K_M.gguf", local_dir=llm)
        os.replace(p, os.path.join(llm, "gemma-3-4b-it-q4_k_m.gguf"))

    print("3/3 화자 구분 TitaNet-small")
    fetch("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
          "nemo_en_titanet_small.onnx", os.path.join(ROOT, "speaker", "nemo_en_titanet_small.onnx"))
    print("완료:", ROOT)


if __name__ == "__main__":
    main()
