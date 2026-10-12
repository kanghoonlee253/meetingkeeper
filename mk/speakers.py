"""Speaker separation for the loopback stream + the saved name dictionary.

Each utterance gets a voice embedding (NVIDIA TitaNet-small via sherpa-onnx). Utterances are grouped
online by cosine similarity into "Speaker N" clusters. When the user names a cluster, its voice is
stored in the name dictionary so the same person is recognised automatically in later meetings.
"""
import json
import os
import threading

import numpy as np
import sherpa_onnx

from mk.transcriber import MODELS_DIR

APP_DIR = os.path.dirname(MODELS_DIR)
NAMES_FILE = os.path.join(APP_DIR, "data", "names.json")
EMBED_MODEL = os.path.join(MODELS_DIR, "speaker", "nemo_en_titanet_small.onnx")

# Tuned on 3 real AMI meetings (4 speakers each, 671 utterances): 92.8% of speech time attributed
# correctly with 6-7 clusters per meeting. 0.5 / 1.0 s gave 94.5% but 15-31 clusters to name.
SAME_SPEAKER = 0.4
KNOWN_SPEAKER = 0.5       # match against saved voices in the name dictionary
SHORT_SEC = 2.0           # shorter utterances give unreliable embeddings: never start a new speaker
SHORT_MATCH = 0.25
# Speaker change inside one utterance. On the same AMI meetings this raised correctly attributed
# speech time from 87.8% to 95.5% at ~0.15 s per utterance (hop 0.5 s, 2 threads). Pieces are cut
# only where the voice changes, so one speaker's sentences stay whole for translation.
CHANGE_THRESHOLD = 0.2


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SpeakerEmbedder:
    def __init__(self, threads=1):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMBED_MODEL, num_threads=threads)
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, audio16k):
        s = self.ext.create_stream()
        s.accept_waveform(16000, audio16k)
        s.input_finished()
        return _unit(np.array(self.ext.compute(s), dtype=np.float32))

    def change_points(self, audio16k, threshold=CHANGE_THRESHOLD, win=1.5, hop=0.5, min_piece=1.0):
        """Sample offsets where the speaker changes inside one utterance (no pause between turns).
        Compares the 1.5 s before and after each candidate point; cuts where they sound like
        different people, lowest similarity first, keeping pieces >= min_piece seconds."""
        sr = 16000
        if len(audio16k) < (2 * min_piece + 0.5) * sr:
            return []
        W, H = int(win * sr), int(hop * sr)
        starts = list(range(0, len(audio16k) - W + 1, H))
        windows = [self(audio16k[s:s + W]) for s in starts]
        candidates = []
        for k, s in enumerate(starts):
            t = s + W                      # boundary: window k ends here, window j starts here
            j = t // H
            if j < len(starts) and min_piece * sr <= t <= len(audio16k) - min_piece * sr:
                candidates.append((float(np.dot(windows[k], windows[j])), t))
        cuts = []
        for sim, t in sorted(candidates):
            if sim >= threshold:
                break
            if all(abs(t - c) >= min_piece * sr for c in cuts):
                cuts.append(t)
        return sorted(cuts)


class NameBook:
    """Saved names (+ optional voiceprints), persisted to data/names.json."""

    def __init__(self, path=NAMES_FILE):
        self.path = path
        self.me = "나"
        self.entries = {}          # name -> {"voice": [floats] | None, "n": int}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        self.me = d.get("me", "나")
        self.entries = {e["name"]: {"voice": e.get("voice"), "n": e.get("n", 0)} for e in d.get("names", [])}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        d = {"me": self.me,
             "names": [{"name": k, "voice": v["voice"], "n": v["n"]} for k, v in sorted(self.entries.items())]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def names(self):
        return sorted(self.entries, key=str.lower)

    def add(self, name):
        self.entries.setdefault(name, {"voice": None, "n": 0})
        self.save()

    def remove(self, name):
        self.entries.pop(name, None)
        self.save()

    def clear_voice(self, name):
        if name in self.entries:
            self.entries[name] = {"voice": None, "n": 0}
            self.save()

    def learn_voice(self, name, centroid, count):
        """Blend a meeting's cluster centroid into the saved voiceprint (weighted by utterance count)."""
        e = self.entries.setdefault(name, {"voice": None, "n": 0})
        if e["voice"] is None:
            v, n = centroid, count
        else:
            v = _unit(np.array(e["voice"]) * e["n"] + centroid * count)
            n = e["n"] + count
        e["voice"], e["n"] = [round(float(x), 5) for x in v], min(n, 200)
        self.save()

    def match(self, emb, exclude=()):
        best, best_sim = None, KNOWN_SPEAKER
        for name, e in self.entries.items():
            if e["voice"] is None or name in exclude:
                continue
            sim = float(np.dot(emb, e["voice"]))
            if sim >= best_sim:
                best, best_sim = name, sim
        return best


class SpeakerTracker:
    """Online clustering of loopback utterances into speakers. Thread-safe."""

    def __init__(self, namebook):
        self.book = namebook
        self.lock = threading.Lock()
        self.sums = []             # cluster id -> sum of unit embeddings
        self.counts = []
        self.names = []            # cluster id -> name or None
        self.alias = {}            # merged cluster id -> surviving id
        self.last = None

    def resolve(self, cid):
        while cid in self.alias:
            cid = self.alias[cid]
        return cid

    def label(self, cid):
        cid = self.resolve(cid)
        return self.names[cid] or f"Speaker {cid + 1}"

    def live_ids(self):
        return [c for c in range(len(self.sums)) if c not in self.alias]

    def assign(self, emb, duration):
        with self.lock:
            live = [c for c in range(len(self.sums)) if c not in self.alias]
            sims = {c: float(np.dot(_unit(self.sums[c]), emb)) for c in live}
            best = max(sims, key=sims.get) if sims else None
            if duration < SHORT_SEC:
                # Too short to trust: attach to the closest/last speaker without updating the centroid.
                if best is not None and sims[best] >= SHORT_MATCH:
                    return best
                if self.last is not None:
                    return self.resolve(self.last)
            if best is not None and sims[best] >= SAME_SPEAKER:
                self.sums[best] = self.sums[best] + emb
                self.counts[best] += 1
                self.last = best
                return best
            used = {n for n in self.names if n}
            self.sums.append(emb.copy())
            self.counts.append(1)
            self.names.append(self.book.match(emb, exclude=used))
            self.last = len(self.sums) - 1
            return self.last

    def set_name(self, cid, name):
        """Names a cluster. If another cluster already has that name, the two are merged.
        Returns the surviving cluster id."""
        with self.lock:
            cid = self.resolve(cid)
            for other in range(len(self.sums)):
                if other != cid and other not in self.alias and self.names[other] == name:
                    self.sums[other] = self.sums[other] + self.sums[cid]
                    self.counts[other] += self.counts[cid]
                    self.alias[cid] = other
                    if self.last == cid:
                        self.last = other
                    return other
            self.names[cid] = name
            return cid

    def new_cluster(self):
        with self.lock:
            self.sums.append(np.zeros_like(self.sums[0]))
            self.counts.append(0)
            self.names.append(None)
            return len(self.sums) - 1

    def move_embedding(self, emb, src, dst):
        """One utterance was assigned to the wrong speaker: move its embedding."""
        with self.lock:
            src, dst = self.resolve(src), self.resolve(dst)
            if emb is None or src == dst:
                return
            if self.counts[src] > 1:
                self.sums[src] = self.sums[src] - emb
                self.counts[src] -= 1
            self.sums[dst] = emb.copy() if self.counts[dst] == 0 else self.sums[dst] + emb
            self.counts[dst] += 1

    def named_centroids(self):
        """[(name, unit centroid, count)] for every named cluster — used to learn voices at the end."""
        with self.lock:
            return [(self.names[c], _unit(self.sums[c]), self.counts[c])
                    for c in range(len(self.sums))
                    if c not in self.alias and self.names[c] and self.counts[c] > 0]
