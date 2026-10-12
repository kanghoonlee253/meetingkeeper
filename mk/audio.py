"""Windows audio capture: WASAPI loopback (what you hear) + microphone (your voice).

Audio is kept in memory only and never written to disk.
"""
import queue

import numpy as np
import pyaudiowpatch as pyaudio

SOURCE_ME = "me"        # microphone
SOURCE_OTHERS = "others"  # loopback of the default output device (earphones)


class AudioCapture:
    """Opens the loopback and mic streams and pushes mono float32 chunks into queues."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._streams = []
        self.queues = {SOURCE_ME: queue.Queue(), SOURCE_OTHERS: queue.Queue()}
        self.devices = {}

    def _wasapi(self):
        return self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

    def list_devices(self):
        """Returns (loopbacks, mics, default_loopback_index, default_mic_index) for WASAPI."""
        wasapi = self._wasapi()
        loopbacks = [dict(d) for d in self._pa.get_loopback_device_info_generator()]
        mics = []
        for i in range(self._pa.get_device_count()):
            d = self._pa.get_device_info_by_index(i)
            if (d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0
                    and not d.get("isLoopbackDevice")):
                mics.append(dict(d))
        speakers = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_loop = next((d["index"] for d in loopbacks if speakers["name"] in d["name"]),
                            loopbacks[0]["index"] if loopbacks else None)
        return loopbacks, mics, default_loop, wasapi["defaultInputDevice"]

    def _open(self, source, device, is_loopback):
        channels = max(1, int(device["maxInputChannels"]))
        rate = int(device["defaultSampleRate"])
        q = self.queues[source]

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            q.put(data.copy())
            return (None, pyaudio.paContinue)

        stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=int(rate * 0.032),
            stream_callback=callback,
        )
        self._streams.append(stream)
        self.devices[source] = {"name": device["name"], "rate": rate, "loopback": is_loopback}
        return rate

    def start(self, loopback_index, mic_index):
        """Opens both streams. Returns {source: sample_rate}."""
        rates = {
            SOURCE_OTHERS: self._open(SOURCE_OTHERS, self._pa.get_device_info_by_index(loopback_index), True),
            SOURCE_ME: self._open(SOURCE_ME, self._pa.get_device_info_by_index(mic_index), False),
        }
        for s in self._streams:
            s.start_stream()
        return rates

    def stop(self):
        for s in self._streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        self._streams.clear()

    def close(self):
        self.stop()
        self._pa.terminate()
