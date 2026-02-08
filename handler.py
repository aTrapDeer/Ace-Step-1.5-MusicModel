import base64, io
import numpy as np
import soundfile as sf

class EndpointHandler:
    def __init__(self, path=""):
        # Later: load ACE-Step pipeline/model here once.
        self.ready = True

    def __call__(self, data):
        inputs = data.get("inputs", data)
        duration = int(inputs.get("duration_sec", 6))
        sr = int(inputs.get("sample_rate", 44100))
        seed = int(inputs.get("seed", 42))

        rng = np.random.default_rng(seed)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.07*np.sin(2*np.pi*440*t) + 0.01*rng.standard_normal(len(t))).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return {
            "audio_base64_wav": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "sample_rate": sr,
            "duration_sec": duration
        }
