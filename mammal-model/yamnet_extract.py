"""YAMNet feature extraction for transfer learning.

The vendored YAMNet TFLite (vendor/yamnet.tflite, same artifact + SHA-256 as the
sibling mammal-sounds module downloads at runtime) exposes ONE output: the
521 AudioSet class posteriors per 0.96 s frame. There is no embedding output
in this TFLite build, so the transfer-learning feature is the 521-dim score
vector, mean-pooled over the clip's frames -> (521,).

Runs via ai_edge_litert (already installed in this sandbox).
"""
import os
import numpy as np

from config import YAMNET_N_SCORES
from features import frame_for_yamnet

VENDOR_MODEL = "/home/hatch/workspace/model-verify/yamnet.tflite"  # DRAFT: verified 15600->521 signature


class YamnetFeatureExtractor:
    def __init__(self, model_path=VENDOR_MODEL):
        from ai_edge_litert.interpreter import Interpreter
        self._it = Interpreter(model_path=model_path)
        self._it.allocate_tensors()
        in_d = self._it.get_input_details()[0]
        out_d = self._it.get_output_details()[0]
        self._in_idx = in_d["index"]
        self._in_shape = tuple(int(s) for s in in_d["shape"])
        self._out_idx = out_d["index"]
        assert int(np.prod(self._in_shape)) == 15600, self._in_shape

    def frame_scores(self, frames):
        """frames: (n_frames, 15600) float32 -> (n_frames, 521) float32."""
        outs = np.empty((len(frames), YAMNET_N_SCORES), dtype=np.float32)
        for i, f in enumerate(frames):
            self._it.set_tensor(self._in_idx,
                                np.asarray(f, dtype=np.float32).reshape(self._in_shape))
            self._it.invoke()
            outs[i] = self._it.get_tensor(self._out_idx).reshape(-1)
        return outs

    def clip_vector(self, x16k):
        """3 s @16 kHz waveform -> (521,) mean-pooled YAMNet score vector."""
        frames = frame_for_yamnet(x16k)
        return self.frame_scores(frames).mean(axis=0).astype(np.float32)


if __name__ == "__main__":
    # smoke test: silence should give a valid 521-vector
    ex = YamnetFeatureExtractor()
    v = ex.clip_vector(np.zeros(48000, dtype=np.float32))
    print("vector shape:", v.shape, "sum:", v.sum())
    top = np.argsort(v)[-3:][::-1]
    print("top classes for silence:", top, v[top])
