"""WavLM speech-emotion backend.

An alternative to the default wav2vec2 model. WavLM-based systems won the 2024
MSP-Podcast A/D/V challenge and the literature reports WavLM generally
outperforming both wav2vec2 and HuBERT on emotion tasks, so this is worth having
available — but the claim is a benchmark result, not something the three
labelled clips here can confirm.

Measured on those clips, both models rank the calls identically by arousal. The
default clusters the two `medium` clips tightly (0.600 / 0.568) and separates the
`high` one; this model spreads the mediums much further apart (0.552 / 0.438),
which is worse within-class consistency on the only real data available. Three
examples cannot settle it either way, so neither is declared the winner: the
default stays, and this is selectable with `SER_BACKEND=wavlm`.

The published checkpoint ships only weights and expects a wrapper class from a
GitHub repository that is not on PyPI. Rather than vendor a dependency, the
architecture is rebuilt here from the checkpoint's own tensor shapes:

    WavLM-large backbone
      -> 25 hidden states (input embeddings + 24 layers)
      -> softmax-weighted sum over a learned 25-vector
      -> three 1x1 convolutions, 1024 -> 256 -> 256 -> 256, with ReLU
      -> mean pooling over time
      -> a two-layer head per dimension, sigmoid to [0, 1]

Costs against the default: about 1.2 GB more image, and roughly twice the CPU
time per window.
"""

from __future__ import annotations

import os
import threading

import numpy as np

MODEL_ID = "tiantiaf/wavlm-large-msp-podcast-emotion-dim"
BACKBONE_ID = "microsoft/wavlm-large"
HIDDEN = 1024
PROJECTION = 256

_MODEL = None
_LOCK = threading.Lock()
_FAILED = ""


def _build():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import WavLMModel

    weights = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))

    backbone = WavLMModel.from_pretrained(BACKBONE_ID)
    backbone.load_state_dict(
        {k[len("backbone_model."):]: v for k, v in weights.items()
         if k.startswith("backbone_model.")},
        strict=False,
    )
    backbone.eval()

    class Emotion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            # One learned weight per hidden state, softmaxed at inference. The
            # checkpoint carries 25 because the config sets use_conv_output,
            # which includes the feature-extractor output alongside the 24
            # transformer layers.
            self.layer_weights = nn.Parameter(weights["weights"].clone())

            self.seq = nn.Sequential(
                nn.Conv1d(HIDDEN, PROJECTION, 1), nn.ReLU(), nn.Dropout(0.1),
                nn.Conv1d(PROJECTION, PROJECTION, 1), nn.ReLU(), nn.Dropout(0.1),
                nn.Conv1d(PROJECTION, PROJECTION, 1),
            )
            for idx in (0, 3, 6):
                self.seq[idx].weight.data = weights[f"model_seq.{idx}.weight"]
                self.seq[idx].bias.data = weights[f"model_seq.{idx}.bias"]

            def head(prefix: str) -> nn.Sequential:
                block = nn.Sequential(
                    nn.Linear(PROJECTION, PROJECTION), nn.ReLU(), nn.Linear(PROJECTION, 1)
                )
                block[0].weight.data = weights[f"{prefix}.0.weight"]
                block[0].bias.data = weights[f"{prefix}.0.bias"]
                block[2].weight.data = weights[f"{prefix}.2.weight"]
                block[2].bias.data = weights[f"{prefix}.2.bias"]
                return block

            self.arousal = head("arousal_layer")
            self.valence = head("valence_layer")
            self.dominance = head("dominance_layer")

        def forward(self, waveform):
            states = self.backbone(waveform, output_hidden_states=True).hidden_states
            stacked = torch.stack(states, dim=0)
            weighting = torch.softmax(self.layer_weights, dim=0)[:, None, None, None]
            pooled = (stacked * weighting).sum(dim=0)

            projected = self.seq(pooled.transpose(1, 2)).transpose(1, 2)
            pooled = projected.mean(dim=1)

            return (
                torch.sigmoid(self.arousal(pooled)),
                torch.sigmoid(self.valence(pooled)),
                torch.sigmoid(self.dominance(pooled)),
            )

    model = Emotion()
    model.eval()

    if os.environ.get("SER_QUANTIZE", "1") != "0":
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except (RuntimeError, NotImplementedError):
            pass

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return model


def load():
    """Load once per process. Returns None when unavailable."""
    global _MODEL, _FAILED
    if _MODEL is not None or _FAILED:
        return _MODEL
    with _LOCK:
        if _MODEL is None and not _FAILED:
            try:
                _MODEL = _build()
            except Exception as exc:  # noqa: BLE001
                _FAILED = f"{type(exc).__name__}: {exc}"
    return _MODEL


def predict(window: np.ndarray) -> np.ndarray | None:
    """Arousal, dominance, valence for one window.

    Returned in the same order the default backend uses, so the two are
    interchangeable — note the checkpoint's own heads are ordered
    arousal/valence/dominance, which is not the same thing.
    """
    model = load()
    if model is None:
        return None

    import torch

    # The checkpoint was trained on per-utterance normalised input.
    centred = (window - window.mean()) / (window.std() + 1e-7)
    tensor = torch.from_numpy(centred.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        arousal, valence, dominance = model(tensor)
    return np.array([arousal.item(), dominance.item(), valence.item()], dtype=np.float64)


def failure() -> str:
    return _FAILED
