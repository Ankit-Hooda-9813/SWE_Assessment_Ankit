"""Prosodic features describing *how* the speech sounds.

This is the only acoustic evidence the language model sees in hybrid mode, where
the audio itself never leaves the container. The vector is deliberately small and
interpretable, and it is rendered into plain English before being put in a
prompt — a model reasons better about "pitch range is wide, 2.1x the typical
conversational span" than about a raw standard deviation in hertz.

One rule is enforced downstream in fusion and stated again here because it is
the failure mode the brief calls out by name: loudness on its own is not
evidence of frustration. These features describe variability and dynamics, and
absolute level is reported only as context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.audio.features import EPS, FrameBank, frame_pitch
from app.audio.vad import VadResult


@dataclass
class Prosody:
    f0_median_hz: float
    f0_range_semitones: float      # 5th-95th percentile spread of voiced pitch
    f0_variability: float          # coefficient of variation
    energy_range_db: float         # dynamic range across speech
    energy_variability: float
    speech_rate_proxy: float       # voiced-onset rate per second
    pause_ratio: float             # share of the conversation spent not talking
    mean_pause_sec: float
    longest_pause_sec: float
    speech_level_dbfs: float
    voiced_fraction: float
    jitter: float                  # cycle-to-cycle pitch instability
    shimmer: float                 # cycle-to-cycle amplitude instability

    def to_dict(self) -> dict:
        return {k: round(float(v), 3) for k, v in asdict(self).items()}

    def describe(self) -> str:
        """Render the vector for the prompt as measurements, not adjectives.

        An earlier version wrote this as prose — "the voice is unsteady, with
        noticeable pitch tremor". Measured against the labelled clips, the model
        simply restated those adjectives back as a diagnosis of distress,
        regardless of what was actually said. Loaded wording in an evidence
        block does not inform a judgement, it makes it.

        So each figure is now reported next to its typical conversational range,
        which lets the model see whether a value is unusual without being told
        what it means.
        """
        return (
            f"pitch span {self.f0_range_semitones:.0f} semitones (typical conversational range 6-16); "
            f"loudness span {self.energy_range_db:.0f} dB (typical 15-35); "
            f"voiced-onset rate {self.speech_rate_proxy:.1f}/s (typical 2.0-4.5); "
            f"silence {self.pause_ratio:.0%} of the call (typical 20-50%); "
            f"pitch cycle-to-cycle variation {self.jitter:.3f} (typical 0.01-0.06); "
            f"amplitude variation {self.shimmer:.2f} (typical 0.05-0.30)."
        )


def _semitone_span(f0: np.ndarray) -> float:
    voiced = f0[f0 > 0]
    if voiced.size < 8:
        return 0.0
    lo = np.percentile(voiced, 5)
    hi = np.percentile(voiced, 95)
    if lo <= 0:
        return 0.0
    return float(12.0 * np.log2(max(hi, EPS) / max(lo, EPS)))


def _jitter_shimmer(f0: np.ndarray, energy_db: np.ndarray, voiced: np.ndarray) -> tuple[float, float]:
    seq = f0[voiced & (f0 > 0)]
    if seq.size < 10:
        jitter = 0.0
    else:
        jitter = float(np.mean(np.abs(np.diff(seq))) / (np.mean(seq) + EPS))

    amp = 10.0 ** (energy_db[voiced] / 20.0) if voiced.any() else np.array([])
    if amp.size < 10:
        shimmer = 0.0
    else:
        shimmer = float(np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + EPS))
    return jitter, shimmer


def extract_prosody(fb: FrameBank, vad: VadResult) -> Prosody:
    f0, _voicing = frame_pitch(fb)
    energy = fb.energy_db
    voiced = vad.voiced

    voiced_f0 = f0[voiced & (f0 > 0)]
    f0_median = float(np.median(voiced_f0)) if voiced_f0.size else 0.0
    f0_cv = float(np.std(voiced_f0) / (np.mean(voiced_f0) + EPS)) if voiced_f0.size else 0.0

    speech_energy = energy[voiced] if voiced.any() else energy
    if speech_energy.size:
        e_lo, e_hi = np.percentile(speech_energy, [5, 95])
        energy_range = float(e_hi - e_lo)
        energy_cv = float(np.std(speech_energy) / (abs(np.mean(speech_energy)) + EPS))
    else:
        energy_range, energy_cv = 0.0, 0.0

    total_sec = fb.n_frames * fb.frame_sec
    # Onsets of voiced runs approximate syllable groups without needing an ASR
    # pass, so this stays available in local_only mode.
    onsets = int(np.sum(np.diff(voiced.astype(np.int8)) == 1))
    rate = onsets / max(total_sec, EPS)

    pauses = [s.duration for s in vad.silence]
    speaking = sum(s.duration for s in vad.speech)
    conversational = speaking + sum(pauses)

    jitter, shimmer = _jitter_shimmer(f0, energy, voiced)

    return Prosody(
        f0_median_hz=f0_median,
        f0_range_semitones=_semitone_span(f0[voiced] if voiced.any() else f0),
        f0_variability=f0_cv,
        energy_range_db=energy_range,
        energy_variability=energy_cv,
        speech_rate_proxy=rate,
        pause_ratio=float(sum(pauses) / conversational) if conversational > 0 else 0.0,
        mean_pause_sec=float(np.mean(pauses)) if pauses else 0.0,
        longest_pause_sec=float(max(pauses)) if pauses else 0.0,
        speech_level_dbfs=fb.speech_level_db,
        voiced_fraction=float(voiced.mean()) if voiced.size else 0.0,
        jitter=jitter,
        shimmer=shimmer,
    )
