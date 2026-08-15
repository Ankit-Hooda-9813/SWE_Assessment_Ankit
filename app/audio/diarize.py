"""Separating the customer from the agent.

The schema asks for the emotional tone expressed by *the customer*. These
recordings are single-channel and dual mono, so both parties are mixed together,
and roughly half of every provided clip is the automated agent talking. Measuring
emotion across the whole file therefore averages a synthetic voice into a human
one — the contact-centre literature is blunt about this, describing an
undifferentiated sentiment score as conflating customer frustration with agent
responses to produce a meaningless average.

It is not a theoretical concern here. Measured on the labelled clips, removing
the agent moved every one of them toward its reference label, and fixed an
inversion: with the agent included, `call_002` measured *more* positive than
`call_003`, while the labels say the opposite (neutral versus satisfied).
Customer-only measurement puts them back in the right order.

Two speakers are separated by clustering speech segments on voice
characteristics. Which cluster is the agent is then decided by how synthetic it
sounds: text-to-speech holds pitch far more steadily than a person does, so the
cluster with the lower pitch variability is the agent. A transcript cue is used
in preference when one is available, because an agent that introduces itself is
better evidence than a statistic.

The limitation is worth stating plainly: this assumes exactly two speakers, one
of whom is synthetic. On a human-to-human call the stability heuristic has no
basis, and a real diarizer — `pyannote/speaker-diarization` — should be used
instead. This is the free, dependency-light approximation of one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from app.audio.features import EPS, FrameBank, frame_pitch
from app.audio.vad import Segment, VadResult

MIN_SEGMENT_SEC = 0.5

# An agent identifying itself. Strong evidence, and cheap to check.
AGENT_PHRASES = re.compile(
    r"\b(i'?m \w+ (from|with|at)|how can i help|how may i (help|assist)|"
    r"thank you for calling|this is \w+ (from|with)|i can help with that|"
    r"what type of service)\b",
    re.IGNORECASE,
)


@dataclass
class Diarization:
    customer_spans: list[tuple[float, float]] = field(default_factory=list)
    agent_spans: list[tuple[float, float]] = field(default_factory=list)
    customer_sec: float = 0.0
    agent_sec: float = 0.0
    method: str = "none"
    confident: bool = False
    evidence: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the split is worth trusting.

        Requires a real amount of customer speech: a degenerate clustering that
        assigns almost everything to one side is worse than not splitting at
        all, because it would measure emotion from a couple of seconds.
        """
        return bool(self.customer_spans) and self.customer_sec >= 3.0


def _segment_features(fb: FrameBank, segments: list[Segment]) -> tuple[np.ndarray, list[Segment]]:
    """Describe each speech segment by voice, not by content."""
    f0, voicing = frame_pitch(fb)
    rows: list[list[float]] = []
    kept: list[Segment] = []

    for seg in segments:
        if seg.duration < MIN_SEGMENT_SEC:
            continue
        lo = int(seg.start / fb.frame_sec)
        hi = min(int(seg.end / fb.frame_sec), fb.n_frames)
        window = f0[lo:hi]
        voiced = window[(voicing[lo:hi] > 0.45) & (window > 0)]
        if voiced.size < 8:
            continue

        spectrum = fb.power[lo:hi].mean(axis=0)
        spectrum = spectrum / (spectrum.sum() + EPS)
        centroid = float((spectrum * fb.freqs).sum())
        # Rough timbre signature: how energy splits across four broad bands.
        bands = [
            float(spectrum[(fb.freqs >= lo_hz) & (fb.freqs < hi_hz)].sum())
            for lo_hz, hi_hz in ((0, 500), (500, 1500), (1500, 3000), (3000, 8000))
        ]

        rows.append([
            float(np.median(voiced)),
            float(np.std(voiced) / (np.mean(voiced) + EPS)),  # pitch stability
            centroid,
            *bands,
        ])
        kept.append(seg)

    return np.array(rows, dtype=np.float64), kept


def diarize(
    fb: FrameBank, vad: VadResult, transcript_segments: list[dict] | None = None
) -> Diarization:
    """Split speech into customer and agent."""
    features, segments = _segment_features(fb, vad.speech)
    if features.shape[0] < 4:
        return Diarization(method="too-few-segments")

    from sklearn.cluster import KMeans

    normalised = (features - features.mean(axis=0)) / (features.std(axis=0) + EPS)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(normalised)

    if len(set(labels.tolist())) < 2:
        return Diarization(method="single-cluster")

    # Pitch variability, column 1. Synthetic speech is markedly steadier.
    variability = [float(features[labels == k, 1].mean()) for k in (0, 1)]
    agent_cluster = int(np.argmin(variability))
    method = "pitch-stability"

    # A self-introduction beats the statistic when we have the words for it.
    if transcript_segments:
        votes = {0: 0, 1: 0}
        for entry in transcript_segments:
            if not AGENT_PHRASES.search(entry.get("text", "")):
                continue
            mid = (float(entry.get("start", 0)) + float(entry.get("end", 0))) / 2
            for idx, seg in enumerate(segments):
                if seg.start <= mid <= seg.end:
                    votes[int(labels[idx])] += 1
                    break
        if votes[0] != votes[1]:
            agent_cluster = 0 if votes[0] > votes[1] else 1
            method = "transcript-cue"

    customer_cluster = 1 - agent_cluster
    customer = [(s.start, s.end) for s, k in zip(segments, labels) if k == customer_cluster]
    agent = [(s.start, s.end) for s, k in zip(segments, labels) if k == agent_cluster]

    customer_sec = sum(e - s for s, e in customer)
    agent_sec = sum(e - s for s, e in agent)
    separation = abs(variability[0] - variability[1]) / (max(variability) + EPS)

    return Diarization(
        customer_spans=customer,
        agent_spans=agent,
        customer_sec=customer_sec,
        agent_sec=agent_sec,
        method=method,
        # A weak split is reported rather than trusted; callers fall back to the
        # whole clip, which is the behaviour that was correct before this existed.
        confident=separation >= 0.15 and customer_sec >= 3.0,
        evidence={
            "segments": len(segments),
            "customer_sec": round(customer_sec, 1),
            "agent_sec": round(agent_sec, 1),
            "pitch_variability": [round(v, 3) for v in variability],
            "separation": round(separation, 3),
            "method": method,
        },
    )


def customer_transcript(
    transcript_segments: list[dict] | None, diarization: Diarization
) -> str:
    """The customer's words only, for the tone prompt.

    Returned empty when the split is not trustworthy, so the caller keeps the
    full transcript rather than acting on a bad separation.
    """
    if not transcript_segments or not diarization.ok:
        return ""
    parts = []
    for entry in transcript_segments:
        mid = (float(entry.get("start", 0)) + float(entry.get("end", 0))) / 2
        if any(start <= mid <= end for start, end in diarization.customer_spans):
            text = (entry.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)
