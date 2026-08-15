"""Runtime configuration.

Everything tunable lives here so the deployed Space can be reconfigured through
environment variables without a code change. Thresholds that the evaluation
harness fits are in `Thresholds`; they are written back by `eval/tune.py`.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrivacyMode(str, Enum):
    """Controls what, if anything, leaves this container."""

    LOCAL_ONLY = "local_only"  # nothing leaves; local ASR + local tone classifier
    HYBRID = "hybrid"          # transcript + numeric prosody leave; audio never does
    FULL = "full"              # audio itself is sent to a multimodal model


class Thresholds(BaseModel):
    """Decision boundaries for the measured fields.

    Defaults are physically motivated starting points. `eval/tune.py` fits them
    on the synthetic dev set and writes the fitted values to thresholds.json.
    """

    # --- voice activity ---
    vad_rel_db: float = Field(9.0, description="dB above the noise floor counted as speech")
    vad_abs_floor_dbfs: float = Field(-55.0, description="absolute level below which nothing is speech")

    # --- long silence ---
    # Measured on gaps strictly inside the conversation; leading/trailing dead air
    # is a recording artifact, not a call-flow problem.
    #
    # Calibrated against the only evidence available: all three provided clips are
    # labelled false, and the largest genuine internal gap among them is 7.31 s
    # (call_003, 113.6-120.9 s, real dead air at -65 dBFS). With no positive
    # example in the labelled set the boundary above 7.3 s is unconstrained, so
    # this sits just above the observed maximum. See the memo's limitations.
    long_silence_sec: float = 8.0
    silence_edge_exclude_sec: float = 0.5

    # --- background noise ---
    # A window counts as noisy only if the non-speech residual is BOTH loud enough
    # to hear in absolute terms and close enough to the speech level to intrude.
    noise_window_sec: float = 2.0
    noise_hop_sec: float = 1.0
    noise_audible_dbfs: float = -51.0   # absolute audibility gate on the window floor
    noise_snr_db: float = 42.0          # speech-to-floor range below this is intrusive
    noise_min_dwell_frac: float = 0.06  # fraction of clip that must be noisy
    noise_min_dwell_sec: float = 1.2    # ...and an absolute floor for short clips

    # severity is driven by how much of the clip is affected and how badly
    severity_low_frac: float = 0.05
    severity_medium_frac: float = 0.10
    severity_high_frac: float = 0.45
    severity_high_snr_db: float = 14.0

    # --- audio quality ---
    clip_ratio_slight: float = 0.0008
    clip_ratio_severe: float = 0.010
    # -35 dB band edge of the speech spectrum. Wideband calls land near 3.6 kHz,
    # narrowband telephony near 3.4 kHz, genuinely muffled audio well below.
    bandwidth_slight_hz: float = 2900.0
    bandwidth_severe_hz: float = 1900.0
    low_level_slight_dbfs: float = -34.0
    low_level_severe_dbfs: float = -46.0
    dropout_slight_frac: float = 0.010
    dropout_severe_frac: float = 0.050
    # Decay time from local speech level to -25 dB, in milliseconds.
    echo_slight_ms: float = 70.0
    echo_severe_ms: float = 95.0

    # --- speaker overlap ---
    # Fitted on the synthetic dev set; see app/audio/overlap.py for why this is
    # the weakest field in the system.
    #
    # v2: re-measured against the 150-clip ovlp_ subset specifically
    # (eval/tune_overlap.py), not the full 600-clip dev set where 559 clips
    # are trivially negative and dilute the signal. AUC on that proper subset
    # is 0.593, not the 0.66 previously documented — worth flagging as a
    # measurement correction, not just a re-tune. 0.25 is the F1-optimal
    # cutoff on that set (F1 0.464 vs 0.368 at the old 0.27). It does NOT
    # rescue call_003 (the one known-call miss): that clip's competing
    # fraction, 0.196, sits at the 10th percentile of the *positive* class —
    # among clips that genuinely do overlap, 90% show a stronger signal than
    # this one does. Lowering the threshold far enough to catch it collapses
    # precision (0.275 at 0.15, i.e. 3 in 4 "detected" overlaps would be
    # false). That miss is a real limit of this detector on a weak instance,
    # not a threshold miscalibration — the actual fix is the pyannote
    # backend already wired in app/audio/overlap.py, blocked only on a
    # one-time licence acceptance at huggingface.co/pyannote/segmentation-3.0.
    overlap_min_sec: float = 0.35        # used only by the pyannote backend
    overlap_frame_score: float = 0.85    # cepstral second-peak ratio per frame
    overlap_frame_fraction: float = 0.25 # share of pitched frames that must compete


class ProviderLimits(BaseModel):
    """Self-imposed ceilings, set below each provider's published free tier.

    We throttle ourselves rather than collect 429s. Values are deliberately
    conservative; raise them if you move to a paid tier.
    """

    requests_per_minute: int
    requests_per_day: int
    max_concurrent: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- behaviour ---
    privacy_mode: PrivacyMode = PrivacyMode.HYBRID
    tone_provider_order: str = "gemini,groq,local"
    asr_backend: str = "auto"  # auto | local | groq | none

    # Transcription uploads the audio file. `hybrid` promises the audio never
    # leaves, so it uses local Whisper unless this is explicitly turned on —
    # otherwise the mode's own description would be false, and the brief
    # requires disclosing exactly where customer audio goes.
    #
    # Turning it on is a reasonable choice with eyes open: Groq is ~38x faster
    # than local Whisper on two vCPU and states it does not train on API data.
    # It is off by default because a privacy claim should never be quietly
    # weaker than it reads.
    allow_asr_upload: bool = False

    # Speaker diarization, off by default despite being the textbook-correct
    # thing to do. The schema asks about the customer and roughly half of each
    # recording is the automated agent, so measuring across both is wrong in
    # principle — the contact-centre literature is explicit that an
    # undifferentiated sentiment score conflates the two.
    #
    # Measured, it still makes this system worse. The agent turns out to be the
    # higher-arousal speaker (a bright TTS voice at 0.715 against a frustrated
    # human at 0.651), and the arousal thresholds in app/ser/mapping.py were
    # fitted on whole-clip values that included it. Switching to customer-only
    # shifts every measurement down by roughly 0.05 and drops two of the three
    # labelled clips a whole intensity band. Re-fitting the thresholds on two
    # examples would be fitting noise.
    #
    # The separation itself is also not reliable enough: one labelled clip
    # produced a tied vote and no usable evidence, and the acoustic fallback is
    # demonstrably wrong — it reads short clipped utterances as synthetic
    # because they carry less pitch range than flowing sentences.
    #
    # Turn on with a real diarizer (pyannote/speaker-diarization) and enough
    # labelled data to recalibrate the thresholds against customer-only audio.
    diarization_enabled: bool = False

    # Second, independent SER opinion (emotion2vec+, categorical) alongside
    # the shipped wav2vec2-dim dimensional model. Local, so it costs nothing
    # and works in every privacy mode. Used narrowly in app/ser/mapping.py to
    # corroborate the frustrated->upset boundary specifically — see that
    # module's docstring for the measured case this fixes. Default on: it is
    # a local model like the one it complements, and degrades to "no signal"
    # rather than failing the clip if funasr is not installed.
    ser_ensemble_enabled: bool = True

    # Adaptive self-consistency for tone. Extra samples are drawn only for
    # answers that look shaky, so the typical clip still costs one call.
    self_consistency_enabled: bool = True
    self_consistency_samples: int = 3
    self_consistency_confidence: float = 0.80

    # --- credentials ---
    gemini_api_key: str = ""
    # Accepts GROK_API_KEY too. Groq (the inference provider, keys `gsk_`) and
    # Grok (xAI's model, keys `xai-`) are different products with confusingly
    # similar names, and the misspelling is easy to make.
    groq_api_key: str = Field("", validation_alias=AliasChoices("GROQ_API_KEY", "GROK_API_KEY"))
    cerebras_api_key: str = ""

    # --- models ---
    # Ordered by accuracy, tried in order. Measured, not assumed:
    # gemini-3.5-flash-lite scored 0/3 on tone and answered `neutral` to
    # everything, while flash scores 2/3 on the identical prompt — a five-class
    # emotional judgement is past what the lite tier does well.
    #
    # The chain exists because the free tier meters per model per day, and meters
    # the good model hard (20 requests/day for gemini-3.5-flash). Rotating models
    # multiplies the daily budget; later entries are less accurate but keep a
    # batch moving once the better quota is spent.
    gemini_model: str = "gemini-3.5-flash"
    gemini_model_chain: str = "gemini-3.5-flash,gemini-flash-latest,gemini-3.5-flash-lite"
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_asr_model: str = "whisper-large-v3-turbo"
    local_whisper_model: str = "small"
    local_whisper_compute: str = "int8"

    # --- dashboard ---
    dashboard_user: str = "autoace"
    dashboard_password: str = "change-me"
    session_secret: str = "dev-secret-change-in-production"

    # --- batch ---
    max_batch_files: int = 200
    max_file_mb: int = 60
    worker_concurrency: int = 2
    result_ttl_minutes: int = 120

    @property
    def gemini_models(self) -> list[str]:
        chain = [m.strip() for m in self.gemini_model_chain.split(",") if m.strip()]
        # Whatever `gemini_model` is set to leads, so overriding it by env still
        # works without having to restate the whole chain.
        return [self.gemini_model] + [m for m in chain if m != self.gemini_model]

    @property
    def gemini_limits(self) -> ProviderLimits:
        # Per model, per day. Measured against the live free tier, which returned
        # quotaValue=20 for gemini-3.5-flash — an order of magnitude below what
        # this was originally set to. Sitting just under it avoids spending
        # requests to discover the wall.
        return ProviderLimits(requests_per_minute=8, requests_per_day=18, max_concurrent=1)

    @property
    def groq_limits(self) -> ProviderLimits:
        return ProviderLimits(requests_per_minute=25, requests_per_day=900, max_concurrent=2)

    @property
    def tone_providers(self) -> list[str]:
        return [p.strip() for p in self.tone_provider_order.split(",") if p.strip()]

    def audio_may_leave(self) -> bool:
        """Whether the audio itself may be sent to the tone model."""
        return self.privacy_mode is PrivacyMode.FULL

    def asr_may_upload(self) -> bool:
        """Whether transcription may send the audio to a hosted service.

        `full` already permits it by definition. `hybrid` requires the explicit
        opt-in, because its whole promise is that the recording stays put.
        """
        if self.privacy_mode is PrivacyMode.LOCAL_ONLY:
            return False
        if self.privacy_mode is PrivacyMode.FULL:
            return True
        return self.allow_asr_upload

    def text_may_leave(self) -> bool:
        return self.privacy_mode in (PrivacyMode.HYBRID, PrivacyMode.FULL)


SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".webm", ".aac", ".wma", ".mp4",
}

TARGET_SR = 16_000

_settings: Settings | None = None
_thresholds: Thresholds | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_thresholds() -> Thresholds:
    """Load fitted thresholds if the eval harness has produced them."""
    global _thresholds
    if _thresholds is None:
        path = os.path.join(os.path.dirname(__file__), "thresholds.json")
        if os.path.exists(path):
            _thresholds = Thresholds.model_validate_json(open(path).read())
        else:
            _thresholds = Thresholds()
    return _thresholds


def set_thresholds(t: Thresholds) -> None:
    global _thresholds
    _thresholds = t
