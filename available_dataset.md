Yes. I did a much deeper search, this time filtering for actual downloadable audio + machine-readable labels + relevance to your AutoAce schema. I also checked the requirements again: your system is scored separately on emotional tone, background noise, audio quality, overlap and silence, with macro-F1/confusion-matrix validation expected.

The result is much better than my first search.

My strongest finding: HaessigDB
1. HaessigDB — best match for your emotional-tone requirement

This is the most interesting dataset I found for your particular problem.

It contains 3,755 labeled audio snippets, with WAV files actually downloadable from Hugging Face. The dataset is specifically acted call-center-style English speech, with four professional actors performing synthetic banking-support conversations where the customer becomes increasingly irritated. Each clip has human ratings for:

aggression: 1–10
frustration: 1–10
annoyance: 1–10
actor
call ID
sentence position
transcript

The full audio is downloadable, and the Hugging Face dataset has an Apache-2.0 license.

HaessigDB on Hugging Face — downloadable audio + labels

This is much closer to AutoAce than RAVDESS/CREMA-D, because it has:

call-center-style speech
+
customer getting irritated
+
frustration
+
aggression
+
annoyance
+
intensity
+
call progression

You can derive a controlled mapping into your schema:

HaessigDB
    frustration/aggression/annoyance
              ↓
      emotional_tone
              +
      emotional_intensity

I would not pretend that its 1–10 ratings are identical to your low | medium | high. Instead, define a transparent thresholding rule, for example:

1–3  -> low
4–6  -> medium
7–10 -> high

and document that this is a derived mapping, not the original annotation.

Even better, because you have three separate dimensions, you can distinguish:

frustration high + aggression low
        ≠
frustration high + aggression high

That is useful for your frustrated vs upset/distressed distinction.

One limitation

It is acted call-center-style speech, not actual production customer calls. So I'd use it as your emotion benchmark, not claim it represents real production distribution.

2. CREMA-D — best large clean emotion + intensity benchmark

CREMA-D is another very strong downloadable source.

The Hugging Face mirror has:

7,442 WAV clips
91 actors
6 emotions
4 intensity levels

The dataset exposes emotion_code, emotion_intensity, intensity_code, actor ID and audio directly.

CREMA-D on Hugging Face

Its emotion classes are:

anger
disgust
fear
happy
neutral
sad

and intensity includes:

low
medium
high
unspecified

This makes it useful for checking whether your model can distinguish:

neutral
vs
positive-ish
vs
anger
vs
high-intensity negative emotion

Again, don't directly claim:

anger == upset
happy == satisfied

Those are semantic mappings you construct for your benchmark.

CREMA-D's official dataset also provides the underlying annotation files, and the dataset license is Open Database License / Database Contents License.

3. IEMOCAP — best richer emotion benchmark

I found a usable downloadable Hugging Face mirror with 10k audio examples and much richer annotation fields.

It exposes:

frustrated
angry
sad
disgust
excited
fear
neutral
surprise
happy


EmoAct
EmoVal
EmoDom

along with audio, transcription, gender and acoustic features.

IEMOCAP Hugging Face dataset

This is especially valuable for your emotional_intensity idea because EmoAct is an activation/arousal-style dimension rather than just a categorical emotion.

There is also an IEMOCAP dataset mirror specifically exposing utterance labels and VAD-related information.

Why I like IEMOCAP for you

Your production task explicitly says:

don't infer frustration/distress simply from loudness.

IEMOCAP gives you richer emotional annotation than a simple one-hot emotion class, so it is useful when analyzing whether your model is learning actual expressive characteristics rather than just loudness.

4. NISQA — best dataset I found for audio quality

This one is extremely relevant to your:

audio_quality

field.

The NISQA Corpus contains 14,000+ speech samples with simulated and real conditions including:

codecs
packet loss
background noise
mobile phone
Zoom
Skype
WhatsApp
real phone/VoIP calls

Each sample has human ratings for:

overall quality
noisiness
coloration
discontinuity
loudness

and more than 97,000 human ratings overall.

NISQA Corpus on Zenodo

The official archive is around 15.9 GB, so this is a serious dataset, not a toy dataset.

Even better for your case, NISQA has:

NISQA_TEST_LIVETALK

which contains real phone and VoIP calls.

This is a very good source for validating:

audio_quality

and to some extent:

background_noise_present
background_noise_severity

because it includes distorted/noisy transmissions.

Important license warning

NISQA is not one simple universally commercial license. The corpus inherits licenses from its source audio/noise datasets, and some parts are restricted to non-commercial research.

For your interview/technical trial, that's different from deploying the dataset commercially, but I'd document the exact subset and license you use.

5. AMI / AMI-derived overlap dataset — excellent for speaker overlap

I found a particularly convenient derived dataset called AMI 2-Speaker Test Set.

It provides:

audio
speaker1_start
speaker1_end
speaker2_start
speaker2_end
overlap_ratio

and 50 real conversational clips from AMI. Overlap ratios range from 0 to roughly 0.9.

AMI 2-Speaker Test Set on Hugging Face

This is almost exactly what you need for:

speaker_overlap_present

You can define:

overlap_ratio > 0
    → true


overlap_ratio == 0
    → false

or use a threshold if you want "enough overlap to affect understanding."

The underlying AMI corpus itself contains about 100 hours of meeting recordings and public audio/annotations under CC BY 4.0.

There are also downloadable Hugging Face mirrors containing the AMI audio and annotations.

6. NOIZEUS — very useful for controlled noise severity

This is not a general emotion dataset, but it is excellent for your noise experiment because the noise conditions are explicitly controlled.

It contains environments such as:

babble
car
exhibition hall
restaurant
street
airport
train station
train

and noisy speech at:

0 dB
5 dB
10 dB
15 dB

SNR. The files are downloadable WAV files.

NOIZEUS downloadable noisy-speech corpus

This is almost perfect for validating:

background_noise_present
background_noise_type
background_noise_severity

because you've actually got known noise classes and known SNR.

For example:

restaurant + 15 dB
restaurant + 10 dB
restaurant + 5 dB
restaurant + 0 dB

gives you a controlled severity gradient.

7. MUSAN — useful noise library for building your own controlled benchmark

MUSAN has speech, music and noise recordings and is openly downloadable. The corpus is around 109 hours, with roughly:

60h speech
42h music
6h noise

in one Hugging Face mirror.

MUSAN on Hugging Face

This isn't as useful as NOIZEUS for ground-truth severity, but it's very useful for generating controlled cases:

clean emotion clip
        +
MUSAN noise/music
        +
chosen SNR
        ↓
known synthetic ground truth

For example:

CREMA-D angry
       +
MUSAN noise @ 10 dB
       ↓
emotion = angry
noise_present = true
noise_severity = medium

That lets you systematically test your pipeline.

8. RAVDESS — useful, but below CREMA-D/HaessigDB for this project

RAVDESS is definitely downloadable and labeled.

It has:

1,440 speech recordings

with:

emotion
intensity
actor
gender
statement
repetition

and two emotion intensity levels.

RAVDESS Hugging Face mirror

I would use it as a secondary emotion sanity-check dataset, not your main validation dataset.

Why?

Because your target is customer calls, while RAVDESS is professionally acted emotional speech.

One dataset I specifically do NOT recommend despite looking perfect
ES-Port

At first glance, ES-Port looked like the holy grail:

actual telecom technical-support calls
1,170 dialogues
~40 hours
spontaneous speech
background noise annotations
silence annotations
overlapping speech
background conversation
music
laughter
multiple acoustic events

Its annotation schema is incredibly relevant to AutoAce.

For example, it explicitly annotates:

electric noise
background laughter
background cough
background conversation
music
silence

and the corpus also contains overlapping speech.

But there is a critical problem: the raw audio cannot be publicly released.

The researchers explicitly state that the raw audio contains sensitive data and cannot be distributed under GDPR. Only anonymized transcriptions/annotations were released.

So:

great conceptual match, but fails your "downloadable audio + labels" requirement.

I would not use it for your actual benchmark.

Another interesting one: real customer-call dataset on Kaggle

I also found a downloadable Customer Call Center Dataset + Analysis containing:

simulated customer-support calls
.wav audio
transcripts
CSVs
5-emotion sentiment labels
call types

and it is listed under CC BY 4.0.

Customer Call Center Dataset + Analysis on Kaggle

This is worth downloading and inspecting.

However, I would rank it below HaessigDB because the published description says the calls are simulated, and its documentation is much less rigorous than the established research corpora