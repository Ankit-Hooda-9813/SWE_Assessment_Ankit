import warnings
warnings.filterwarnings("ignore")
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
from eval.tune_overlap_sortformer import derive_hv_truth, overlap_seconds_from_segments, REPO
import json, random, numpy as np, soundfile as sf

model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
model.eval()

transcripts = sorted((REPO / "transcript").glob("*.json"))
random.seed(11)
sids = [p.stem for p in random.sample(transcripts, 30)]

cfg = DiarizeConfig(max_num_of_spks=2, num_workers=0)

fracs, labels = [], []
for i, sid in enumerate(sids):
    transcript = json.loads((REPO / "transcript" / f"{sid}.json").read_text())
    agent, sr = sf.read(REPO / "audio" / "agent" / f"{sid}.wav", dtype="float32")
    caller, _ = sf.read(REPO / "audio" / "caller" / f"{sid}.wav", dtype="float32")
    n = max(len(agent), len(caller))
    agent = np.pad(agent, (0, n - len(agent)))
    caller = np.pad(caller, (0, n - len(caller)))
    mixed = np.clip(agent + caller, -1.0, 1.0)
    wav_path = f"/tmp/hv_2spk_{sid}.wav"
    sf.write(wav_path, mixed, sr)
    result = model.diarize(audio=[wav_path], batch_size=1, verbose=False, override_config=cfg)
    overlap_sec, n_spk = overlap_seconds_from_segments(result[0])
    fracs.append(overlap_sec)
    labels.append(derive_hv_truth(transcript))
    print(f"  [{i+1}/30] {sid}: n_spk_detected={n_spk} overlap_sec={overlap_sec:.2f} truth={labels[-1]}")

fracs = np.array(fracs)
labels = np.array(labels)
pred = fracs >= 0.35
acc = (pred == labels).mean()
tp = int((pred & labels).sum())
fp = int((pred & ~labels).sum())
fn = int((~pred & labels).sum())
tn = int((~pred & ~labels).sum())
print(f"\nmax_num_of_spks=2: accuracy={acc:.3f} tp={tp} fp={fp} fn={fn} tn={tn}")
