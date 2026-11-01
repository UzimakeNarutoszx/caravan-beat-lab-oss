import numpy as np, soundfile as sf, librosa, warnings, json
from scipy.interpolate import UnivariateSpline
warnings.filterwarnings("ignore")
sr = 44100
SEP = "caravan_work/sep/htdemucs"
OUT = "caravan_输出"

# ---------- beat grid ----------
y22, _ = librosa.load(f"{SEP}/drums.wav", sr=22050, mono=True)
oe = librosa.onset.onset_strength(y=y22, sr=22050, hop_length=256)
tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=22050, hop_length=256,
                                        start_bpm=140, tightness=100, trim=False)
tb = librosa.frames_to_time(beats, sr=22050, hop_length=256)
k = np.arange(len(tb), dtype=float)
# smooth beat-time curve (allows slow tempo drift, removes frame jitter)
spl = UnivariateSpline(k, tb, k=3, s=len(tb)*2e-4)
ts = np.clip(spl(k), tb[0]-0.02, tb[-1]+0.02)
dev = np.abs(ts - tb)
print("beats", len(tb), "first %.3f last %.3f"%(tb[0], tb[-1]))
print("smooth dev: median %.3fs p95 %.3fs max %.3fs"%(np.median(dev), np.percentile(dev,95), dev.max()))
B = 60.0/140.0   # nominal beat for count-in
t0 = tb[0]       # first beat of the song

# ---------- click track ----------
music_dur = 555.0
pre = 0.30
count_len = 8*B
shift = pre + count_len - t0      # song sample 0 is placed here
total = int(np.ceil((pre + count_len + music_dur + 1.5)*sr))
click = np.zeros(total, dtype=np.float32)

def add_click(t, freq, amp, decay=0.012, dur=0.08):
    n = int(dur*sr); i = int(round(t*sr))
    if i < 0 or i+n > click.size: return
    tt = np.arange(n)/sr
    w = np.sin(2*np.pi*freq*tt)*np.exp(-tt/decay)
    w += 0.35*np.sin(2*np.pi*freq*2.02*tt)*np.exp(-tt/(decay*0.6))
    click[i:i+n] += (amp*w).astype(np.float32)

# count-in: 1-2 at 70 (beats 8 and 6 before downbeat), then 1-2-3-4 at 140
c0 = pre
add_click(c0 + 0*2*B, 2000, 0.55)          # "1" (70bpm)
add_click(c0 + 1*2*B, 1550, 0.36)          # "2" (70bpm)
for j in range(4):                          # "1-2-3-4" at 140
    t = c0 + 4*B + j*B
    add_click(t, 2000 if j==0 else 1550, 0.55 if j==0 else 0.36)
# song clicks
for i, t in enumerate(ts):
    tt = t + shift
    acc = (i % 4 == 0)
    add_click(tt, 2000 if acc else 1550, 0.50 if acc else 0.33)
print("click track built; peak %.3f"%np.abs(click).max())

# ---------- drumless mix ----------
mix = np.zeros((total, 2), dtype=np.float32)
for stem in ("bass", "other"):
    x, sr2 = sf.read(f"{SEP}/{stem}.wav", dtype="float32", always_2d=True)
    assert sr2 == sr
    n = min(len(x), total - int(shift*sr))
    a = int(round(shift*sr))
    mix[a:a+n] += x[:n]
print("music peak %.3f"%np.abs(mix).max())
mix *= 0.72/np.abs(mix).max()             # headroom for clicks
mix += click[:, None]
peak = np.abs(mix).max()
print("pre-limit peak %.3f"%peak)
if peak > 0.97:
    mix *= 0.97/peak
# short tail fade
fade = int(1.2*sr)
mix[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
sf.write(f"{OUT}/Caravan_无鼓_预备拍_节拍器.wav", mix, sr, subtype="PCM_16")
# music only (no clicks), no count-in
mo = np.zeros((total, 2), dtype=np.float32)
for stem in ("bass", "other"):
    x, _ = sf.read(f"{SEP}/{stem}.wav", dtype="float32", always_2d=True)
    a = int(round(shift*sr)); n = min(len(x), total-a)
    mo[a:a+n] += x[:n]
mo *= 0.89/np.abs(mo).max()
mo[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
sf.write(f"{OUT}/Caravan_无鼓纯伴奏.wav", mo, sr, subtype="PCM_16")
sf.write(f"{OUT}/Caravan_节拍器_预备拍.wav", click*0.8, sr, subtype="PCM_16")
np.savez("caravan_work/beats.npz", tb=tb, ts=ts, shift=shift, pre=pre, B=B)
print("durations", total/sr)
