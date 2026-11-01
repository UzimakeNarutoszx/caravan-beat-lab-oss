import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
y,_ = librosa.load("caravan_work/caravan.wav", sr=sr, mono=True)
hop=512
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
print("  t   rms_db  tempo(w30s)")
for s in range(0, 555, 15):
    m = (t>=s)&(t<s+15)
    o = oe[(t>=s)&(t<s+30)]
    tg = librosa.feature.tempo(onset_envelope=o, sr=sr, hop_length=hop, start_bpm=140, std_bpm=8, aggregate=None)
    print("%4d  %6.1f  %s"%(s, 20*np.log10(max(rms[m].mean(),1e-9)), np.round(np.median(tg),1)))
