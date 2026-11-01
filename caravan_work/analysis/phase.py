import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
y,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=140, tightness=100, trim=False)
tb = librosa.frames_to_time(beats, sr=sr, hop_length=256)
print("n beats", len(tb), "first", np.round(tb[:6],3), "last", np.round(tb[-3:],2))
# low band energy at beat times, averaged by phase class
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)); f = librosa.fft_frequencies(sr=sr, n_fft=2048)
low = S[(f>40)&(f<160)].sum(0); lowdb = librosa.amplitude_to_db(low, ref=np.max)
idx = np.round(tb*sr/256).astype(int); idx=np.clip(idx,0,len(lowdb)-1)
vals = lowdb[idx]
for ph in range(4):
    v = vals[ph::4]
    print("phase",ph,"n",len(v),"mean lowdb %.2f"%v.mean())
# intervals per 30s
for s in range(0,555,30):
    m=(tb>=s)&(tb<s+30)
    if m.sum()>5:
        d=np.diff(tb[m]); print("%3d bpm %.2f"%(s,60/np.median(d)), end=" | ")
print()
