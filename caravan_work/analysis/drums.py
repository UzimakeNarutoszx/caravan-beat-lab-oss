import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
y, _ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
print("dur", len(y)/sr)
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
low = S[(freqs>30)&(freqs<150)].sum(0)
mid = S[(freqs>150)&(freqs<800)].sum(0)
high = S[(freqs>4000)].sum(0)
t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=256)
for name, e in [("low",low),("mid",mid),("high",high)]:
    ed = librosa.amplitude_to_db(e, ref=np.max)
    pk = librosa.util.peak_pick(ed, pre_max=5, post_max=5, pre_avg=15, post_avg=15, delta=8, wait=5)
    ts = t[pk]
    print(name, "n=",len(ts), "first", np.round(ts[:30],3))
    if len(ts)>5:
        d=np.diff(ts); u,c=np.unique(np.round(d/0.005)*0.005, return_counts=True)
        i=np.argsort(c)[-10:][::-1]; print("   common d", [(round(float(u[j]),3),int(c[j])) for j in i])
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
for bpm0 in (70,140,280):
    tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=bpm0, tightness=200, trim=False)
    tb = librosa.frames_to_time(beats, sr=sr, hop_length=256)
    d = np.diff(tb)
    print("bpm0",bpm0,"tempo",tempo,"n",len(tb),"first",np.round(tb[:6],2),"median ioi %.4f std %.4f"%(np.median(d), d.std()))
# windowed tempo 20s
for s in range(0,540,60):
    seg = y[s*sr:(s+30)*sr]
    o = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=256)
    tt = librosa.feature.tempo(onset_envelope=o, sr=sr, hop_length=256, start_bpm=140, std_bpm=4)
    tt2 = librosa.feature.tempo(onset_envelope=o, sr=sr, hop_length=256, start_bpm=280, std_bpm=8)
    print("t=%3d tempo140 %.2f tempo280 %.2f"%(s, tt, tt2))
