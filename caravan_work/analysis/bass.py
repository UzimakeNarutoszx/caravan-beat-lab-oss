import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
for seg_start in (30, 60, 300, 420):
    y, _ = librosa.load("caravan_work/sep/htdemucs/bass.wav", sr=sr, mono=True, offset=seg_start, duration=20)
    oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256, aggregate=np.median)
    on = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, hop_length=256, units='time', delta=0.03, wait=2)
    d = np.diff(on)
    print("bass seg",seg_start,"n",len(on),"onsets",np.round(on[:20],2))
    if len(d):
        u,c = np.unique(np.round(d/0.01)*0.01, return_counts=True)
        i=np.argsort(c)[-8:][::-1]
        print("  d", [(round(float(u[j]),3),int(c[j])) for j in i])
    for bpm0 in (70,140,280):
        tt = librosa.feature.tempo(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=bpm0, std_bpm=3)
        print("  tempo",bpm0,np.round(np.atleast_1d(tt),2))
# 20-60s spectrogram/pulse clarity via tempogram
y, _ = librosa.load("caravan_work/sep/htdemucs/bass.wav", sr=sr, mono=True, offset=30, duration=40)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
tg = librosa.feature.tempogram(onset_envelope=oe, sr=sr, hop_length=256, win_length=256)
bt = np.median(tg, axis=1)
freqs = librosa.tempo_frequencies(len(bt), sr=sr, hop_length=256)
sel = (freqs>40)&(freqs<400)
idx = np.argsort(bt[sel])[-10:][::-1]
print("tempogram peaks", [(round(float(freqs[sel][j]),2), round(float(bt[sel][j]),3)) for j in idx])
