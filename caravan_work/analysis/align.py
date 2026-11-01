import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
y,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
on = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, hop_length=256, units='time', delta=0.02, wait=1)
print("onsets", len(on), "density", len(on)/555)
for bpm0 in (140, 280):
    tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=bpm0, tightness=100, trim=False)
    tb = librosa.frames_to_time(beats, sr=sr, hop_length=256)
    # nearest onset distance
    idx = np.searchsorted(on, tb)
    idx = np.clip(idx, 1, len(on)-1)
    dist = np.minimum(np.abs(tb-on[idx-1]), np.abs(tb-on[idx]))
    print("bpm0",bpm0,"n",len(tb),"median dist %.3f"%np.median(dist),
          "frac<30ms %.2f frac<60ms %.2f"%(np.mean(dist<0.03), np.mean(dist<0.06)))
    # local tempo per 60s window (median ioi)
    for s in range(0,540,60):
        m=(tb>=s)&(tb<s+60)
        if m.sum()>4:
            d=np.diff(tb[m]); print("   %3d-%3d beats %3d bpm %.1f"%(s,s+60,m.sum(),60/np.median(d)))
