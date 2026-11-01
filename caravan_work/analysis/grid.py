import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
y,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
for bpm0 in (140,143,143.5):
    tempo, beats = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=bpm0, tightness=300, trim=False)
    tb = librosa.frames_to_time(beats, sr=sr, hop_length=256)
    n = np.arange(len(tb))
    # linear fit
    A = np.vstack([n, np.ones_like(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, tb, rcond=None)
    pred = A@coef
    r = tb-pred
    print("bpm0",bpm0,"tempo",np.round(tempo,2),"n",len(tb),"fitted beat %.5f s -> %.3f bpm"%(coef[0],60/coef[0]),
          "resid std %.4f max %.3f"%(r.std(), np.abs(r).max()))
    for q in (0,0.25,0.5,0.75,1.0):
        i=int(q*(len(tb)-1)); print("   t=%6.1f resid %+.3f"%(tb[i], r[i]))
    # onset strength at beats
    env = oe
    idx = np.clip(np.round(tb*sr/256).astype(int),0,len(env)-1)
    print("   mean onset at beats %.2f vs overall %.2f, frac>0.5max %.2f"%(env[idx].mean(), env.mean(), (env[idx]>0.5*env.max()).mean()))
