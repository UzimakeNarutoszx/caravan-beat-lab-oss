import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050; hop=256
y,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
ft = lambda t: t*sr/hop
dur=555.0
WIN=16.0; HOPw=16.0
tempos=[]; starts=[]; phases=[]
prev_end=None; prev_T=None
for s in np.arange(0, dur-8, HOPw):
    a=int(ft(s)); b=int(ft(min(s+WIN,dur)))
    e=env[a:b]
    tg = librosa.feature.tempo(onset_envelope=e, sr=sr, hop_length=hop, start_bpm=141, std_bpm=2.5, aggregate=None)
    T = float(np.median(tg))
    # grid phase search
    best=(-1,None)
    for off in np.arange(0, T, 0.004):
        g = np.arange(off, (b-a)*hop/sr, T)
        idx = np.clip(np.round(ft(g)).astype(int),0,len(e)-1)
        sc = e[idx].sum()/len(idx)
        if sc>best[0]: best=(sc, off)
    sc, off = best
    # continuity: expected offset such that the grid matches prev end
    if prev_end is not None:
        # find offset maximizing score within +-0.06 of the predicted continuation
        pred = (prev_end + T - s*0)  # relative
        cand = [(np.nan,None)]
        bestc=(-1,None)
        for o in np.arange(off-0.09, off+0.09, 0.004):
            g = np.arange(o, (b-a)*hop/sr, T)
            idx = np.clip(np.round(ft(g)).astype(int),0,len(e)-1)
            sco = e[idx].sum()/len(idx)
            if sco>bestc[0]: bestc=(sco,o)
        if bestc[1] is not None and bestc[0] > 0.75*sc:
            off = bestc[1]
    tempos.append(T); starts.append(s); phases.append(off)
    prev_T=T; prev_end = s + off + np.floor(((b-a)*hop/sr - off)/T)*T
    print("seg %5.1f T=%.4f (%.2f bpm) phase=%.3f score=%.3f"%(s,T,60/T,off,sc))
np.savez("caravan_work/segclick.npz", starts=starts, tempos=tempos, phases=phases)
