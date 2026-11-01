import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050; hop=256; fps=sr/hop
y,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=sr, mono=True)
env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
env = env/(env.max()+1e-9)
dur=555.0; WIN=16.0; STEP=8.0
clicks=[]; prev_T=60/141.2; prev_phase=None
seg_starts=np.arange(0, dur-WIN+1, STEP)
for s in seg_starts:
    a=int(s*fps); b=int((s+WIN)*fps); e=env[a:b]
    T=prev_T; ph=0.0; sc=0.0
    if e.max()>=0.05:
        bpm0=60/prev_T
        best=(-1,prev_T,0.0)
        for bpm in np.arange(max(137.5,bpm0-2.5), min(145.5,bpm0+2.5), 0.05):
            Tt=60/bpm; n=int(WIN/Tt); base=np.arange(n)*Tt
            phs=np.arange(0,Tt,0.005)
            idx=np.round((base[None,:]+phs[:,None])*fps).astype(int)
            idx=np.clip(idx,0,len(e)-1)
            scv=e[idx].sum(1)/n; j=int(np.argmax(scv))
            if scv[j]>best[0]: best=(scv[j],Tt,phs[j])
        sc,T,ph=best
        if prev_phase is not None:
            pred=(prev_phase-s)%T
            diff=((ph-pred+T/2)%T)-T/2
            if abs(diff)>0.05:
                cand=pred+np.arange(-0.045,0.045,0.005)
                idx=np.round((np.arange(int(WIN/T))[None,:]*T+cand[:,None])*fps).astype(int)
                idx=np.clip(idx,0,len(e)-1); sc2=e[idx].sum(1)/idx.shape[1]
                j=int(np.argmax(sc2))
                if sc2[j]>0.85*sc: ph=cand[j]%T; sc=sc2[j]
    t=s+ph+np.arange(int((WIN-0.001)/T))*T
    lo=s+(0 if s==0 else (WIN-STEP)/2); hi=s+(WIN if s==seg_starts[-1] else (WIN+STEP)/2)
    sel=(t>=lo)&(t<hi); clicks.extend(t[sel].tolist())
    prev_T=T; prev_phase=s+ph+np.floor((WIN-ph)/T)*T
    print("seg %5.1f bpm=%.2f ph=%.3f sc=%.2f n=%d"%(s,60/T,ph,sc,sel.sum()))
clicks=np.array(sorted(clicks))
print("total clicks",len(clicks),"first",clicks[:4].round(3),"last",clicks[-2:].round(2))
iv=np.diff(clicks)
print("interval mean %.4f std %.4f min %.3f max %.3f"%(iv.mean(),iv.std(),iv.min(),iv.max()))
on=librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units='time', delta=0.02, wait=1)
idx=np.searchsorted(on,clicks); idx=np.clip(idx,1,len(on)-1)
dist=np.minimum(np.abs(clicks-on[idx-1]),np.abs(clicks-on[idx]))
print("click->onset median %.3f frac<30ms %.2f <60ms %.2f"%(np.median(dist),np.mean(dist<0.03),np.mean(dist<0.06)))
np.save("caravan_work/click_times.npy",clicks)
