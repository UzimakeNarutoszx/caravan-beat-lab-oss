import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
y, sr = librosa.load("caravan_work/caravan.wav", sr=22050, mono=True, offset=20, duration=120)
oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256, aggregate=np.median)
on = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, hop_length=256, units='time', delta=0.05, wait=3)
print("n onsets", len(on))
ioi = np.diff(on)
print("ioi hist", np.round(np.histogram(ioi, bins=np.arange(0.05,1.2,0.02))[0],0))
print("top ioi", [(round(float(v),3), int(c)) for v,c in zip(*np.unique(np.round(ioi,2), return_counts=True))][:0])
u,c = np.unique(np.round(ioi/0.01)*0.01, return_counts=True)
idx=np.argsort(c)[-15:][::-1]
print("common ioi", [(round(float(u[i]),3), int(c[i])) for i in idx])
# phase clustering for candidate periods
for period in (0.8571, 0.4286, 0.2857):
    ph = (on % period)/period
    h,_ = np.histogram(ph, bins=12, range=(0,1))
    print("period",round(period,4),"phase hist",h)
