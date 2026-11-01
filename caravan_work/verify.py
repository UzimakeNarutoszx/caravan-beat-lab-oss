import numpy as np, soundfile as sf, librosa, warnings
warnings.filterwarnings("ignore")
# 1) count-in clicks
c, sr = sf.read("caravan_输出/Caravan_节拍器_预备拍.wav", dtype="float32")
c = c[:,0] if c.ndim>1 else c
env = np.abs(c)
peaks = []
i=0
while i < len(env):
    if env[i] > 0.1:
        j = i
        while j < len(env) and env[j] > 0.02: j += 1
        peaks.append((i/sr, env[i:j].max()))
        i = j + int(0.05*sr)
    else:
        i += 1
print("n click events", len(peaks))
print("first 10:", [(round(t,3), round(float(a),2)) for t,a in peaks[:10]])
if len(peaks) > 10:
    d = np.diff([t for t,_ in peaks[:10]])
    print("first intervals", np.round(d,3))
# 2) drumless residual: onset strength ratio drums vs mix vs drumless
dm,_ = librosa.load("caravan_work/sep/htdemucs/drums.wav", sr=22050, mono=True)
nm,_ = librosa.load("caravan_输出/Caravan_无鼓纯伴奏.wav", sr=22050, mono=True, duration=555, offset=3.6)
def flux(y):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))**2
    D = np.diff(S, axis=1)
    D[D<0]=0
    return np.sqrt(D).sum(0)
fd, fn = flux(dm), flux(nm)
print("onset flux drums mean %.0f, drumless mean %.0f, ratio %.2f"%(fd.mean(), fn.mean(), fn.mean()/fd.mean()))
# band energy 2-8k (cymbals/snare) ratio
def band(y, lo, hi):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)); f=librosa.fft_frequencies(sr=22050, n_fft=2048)
    return (S[(f>lo)&(f<hi)]**2).sum()
print("drumless/drums high-band energy ratio %.2f"%(band(nm,2000,8000)/band(dm,2000,8000)))
