import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
sr=22050
a, _ = librosa.load("caravan_work/caravan.wav", sr=sr, mono=True)
b, _ = librosa.load("./local-data", sr=sr, mono=True)
print(len(a)/sr, len(b)/sr)
# chroma for both (CQT) -> compare 30s windows of b to a via sliding chroma correlation
ca = librosa.feature.chroma_cqt(y=a, sr=sr, hop_length=2048)
cb = librosa.feature.chroma_cqt(y=b, sr=sr, hop_length=2048)
def best(cb_slice):
    sims = []
    for i in range(ca.shape[1]-cb_slice.shape[1]+1):
        A = ca[:, i:i+cb_slice.shape[1]]
        d = np.mean(np.sum(A*cb_slice,0)/(np.linalg.norm(A,axis=0)*np.linalg.norm(cb_slice,axis=0)+1e-9))
        sims.append(d)
    sims=np.array(sims); j=int(np.argmax(sims))
    return j*2048/sr, sims[j]
for start in [0, 60, 120, 240, 300, 315]:
    s = int(start*sr/2048); e = s+int(20*sr/2048)
    t, sc = best(cb[:, s:e])
    print("mp3 t=%3d best m4a t=%6.1f sim=%.3f"%(start, t, sc))
