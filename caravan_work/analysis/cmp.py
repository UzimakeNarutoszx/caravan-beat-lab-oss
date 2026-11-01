import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
a, sr = librosa.load("caravan_work/caravan.wav", sr=22050, mono=True, duration=45)
b, sr2 = librosa.load("./local-data", sr=22050, mono=True, duration=45)
for name, y in [("m4a",a),("mp3",b)]:
    oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    print(name, "dur", len(y)/sr, "tempo70", librosa.feature.tempo(onset_envelope=oe, sr=sr, start_bpm=70, std_bpm=2),
          "tempo140", librosa.feature.tempo(onset_envelope=oe, sr=sr, start_bpm=140, std_bpm=2))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(name, "chroma mean", chroma.mean(1).round(2))
# cross-similarity of chroma
ca = librosa.feature.chroma_cqt(y=a, sr=sr)
cb = librosa.feature.chroma_cqt(y=b, sr=sr)
D, wp = librosa.sequence.dtw(X=ca, Y=cb, metric='cosine', subseq=True)
print("dtw cost", D[-1,-1], "path len", wp.shape)
# top energy frames
print("rms a", np.round(librosa.feature.rms(y=a, hop_length=512)[0].max(),3), "b", np.round(librosa.feature.rms(y=b, hop_length=512)[0].max(),3))
