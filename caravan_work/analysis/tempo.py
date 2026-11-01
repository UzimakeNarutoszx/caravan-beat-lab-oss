import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
y, sr = librosa.load("caravan_work/caravan.wav", sr=22050, mono=True)
print("dur", len(y)/sr)
# global tempo estimate
onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
for start_bpm in (70, 140, 280):
    t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=start_bpm, std_bpm=1.0)
    print("tempo est start", start_bpm, t)
# tempogram
tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, win_length=384)
freqs = librosa.tempo_frequencies(tg.shape[0], sr=sr)
mean_tg = tg.mean(axis=1)
idx = np.argsort(mean_tg)[-10:][::-1]
print("top tempogram freqs", [(round(float(freqs[i]),2), round(float(mean_tg[i]),3)) for i in idx])
# autocorrelation of onset envelope
ac = librosa.autocorrelate(onset_env, max_size=sr*4)
lags = np.arange(len(ac))/ (len(onset_env)/ (len(y)/sr))  # frames->sec
sec = np.arange(len(ac))* (512/sr)
peaks = []
for bpm in np.arange(50, 320, 0.25):
    lag = 60.0/bpm
    i = int(round(lag/(512/sr)))
    if 0 < i < len(ac): peaks.append((ac[i], bpm))
peaks.sort(reverse=True)
print("ac top bpm", [(round(b,2), round(float(a),1)) for a,b in peaks[:12]])
