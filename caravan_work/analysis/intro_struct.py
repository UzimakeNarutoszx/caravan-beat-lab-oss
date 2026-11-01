import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
y, sr = librosa.load("caravan_work/caravan.wav", sr=22050, mono=True, duration=60)
# low band energy envelope (kick proxy) and high band
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
low = S[(freqs>30)&(freqs<120)].sum(0)
mid = S[(freqs>200)&(freqs<2000)].sum(0)
high = S[(freqs>5000)].sum(0)
def envdb(x): return librosa.amplitude_to_db(x, ref=np.max)
t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=256)
for name, e in [("low",low),("mid",mid),("high",high)]:
    ed = envdb(e)
    # peaks
    pk = librosa.util.peak_pick(ed, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=10, wait=10)
    print(name,"peak times", t[pk][:40].round(2))
# tempo of low band
oe = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr, hop_length=256)
for bpm0 in (70,140):
    print("low tempo", bpm0, librosa.feature.tempo(onset_envelope=oe, sr=sr, hop_length=256, start_bpm=bpm0, std_bpm=1.0))
