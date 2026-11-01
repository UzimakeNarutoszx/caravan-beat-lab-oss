import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, librosa, librosa.display
for name, path in [("drums","caravan_work/sep/htdemucs/drums.wav"),("mix","caravan_work/caravan.wav")]:
    for start in (0, 300):
        y, sr = librosa.load(path, sr=22050, mono=True, offset=start, duration=10)
        fig, ax = plt.subplots(2,1, figsize=(16,6), sharex=True)
        librosa.display.waveshow(y, sr=sr, ax=ax[0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=128)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=128, x_axis='time', y_axis='log', ax=ax[1])
        ax[1].set_ylim(30, 8000)
        for a in ax: a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"caravan_work/analysis/{name}_{start}.png", dpi=90); plt.close(fig)
print("ok")
