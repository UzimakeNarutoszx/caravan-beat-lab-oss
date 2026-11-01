import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
y, sr = librosa.load("caravan_work/caravan.wav", sr=22050, mono=True)
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
for start_bpm in (70,140):
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512,
                                           start_bpm=start_bpm, tightness=100, trim=True)
    t = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    d = np.diff(t)
    print("start_bpm",start_bpm,"tempo",tempo,"nbeats",len(t),"first",t[:8].round(3),"last",t[-3:].round(3))
    print("  intervals mean %.4f median %.4f std %.4f min %.3f max %.3f -> bpm %.2f"%(d.mean(),np.median(d),d.std(),d.min(),d.max(),60/np.median(d)))
    # local bpm drift
    print("  first20 bpm", round(60/np.median(d[:20]),2), "last20", round(60/np.median(d[-20:]),2))
# RMS / onset at start
rms = librosa.feature.rms(y=y, hop_length=512)[0]
rmsdb = librosa.amplitude_to_db(rms, ref=np.max)
t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
for s in [0,1,2,3,4,5,6,8,10,12]:
    i = np.argmin(abs(t-s)); print("t",s,"rms_db",round(float(rmsdb[i]),1))
# first significant onset
on = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512, units='time')
print("first onsets", on[:15].round(3))
