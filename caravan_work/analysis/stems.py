import numpy as np, soundfile as sf, warnings
warnings.filterwarnings("ignore")
for s in ("drums","bass","other","vocals"):
    x, sr = sf.read(f"caravan_work/sep/htdemucs/{s}.wav", always_2d=True)
    rms = np.sqrt((x**2).mean())
    print(s, "rms_db %.1f"% (20*np.log10(rms+1e-12)), "peak %.3f"%np.abs(x).max())
