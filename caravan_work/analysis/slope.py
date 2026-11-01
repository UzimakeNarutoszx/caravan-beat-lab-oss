import numpy as np
d = np.load("caravan_work/beats.npz")
tb, ts = d["tb"], d["ts"]
k = np.arange(len(tb))
# local tempo from regression in windows of 100 beats
print(" t(s)  bpm(window 60 beats)")
for c in range(0, len(tb)-60, 60):
    sl = np.polyfit(k[c:c+60], tb[c:c+60], 1)[0]
    print("%6.1f  %.2f"%(tb[c], 60/sl), end="  |  ")
print()
# compare constant tempo fit
coef = np.polyfit(k, tb, 1); pred = np.polyval(coef, k); r = tb-pred
print("constant %.3f bpm resid std %.3f max %.3f"%(60/coef[0], r.std(), np.abs(r).max()))
print("resid at 0,25,50,75,100%%:", [round(float(r[int(q*(len(r)-1))]),2) for q in (0,.25,.5,.75,1)])
# spline slope
sp = np.polyfit(k, ts, 1)
print("spline-based mean bpm %.2f"% (60/np.polyfit(k,ts,1)[0]))
