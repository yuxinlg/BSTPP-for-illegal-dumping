import sys
import numpy as np
from scipy.stats import ks_2samp
pre = np.load(sys.argv[1]); post = np.load(sys.argv[2])
print(f"{'config':6} {'mean_n pre':>10} {'post':>8} {'z':>6}   KS(T) p     KS(X) p")
ok = True
for c in ("cox", "plain", "cov"):
    a, b = pre[c+"_n"].astype(float), post[c+"_n"].astype(float)
    z = (b.mean()-a.mean())/np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    pT = ks_2samp(pre[c+"_T"], post[c+"_T"]).pvalue
    pX = ks_2samp(pre[c+"_X"], post[c+"_X"]).pvalue
    print(f"{c:6} {a.mean():10.2f} {b.mean():8.2f} {z:6.2f}   {pT:9.3f}   {pX:9.3f}")
    ok &= abs(z) < 4 and pT > 0.01 and pX > 0.01
print("DISTRIBUTIONAL: PRESERVED" if ok else "DISTRIBUTIONAL: FAILED -- STOP AND REPORT")
sys.exit(0 if ok else 1)
