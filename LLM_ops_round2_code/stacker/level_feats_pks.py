"""Level features on train cadences with PKS-like noise (Gamma k=12 power noise), matching the validation simulation."""
import numpy as np, os
from level_feats import feats, DATA, CACHE
X=np.load(DATA+'train_cadences.npy',mmap_mode='r'); rng=np.random.default_rng(12)
out=[]
for j in range(0,len(X),2000):
    x=X[j:j+2000].astype(np.float32); m=x>0
    g=np.maximum(rng.gamma(12.,1/12.,size=x.shape),1e-4); x=np.clip(x+24*np.log2(g),1,255)
    x=np.where(m,x,0); out.append(feats(x))
F=np.concatenate(out); np.save(CACHE+'lvl_trainpks.npy',F.astype(np.float32)); print(F.shape)
