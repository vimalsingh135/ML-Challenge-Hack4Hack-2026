"""Per-cadence 'level' features that current norm_t erases: ON-panel brightening relative to OFF bandpass."""
import os
DATA = os.environ.get('DATA', 'data/')      # folder with train/test cadences + meta csv
CACHE = os.environ.get('CACHE', 'cache/')   # folder with the notebooks' oof/oofpks/test .npy files; features are written here too
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings('ignore')
from scipy.ndimage import uniform_filter1d
D=DATA
def feats(X):
    out=[]
    for j in range(0,len(X),1000):
        x=X[j:j+1000].astype(np.float32); x[x==0]=np.nan; x[...,64]=np.nan
        off=x[:,[1,3,5]].reshape(len(x),48,128); bO=np.nanmedian(off,1)
        bO=np.where(np.isnan(bO),np.nanmedian(bO,-1,keepdims=True),bO)
        s=1.4826*np.nanmedian(np.abs(off-np.nanmedian(off,1,keepdims=True)),(1,2))+1e-3
        pm=np.nanmedian(x,2)                         # (B,6,128) per-panel spectrum
        pm=np.where(np.isnan(pm),bO[:,None,:],pm)
        ex=(pm-bO[:,None,:])/s[:,None,None]          # excess spectrum per panel, sigma units
        lvl=np.median(ex,2)                          # (B,6) panel level
        exs=uniform_filter1d(ex,16,axis=2)
        mx=exs.max(2); frac=(exs>1.0).mean(2)
        on,of=[0,2,4],[1,3,5]
        f=np.c_[lvl, mx, frac, lvl[:,on].mean(1)-lvl[:,of].mean(1), lvl[:,on].min(1)-lvl[:,of].max(1),
                mx[:,on].mean(1)-mx[:,of].mean(1), mx[:,on].min(1), frac[:,on].mean(1)-frac[:,of].mean(1),
                (ex[:,on]-ex[:,of]).std(2).mean(1)]
        out.append(f)
    return np.concatenate(out)
if __name__ == '__main__':
  for sp in ['train','test']:
    X=np.load(D+f'{sp}_cadences.npy',mmap_mode='r'); F=feats(X); np.save(CACHE+f'lvl_{sp}.npy',F.astype(np.float32)); print(sp,F.shape)
