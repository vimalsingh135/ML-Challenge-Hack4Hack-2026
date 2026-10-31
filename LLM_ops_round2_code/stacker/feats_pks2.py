"""PKS-like train features v2: Gamma k=12 power noise PLUS per-panel gain offsets (measured: real PKS test has
~40% wider panel-to-panel level spread than our simulation). Offset std = SIG x (per-cadence noise scale)."""
import numpy as np, pandas as pd, torch, sys, os, importlib.util
torch.set_num_threads(2)
SIG = float(os.environ.get('SIG', 0.13)); PART = sys.argv[1]   # 'lvl' or 'trk'
D = os.environ.get('DATA', 'data/'); HERE = os.path.dirname(os.path.abspath(__file__))
X = np.load(D + 'train_cadences.npy', mmap_mode='r'); gap = pd.read_csv(D + 'train_meta.csv').slew_gap_rows.values
rng = np.random.default_rng(12); torch.manual_seed(0)
def noisy(x):
    m = x > 0
    g = np.maximum(rng.gamma(12., 1/12., size=x.shape), 1e-4); x = np.clip(x + 24*np.log2(g), 1, 255)
    xs = np.where(m, x, np.nan); s = 1.4826*np.nanmedian(np.abs(xs - np.nanmedian(xs, (1,2,3), keepdims=True)), (1,2,3))
    off = rng.normal(0, SIG, size=(len(x), 6, 1, 1)) * s[:, None, None, None]
    return np.where(m, np.clip(x + off, 1, 255), 0).astype(np.float32)
if PART == 'lvl':
    ns = {'__name__': 'lf', '__file__': os.path.join(HERE, 'level_feats.py')}; exec(open(os.path.join(HERE, 'level_feats.py')).read().split('if __name__')[0], ns); out = []
    for j in range(0, len(X), 2000): out.append(ns['feats'](noisy(X[j:j+2000].astype(np.float32))))
else:
    ARGV = list(sys.argv); sys.argv = ['x']
    ns = {'__name__': 'tf', '__file__': os.path.join(HERE, 'track_feats.py')}; exec(open(os.path.join(HERE, 'track_feats.py')).read().split("if __name__")[0], ns); sys.argv = ARGV
    out = []
    for j in range(0, len(X), 500):
        F, _ = ns['feats'](torch.from_numpy(noisy(X[j:j+500].astype(np.float32))), torch.from_numpy(gap[j:j+500]).long()); out.append(F)
F = np.concatenate(out).astype(np.float32); np.save(os.environ.get('CACHE', 'cache/') + f'{PART}_trainpks2.npy', F); print(PART, F.shape)
