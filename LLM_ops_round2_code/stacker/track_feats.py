"""Lever 3: ETI track features. A real ETI signal = one narrow line on ONE straight drifting track in GLOBAL time
through ON panels 1,3,5 and absent in OFF. Traps (rfi_intermittent) = ON-only lines that do not connect, or do not drift.
For each drift rate d (channels/row) shift every row so the track becomes vertical, sum rows per panel -> line strength."""
import numpy as np, pandas as pd, torch, sys, importlib.util, time, os
torch.set_num_threads(2)
ARGV = list(sys.argv); sp = importlib.util.spec_from_file_location('g', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_r2g.py')); g = importlib.util.module_from_spec(sp); sys.argv = ['x']; sp.loader.exec_module(g); sys.argv = ARGV
D = os.environ.get('DATA', 'data/')
DR = torch.arange(-2.0, 2.0001, 0.05)                                   # 81 drift rates
ON, OFF = [0, 2, 4], [1, 3, 5]

def feats(x, gap):
    m = (x > 0).float(); z, E = g.norm_t(x, m); z = z * m
    z = z.clamp(-5, 30)
    z = (z + z.roll(1, -1) + z.roll(-1, -1)) / 3                        # tolerate 1-2 channel width / rounding
    B = len(z)
    t = (torch.arange(6)[None, :, None] * (16 + gap[:, None, None]) + torch.arange(16)[None, None, :]).float()   # (B,6,16)
    tc = t[:, ON].mean((1, 2))[:, None, None]
    n_on = m[:, ON].sum((1, 2)).clamp_min(1)                           # valid rows*... per channel
    best = torch.full((B,), -1e9); keep = {}
    S_all = []
    ch = torch.arange(128)
    for d in DR:
        sh = torch.round(d * (t - tc)).long()                          # (B,6,16)
        idx = (ch[None, None, None, :] + sh[..., None])                # sample channel c+shift
        valid = (idx >= 0) & (idx < 128)
        zz = torch.gather(z, 3, idx.clamp(0, 127)) * valid
        P = zz.sum(2) / 4.0                                             # (B,6,128) per-panel line sum / sqrt(16)
        S_all.append(P)
    S = torch.stack(S_all, 1)                                           # (B,81,6,128)
    on, off = S[:, :, ON], S[:, :, OFF]
    con = on.sum(2) / 3 ** 0.5                                          # connected ON track score
    mn = on.min(2).values                                               # weakest ON panel on that track
    offmax = off.max(2).values
    score = mn - 0.5 * offmax.clamp_min(0)
    flat = score.flatten(1); k = flat.argmax(1); di, ci = k // 128, k % 128
    bi = torch.arange(B)
    f = {
        'trk_score': flat.max(1).values,
        'trk_con': con[bi, di, ci], 'trk_min_on': mn[bi, di, ci], 'trk_off': offmax[bi, di, ci],
        'trk_absdrift': DR[di].abs(),
        'con_max': con.flatten(1).max(1).values,
        'min_on_max': mn.flatten(1).max(1).values,
        # disconnected: each ON panel's best line anywhere (any drift, any channel) vs the connected track
        'indep_on_sum': on.amax((1, 3)).sum(1) / 3 ** 0.5,
        'indep_min_on': on.amax((1, 3)).min(1).values,
        'indep_off_max': off.amax((1, 3)).max(1).values,
        # zero-drift line strength ON-only
        'zero_min_on': mn[:, 40].max(1).values, 'zero_off': offmax[:, 40].gather(1, mn[:, 40].argmax(1, keepdim=True))[:, 0],
        # best non-zero-drift ON-only track (|d| >= 0.1)
        'nz_score': score[:, (DR.abs() >= 0.1)].flatten(1).max(1).values,
    }
    f['disconnect_gap'] = f['indep_on_sum'] - f['con_max']
    f['nz_minus_zero'] = f['nz_score'] - f['zero_min_on']
    # level/patchiness from E (scintillation): ON-OFF of smoothed excess
    Es = torch.nn.functional.avg_pool1d(E, 8, 1, 4)[..., :128]
    f['E_on_off'] = E[:, ON].mean((1, 2)) - E[:, OFF].mean((1, 2))
    f['E_on_min'] = E[:, ON].mean(2).min(1).values - E[:, OFF].mean(2).max(1).values
    f['Es_patch'] = Es[:, ON].std(2).mean(1) - Es[:, OFF].std(2).mean(1)
    f['Es_max'] = Es[:, ON].amax(2).mean(1) - Es[:, OFF].amax(2).mean(1)
    return torch.stack(list(f.values()), 1).numpy(), list(f.keys())

def run(split, noise=False, seed=0):
    X = np.load(D + f'{split}_cadences.npy', mmap_mode='r'); gap = pd.read_csv(D + f'{split}_meta.csv').slew_gap_rows.values
    torch.manual_seed(seed); out = []; t0 = time.time()
    for j in range(0, len(X), 500):
        x = torch.from_numpy(X[j:j + 500].astype(np.float32))
        if noise:
            m = (x > 0).float(); x, m = g.byte_aug(x, m, 12., 12., 1.0); x = x * (m > 0)
        F, names = feats(x, torch.from_numpy(gap[j:j + 500]).long()); out.append(F)
        if j == 0: print('first chunk', round(time.time() - t0, 1), 's', flush=True)
    return np.concatenate(out).astype(np.float32), names

if __name__ == '__main__':
    split, noise = sys.argv[1], sys.argv[2] == '1'
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    F, names = run(split, noise)
    tag = f'trk_{split}' + ('pks' if noise else '')
    np.save(os.environ.get('CACHE', 'cache/') + f'{tag}.npy', F); print(tag, F.shape, names)
