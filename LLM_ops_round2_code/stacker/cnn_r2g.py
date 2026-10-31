"""Signal Hunters R2 - CNN on the 'global-time canvas'.
Each cadence -> one 128x128 image: the 6 panels placed on ONE time axis including the slew gaps
(panel p row r at t = p*(16+gap)+r), so a true ETI drift is a single straight line through the 3 ON panels.
Input channels: z (bandpass-flattened, noise-normalised), zr (z minus per-row median: kills broadband rows),
on/off indicator (+1 ON rows, -1 OFF rows, 0 gaps), valid mask.
Domain shift GBT->PKS is attacked with augmentation: frequency zoom (PKS drifts look ~0.64x as steep),
SNR loss, dropped rows, broadband glitch rows, per-panel gain, frequency shift/flip.
Usage: python cnn_r2.py DATA_DIR EPOCHS SEEDS [arch=res|timm_resnet18|timm_effb0] [PL]
Writes oof_<tag>.npy (18000x9) and test_<tag>.npy (10000x9) to ./out/"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn, sys, os, time, math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
LABELS = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
          'rfi_intermittent', 'rfi_persistent', 'noise']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
UP = int(os.environ.get('UP', 1)); LVL = int(os.environ.get('LVL', 1)); P_ETI = float(os.environ.get('P_ETI', 0)); P_TRAP = float(os.environ.get('P_TRAP', 0))

def normalise(X):
    x = X.astype(np.float32); m = (X > 0); x[~m] = np.nan
    x = x - np.nanmedian(x, axis=2, keepdims=True)
    s = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis=(2, 3), keepdims=True)), axis=(2, 3), keepdims=True) + 1e-3
    z = np.nan_to_num(np.clip(x / s, -10, 60))
    return z.astype(np.float16), m

def load(D, split):
    X = np.load(f'{D}/{split}_cadences.npy'); meta = pd.read_csv(f'{D}/{split}_meta.csv')
    return X, (X > 0), meta.slew_gap_rows.values.astype(np.int64), meta

def norm_t(x, m):
    """FIXED normalisation (v2). Old version subtracted each channel's median over time, which ERASED every
    non-drifting line (persistent RFI, non-drifting traps): a 10-sigma persistent line became 0.4 sigma.
    Now: bandpass = per-panel time-median spectrum smoothed ACROSS FREQUENCY (median, 9 channels), so narrow
    lines survive. Channel 64 (GBT-only centre spike, absent in PKS) is blanked in every cadence."""
    x = torch.where(m > 0, x, torch.full_like(x, float('nan')))
    x[..., 64] = float('nan')
    spec = torch.nanmedian(x, dim=2).values                                   # (B,6,128)
    spec = torch.where(torch.isnan(spec), torch.nanmedian(spec, dim=-1, keepdim=True).values, spec)
    pad = Fn.pad(spec, (4, 4), mode='replicate')
    band = pad.unfold(-1, 9, 1).median(-1).values                            # smooth bandpass
    r = x - band[:, :, None, :]
    f = r.flatten(2); med = torch.nanmedian(f, dim=2, keepdim=True).values
    s = 1.4826 * torch.nanmedian((f - med).abs(), dim=2, keepdim=True).values + 1e-3
    z = ((r - med[..., None]) / s[..., None]).clamp(-10, 60)
    # v3: LEVEL channel = what the per-panel bandpass removed, relative to the OFF-panel bandpass.
    # A scintillating star brightens the whole ON panel; the per-panel band absorbs it, so z alone cannot show it.
    bo = band[:, [1, 3, 5]].median(1, keepdim=True).values                    # (B,1,128) OFF bandpass
    so = s[:, [1, 3, 5]].median(1, keepdim=True).values                       # (B,1,1) OFF noise scale
    E = ((band - bo) / so).clamp(-10, 30)                                     # (B,6,128)
    return torch.nan_to_num(z, nan=0.0), torch.nan_to_num(E, nan=0.0)

def byte_aug(x, m, k_lo=5., k_hi=30., p=0.8):
    """PKS-like receiver noise in the POWER domain: power *= Gamma(k)/k  -> skewed log-noise like the test
    telescope (measured: noise MAD 15.6 vs 11.9 bytes, more negative skew). Also extra dropped/glitch rows."""
    B = x.shape[0]
    k = torch.empty(B, 1, 1, 1, device=x.device).uniform_(k_lo, k_hi)
    g = torch.distributions.Gamma(k.expand_as(x), k.expand_as(x)).sample().clamp_min(1e-4)
    sel = (torch.rand(B, 1, 1, 1, device=x.device) < p).float()
    x = x + sel * 24 * torch.log2(g)
    drop = (torch.rand(B, 6, 16, 1, device=x.device) < torch.empty(B, 1, 1, 1, device=x.device).uniform_(0, 0.03)).float()
    gl = (torch.rand(B, 6, 16, 1, device=x.device) < 0.01).float() * torch.empty(B, 6, 16, 1, device=x.device).uniform_(10, 50)
    x = (x + gl).clamp(1, 255)
    m = m * (1 - drop)
    return x, m

# ---------------- GPU-side augmentation + canvas -----------------
def augment(z, m, train, E):
    """z,m: (B,6,16,128) float on device"""
    B = z.shape[0]
    if not train: return z, m, E
    # SNR loss + extra noise (PKS noisier): z' = a*z + sqrt(1-a^2)*N
    a = torch.empty(B, 1, 1, 1, device=z.device).uniform_(0.75, 1.0)
    z = a * z + torch.sqrt(1 - a ** 2) * torch.randn_like(z)
    # per-panel gain on the signal part
    z = z * torch.empty(B, 6, 1, 1, device=z.device).uniform_(0.8, 1.2)
    # frequency zoom (compress/expand channel axis), p=0.7, s in [0.55,1.1]
    sel = torch.rand(B, device=z.device) < (0.15 if not LVL else 0.0)
    if sel.any():
        s = torch.ones(B, device=z.device); s[sel] = torch.empty(int(sel.sum()), device=z.device).uniform_(0.85, 1.15)
        xs = torch.linspace(-1, 1, 128, device=z.device)
        gx = xs[None, :] / s[:, None]                                   # sample positions in input
        grid = torch.stack([gx[:, None, :].expand(B, 96, 128), torch.linspace(-1, 1, 96, device=z.device)[None, :, None].expand(B, 96, 128)], -1)
        zz = Fn.grid_sample(z.reshape(B, 1, 96, 128), grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(B, 6, 16, 128)
        mm = Fn.grid_sample(m.reshape(B, 1, 96, 128).float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).reshape(B, 6, 16, 128)
        outside = (gx.abs() > 1)[:, None, None, :].expand(B, 6, 16, 128)
        # bilinear interpolation shrinks noise std -> renormalise per panel, fill outside with fresh noise
        zz = torch.where(outside, torch.randn_like(zz), zz)
        mm = torch.where(outside, torch.ones_like(mm), mm)
        sd = zz.flatten(2).std(2).clamp_min(0.3)[..., None, None]
        zz = zz / torch.where(sd < 1, sd, torch.ones_like(sd))
        z = torch.where(sel[:, None, None, None], zz, z); m = torch.where(sel[:, None, None, None], mm, m.float())
    m = m.float()
    # frequency shift (roll) and flip
    sh = int(np.random.randint(-30, 31)); z = torch.roll(z, sh, -1); m = torch.roll(m, sh, -1); E = torch.roll(E, sh, -1)
    fl = torch.rand(B, device=z.device) < 0.5
    z = torch.where(fl[:, None, None, None], z.flip(-1), z); m = torch.where(fl[:, None, None, None], m.flip(-1), m)
    E = torch.where(fl[:, None, None], E.flip(-1), E)
    E = E + 0.15 * torch.randn(B, 6, 1, device=z.device)                      # per-panel gain jitter on the level
    return z, m, E

def up(C):
    return Fn.interpolate(C, scale_factor=UP, mode='nearest') if UP > 1 else C

def canvas(z, m, gap, E=None):
    """(B,6,16,128) -> (B,4+LVL,128,128) global-time canvas"""
    B = z.shape[0]; H = 128
    zr = z - z.median(-1, keepdim=True).values
    C = torch.zeros(B, 4 + LVL, H, 128, device=z.device)
    p = torch.arange(6, device=z.device); r = torch.arange(16, device=z.device)
    t = p[None, :, None] * (16 + gap[:, None, None]) + r[None, None, :]          # (B,6,16)
    t = t + ((H - (6 * 16 + 5 * gap)) // 2)[:, None, None]                       # centre vertically
    bi = torch.arange(B, device=z.device)[:, None, None].expand(B, 6, 16)
    sign = torch.tensor([1., -1, 1, -1, 1, -1], device=z.device)[None, :, None].expand(B, 6, 16)
    C[bi, 0, t] = z * m; C[bi, 1, t] = zr * m
    C[bi, 2, t] = sign[..., None].expand(B, 6, 16, 128); C[bi, 3, t] = m
    if LVL: C[bi, 4, t] = 3 * E[:, :, None, :].expand(B, 6, 16, 128)        # x3: same scale as z
    return C


def inject(z, m, gap, yb, p_eti, p_trap):
    """Physics-based synthetic signals (labels follow the priority order):
    - ETI: a narrow drifting line on ONE straight track in global time, ON panels only -> label eti_candidate
    - trap: ON-only lines that do NOT connect (random offsets) or do not drift -> rfi_intermittent
      (only injected into noise / rfi_persistent cadences, whose priority is lower)"""
    B = z.shape[0]; d_ = z.device
    t = (torch.arange(6, device=d_)[None, :, None] * (16 + gap[:, None, None]) + torch.arange(16, device=d_)[None, None, :]).float()
    tc = t[:, [0, 2, 4]].mean((1, 2))[:, None, None]
    ch = torch.arange(128, device=d_).float()[None, None, None, :]
    on = torch.tensor([1., 0, 1, 0, 1, 0], device=d_)[None, :, None, None]
    r = torch.rand(B, device=d_)
    e = (r < p_eti) & (yb != 0)
    tr_ = (r >= p_eti) & (r < p_eti + p_trap) & ((yb == 7) | (yb == 8))
    if e.any() or tr_.any():
        drift = torch.empty(B, device=d_).uniform_(0.15, 1.3) * (torch.randint(0, 2, (B,), device=d_) * 2 - 1)
        c0 = torch.empty(B, device=d_).uniform_(20, 108)
        amp = torch.empty(B, device=d_).uniform_(0.6, 3.0); wid = torch.empty(B, device=d_).uniform_(0.5, 1.2)
        centre = c0[:, None, None] + drift[:, None, None] * (t - tc)                      # (B,6,16) connected track
        kind = torch.randint(0, 2, (B,), device=d_)                                       # trap type
        off = torch.empty(B, 6, 1, device=d_).uniform_(-40, 40)                           # disconnect offsets
        tr_centre = torch.where(kind[:, None, None] == 0, c0[:, None, None] + off + drift[:, None, None] * (t - t[:, :, :1]),
                                c0[:, None, None] + 0 * t)                                # non-connecting / non-drifting
        cen = torch.where(e[:, None, None], centre, tr_centre)
        line = amp[:, None, None, None] * torch.exp(-0.5 * ((ch - cen[..., None]) / wid[:, None, None, None]) ** 2) * on
        sel = (e | tr_)[:, None, None, None].float()
        z = z + sel * line * m
        yb = torch.where(e, torch.zeros_like(yb), torch.where(tr_, torch.full_like(yb, 6), yb))
    return z, yb

# ---------------- model -----------------
class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.sk = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if (i != o or st != 1) else nn.Identity()
    def forward(s, x): return torch.relu(s.b2(s.c2(torch.relu(s.b1(s.c1(x))))) + s.sk(x))

class ResNet(nn.Module):
    def __init__(s, w=40):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(4 + LVL, w, (5, 3), 1, (2, 1), bias=False), nn.BatchNorm2d(w), nn.ReLU())
        s.body = nn.Sequential(Block(w, 2 * w, 2), Block(2 * w, 2 * w, 1),
                               Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 1), Block(4 * w, 6 * w, 2), Block(6 * w, 6 * w, 1),
                               Block(6 * w, 8 * w, 2))
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(16 * w, 9))
    def forward(s, x):
        h = s.body(s.stem(x)); return s.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))

def make(arch):
    if arch == 'res': return ResNet()
    import timm
    name = {'timm_resnet18': 'resnet18', 'timm_effb0': 'efficientnet_b0', 'timm_effb2': 'efficientnet_b2', 'timm_convnext': 'convnext_nano', 'timm_effb3': 'efficientnet_b3'}[arch]
    try: return timm.create_model(name, pretrained=True, in_chans=4 + LVL, num_classes=9)
    except Exception as e: print('pretrained download failed, random init:', e); return timm.create_model(name, pretrained=False, in_chans=4 + LVL, num_classes=9)

def predict(net, Z, M, G, tta=True):
    EE = None
    if isinstance(Z, tuple): Z, EE = Z
    net.eval(); out = []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == 'cuda'):
        for j in range(0, len(Z), 512):
            m = torch.from_numpy(M[j:j + 512]).to(dev).float()
            z = torch.from_numpy(Z[j:j + 512]).to(dev).float()
            if Z.dtype == np.uint8: z, E = norm_t(z, m)
            else: E = torch.from_numpy(EE[j:j + 512]).to(dev).float()
            g = torch.from_numpy(G[j:j + 512]).to(dev)
            p = torch.softmax(net(up(canvas(z, m, g, E))).float(), 1)
            if tta: p = 0.5 * (p + torch.softmax(net(up(canvas(z.flip(-1), m.flip(-1), g, E.flip(-1)))).float(), 1))
            out.append(p.cpu().numpy())
    return np.concatenate(out)

def pks_like(Z, M, seed=0):
    """deterministic 'PKS-like' version of held-out GBT data for shift-aware validation"""
    torch.manual_seed(seed); out_z, out_m, out_e = [], [], []
    for j in range(0, len(Z), 512):
        x = torch.from_numpy(Z[j:j + 512]).to(dev).float(); m = torch.from_numpy(M[j:j + 512]).to(dev).float(); B = len(x)
        x, m = byte_aug(x, m, 12., 12., 1.0); z, E = norm_t(x, m); out_e.append(E.cpu().numpy().astype(np.float16))
        xs = torch.linspace(-1, 1, 128, device=dev) / 1.0
        grid = torch.stack([xs[None, None, :].expand(B, 96, 128), torch.linspace(-1, 1, 96, device=dev)[None, :, None].expand(B, 96, 128)], -1)
        zz = Fn.grid_sample(z.reshape(B, 1, 96, 128), grid, align_corners=True).reshape(B, 6, 16, 128)
        out = (xs.abs() > 1)[None, None, None, :].expand_as(zz)
        zz = torch.where(out, torch.randn_like(zz), zz); sd = zz.flatten(2).std(2).clamp_min(0.3)[..., None, None]
        zz = zz / torch.where(sd < 1, sd, torch.ones_like(sd))
        mm = torch.where(out, torch.ones_like(m), Fn.grid_sample(m.reshape(B, 1, 96, 128), grid, mode='nearest', align_corners=True).reshape(B, 6, 16, 128))
        out_z.append(zz.cpu().numpy().astype(np.float16)); out_m.append(mm.cpu().numpy() > 0.5)
    return (np.concatenate(out_z), np.concatenate(out_e)), np.concatenate(out_m)

def fit(Z, M, G, y, w_cls, epochs, seed, arch, evals):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make(arch).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0.02)
    bs = int(os.environ.get('BS', 128)); n = len(y); steps = epochs * (n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w_cls, device=dev, dtype=torch.float32), label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda'); Y = torch.from_numpy(y)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(n); t0 = time.time(); L = 0
        for j in range(0, n - bs + 1, bs):
            idx = perm[j:j + bs]
            x = torch.from_numpy(Z[idx]).to(dev).float(); m = torch.from_numpy(M[idx]).to(dev).float()
            g = torch.from_numpy(G[idx]).to(dev); yb = Y[idx].to(dev)
            x, m = byte_aug(x, m); z, E = norm_t(x, m)
            if P_ETI + P_TRAP > 0: z, yb = inject(z, m, g, yb, P_ETI, P_TRAP)
            z, m, E = augment(z, m, True, E)
            with torch.autocast(dev, enabled=dev == 'cuda'):
                loss = lossf(net(up(canvas(z, m, g, E))), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); L += loss.item() * bs
        if ep % 5 == 4 or ep == epochs - 1: print(f'   ep{ep} loss {L / n:.3f} {time.time() - t0:.1f}s', flush=True)
    return [predict(net, *e) for e in evals]

if __name__ == '__main__':
    D, EP, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    arch = sys.argv[4] if len(sys.argv) > 4 else 'res'; PL = 'PL' in sys.argv
    tag = os.environ.get('TAG', arch); SMOKE = int(os.environ.get('SMOKE', 0)); os.makedirs('out', exist_ok=True)
    Z, M, G, meta = load(D, 'train'); Zt, Mt, Gt, tmeta = load(D, 'test')
    y = meta.label.map({l: i for i, l in enumerate(LABELS)}).values
    if SMOKE: Z, M, G, y = Z[:SMOKE], M[:SMOKE], G[:SMOKE], y[:SMOKE]; Zt, Mt, Gt = Zt[:SMOKE], Mt[:SMOKE], Gt[:SMOKE]
    cnt = np.bincount(y, minlength=9); w_cls = (cnt.sum() / (9 * np.maximum(cnt, 1))) ** 0.5; w_cls = w_cls / w_cls.mean()
    print('device', dev, 'arch', arch, 'train', Z.shape, 'class weights', w_cls.round(2), flush=True)
    if PL:
        pl = np.load('out/pl.npz'); sel, ypl = pl['sel'], pl['y']
        print('pseudo-labelled test cadences', sel.sum(), flush=True)
    oof = np.zeros((len(y), 8 + 1)); oof_pks = np.zeros_like(oof); pt = np.zeros((len(Zt), 9))
    for s in range(SEEDS):
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y)):
            if os.environ.get('FOLDS') and str(k) not in os.environ['FOLDS'].split(','): continue
            Za, Ma, Ga, ya = Z[a], M[a], G[a], y[a]
            if PL: Za, Ma, Ga, ya = np.concatenate([Za, Zt[sel]]), np.concatenate([Ma, Mt[sel]]), np.r_[Ga, Gt[sel]], np.r_[ya, ypl]
            Zp, Mp = pks_like(Z[b], M[b], seed=k)
            pb, pp, pte = fit(Za, Ma, Ga, ya, w_cls, EP, 100 * s + k + int(os.environ.get('SEED_OFF', 0)), arch, [(Z[b], M[b], G[b]), (Zp, Mp, G[b]), (Zt, Mt, Gt)])
            oof[b] += pb / SEEDS; oof_pks[b] += pp / SEEDS; pt += pte / (5 * SEEDS)
            print(f'seed {s} fold {k}: F1 {f1_score(y[b], pb.argmax(1), average="macro"):.4f} | PKS-like F1 {f1_score(y[b], pp.argmax(1), average="macro"):.4f}', flush=True)
            np.save(f'out/oof_{tag}_partial.npy', oof); np.save(f'out/test_{tag}_partial.npy', pt); np.save(f'out/folds_done_{tag}.npy', np.array([s, k]))
        f = SEEDS / (s + 1)
        print(f'== after seed {s}: OOF F1 {f1_score(y, (oof * f).argmax(1), average="macro"):.4f} | PKS-like {f1_score(y, (oof_pks * f).argmax(1), average="macro"):.4f}', flush=True)
        print('per class (PKS-like):', dict(zip(LABELS, f1_score(y, (oof_pks * f).argmax(1), average=None).round(3))), flush=True)
        np.save(f'out/oof_{tag}.npy', oof * f); np.save(f'out/oofpks_{tag}.npy', oof_pks * f); np.save(f'out/test_{tag}.npy', pt * f)
    print('DONE', flush=True)
