# ML Challenge Hack4Hack 2026

Solutions to both rounds of the Hack4Hack 2026 ML Challenge, two astronomy classification problems ranked by **Macro F1**.

**Team LLM_ops** (Vimal Singh, Piyush, [other members])

| Round | Task | Best public Macro F1 | Public rank | Final rank |
|-------|------|----------------------|-------------|------------|
| 1 (24 h) | Planet Hunters: classify TESS-like star light curves | 0.7850 | 2nd | [add] |
| 2 (finals, 1 day) | Signal Hunters: classify SETI ON/OFF radio cadences | 0.9312 | 3rd | [add] |

Round 1 qualified the team for the Round 2 finals.

---

## Round 1: Planet Hunters

**Task:** For each star, a 27-day light curve (1,315 cadences) plus centroid motion, predict one of 8 labels: `planet`, `eclipsing_binary`, `background_binary`, `starspots`, `pulsating_star`, `flare_star`, `instrument_glitch`, `quiet_star`. Overlapping signals are resolved by a priority order.

**Challenges:** tiny transits near the noise level, look-alike classes on purpose (background binaries, momentum-dump blips), a different observing sector in test, and about 2% label noise in training.

**Approach:** [fill in from your Round 1 notebook, e.g. detrending by CCD, BLS period search, odd/even and secondary-eclipse vetting, centroid-shift features, periodogram features, gradient boosting or 1D CNN]

**Code:** [`round1/` folder, or link if it lives elsewhere]

---

## Round 2: Signal Hunters

**Task:** Each sample is a cadence of 6 spectrograms (ON, OFF, ON, OFF, ON, OFF), each 16 time steps x 128 frequency channels. Predict one of 9 labels: `eti_candidate`, `frb`, `pulsar`, `scintillating_star`, `satellite_pass`, `radar_chirp`, `rfi_intermittent`, `rfi_persistent`, `noise`.

**Challenges:**
- **Domain shift:** train is from the GBT telescope, test from PKS (noisier, different bandpass, different label mix)
- **Rare class:** `eti_candidate` is about 3% of the data but counts equally under Macro F1
- **Adversarial traps:** ON-only lines that don't form one drifting track, zero-drift lines, lines present in ON and OFF panels

**Approach:**
1. **Preprocessing:** per-cadence normalisation and bandpass flattening to reduce the GBT to PKS gap
2. **Signal-processing features:** drift-track search across panels (`trk_*`), ON/OFF presence checks, broadband and dispersion statistics (`feats_*`)
3. **CNN models:** EfficientNet-based models (B0/B2/B3) that take the 6 panels as input channels, with frequency-shift and flip augmentation
4. **Student models:** [describe: e.g. distilled or pseudo-labelled models using PKS test data]
5. **Stacking:** out-of-fold predictions combined by a meta-learner (`stack.py`)

**Score progression (public):** 0.7765, 0.8040, 0.9003, 0.9158, 0.9257, **0.9312** over 13 uploads.

---

## Repository structure

```
.
├── PROBLEM_STATEMENT.md                 # Round 2 task description
├── round1/                              # [Round 1 code and problem statement]
├── starter_notebook.ipynb               # Organizer-provided baseline
├── signal_hunters_*.ipynb               # Round 2 pipelines (v3, v4)
├── stack.py                             # Final stacking script
├── submission_r2_v11_LB0.9257.csv       # Intermediate submission
└── .gitignore
```

Large artifacts (`*.npy`, `*.npz`, `*.pkl`, `work/`, `cache/`) are excluded. Put the competition data in the project root to reproduce.

## Key learnings

- Combining physics-based features (drift tracks, ON/OFF logic) with CNNs beat either alone
- Per-cadence normalisation was critical for transferring across telescopes
- Macro F1 means rare classes need explicit attentioning across telescopes
- Macro F1 means rare classes need explicit attention
