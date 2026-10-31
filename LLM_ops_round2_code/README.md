# Signal Hunters (Round 2) — team LLM_ops

We classify each 6-panel ON/OFF cadence into 9 labels in two stages:

1. **CNNs.** EfficientNet models, each trained with 5-fold cross-validation on Kaggle (`notebooks/`).
2. **Stacker.** A LightGBM model that combines the CNN probabilities with two sets of hand-built physics features (`stacker/`):
   - "level" features for scintillation
   - "track" features for ETI

Everything was trained only on the provided data. No test labels were made by hand.

## Reproduce

1. **Run the notebooks on Kaggle.** Notebook-to-model tags: 0 = effb2_fixs2/effb0_fixs2, 1 = effb2_fix/effb0_fix, 2 = effb2_fix45/effb3_fix, 3 = effb2_fixup2/effb0_fixup2, 5 = effb2_up2s5/effb0_up2s5, 6/7/8/9 = CNN v3 runs 14-17.
   - Settings: GPU T4 x2, competition data attached as input.
   - Each notebook takes about 3–4.5 h.
   - Each one is self-contained: cell 1 writes the training script, and cell 2 writes the pseudo-label file.
   - Copy every `out/oof_*.npy`, `out/oofpks_*.npy` and `out/test_*.npy` into one folder, `cache/`.
2. **Build the features and run the stacker.**
   - Run on CPU, from a folder that contains `data/` (the competition files) and `cache/`.
   - Takes about 20 min.
   ```
   export DATA=data/ CACHE=cache/
   python stacker/level_feats.py          # level features, train + test
   python stacker/level_feats_pks.py      # level features on PKS-like-noised train
   python stacker/track_feats.py train 1  # ETI track features, PKS-like-noised train
   python stacker/track_feats.py test 0   # ETI track features, test
   python stacker/feats_pks2.py lvl       # v2 simulation: + per-panel gain jitter measured on the real test telescope
   python stacker/feats_pks2.py trk
   FEATS=lvl,trk OUT=submission.csv python stacker/stack.py
   ```
   - By default `stack.py` uses the weighted blend of notebooks 1–3 as its input. Our final submission adds individual models through `INDIV`, for example:
   ```
   INDIV=effb2_fixup2,effb0_fixup2,effb2_fix,effb0_fix,effb2_fix45,effb3_fix FEATS=lvl,trk OUT=submission.csv python stacker/stack.py
   ```
   - The exact command for each uploaded file is listed in `SUBMISSIONS.md`.
3. **CNN-only fallback (no stacker):** `python make_submission.py cache test_meta.csv submission.csv`.

## Method

**CNN input image**
- Each cadence becomes one 128×128 "global-time canvas": the 6 panels sit on one time axis, slew gaps included. A real drifting signal then forms one straight line through the 3 ON panels.
- 4 channels: bandpass-flattened z-score, z minus the row median, ON/OFF sign, valid-pixel mask.
- CNN v3 (notebook 6) adds a 5th channel: each panel's brightness level relative to the OFF bandpass.

**Bandpass removal**
- Per-panel time-median spectrum, smoothed by a 9-channel median, so narrow lines are kept.

**Domain shift (GBT train → PKS test)**
Handled with augmentation:
- Gamma-distributed receiver noise in the power domain
- Extra dropped rows and broadband glitch rows
- Per-panel gain changes
- Frequency shift and flip

**Synthetic signals injected during training**
- ETI lines that drift continuously across the ON panels
- "Trap" lines that appear only in ON panels but don't connect or don't drift, labelled rfi_intermittent

**Pseudo-labels**
- Test cadences predicted with probability above 0.85 by earlier models, stored inside cell 2 of each notebook.

**Stacker features**
- *Level:* ON-panel brightening relative to the OFF bandpass. Per-panel normalisation erases this, but it is the main sign of scintillation.
- *Track:* for 81 drift rates, the line strength along one straight track through the 3 ON panels, compared with the OFF panels and with lines that don't connect.
- The stacker trains on PKS-like out-of-fold probabilities, and its training features are computed on PKS-like-noised data.

**Validation**
- Stratified 5-fold out-of-fold macro-F1, plus the same held-out folds with PKS-like noise added.
- All weights were chosen on these scores, never on the public leaderboard.
