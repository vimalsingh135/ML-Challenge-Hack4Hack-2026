"""Final blend for Signal Hunters Round 2 (team LLM_ops).
Put the out/ folders from the three Kaggle notebooks into one folder (all test_*.npy files together), then:
    python make_submission.py <folder_with_npy_files> <test_meta.csv> submission.csv
Each test_<tag>.npy is a (10000, 9) probability matrix = mean of the 5 fold models (with flip TTA)."""
import sys, numpy as np, pandas as pd
LABELS = np.array(['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
                   'rfi_intermittent', 'rfi_persistent', 'noise'])
# weights chosen on held-out folds with simulated PKS-like noise (not on the public leaderboard)
WEIGHTS = {'effb2_fixup2': 0.3, 'effb0_fixup2': 0.3,     # notebook 3 (2x upscaled canvas)
           'effb2_fix': 0.1, 'effb0_fix': 0.1,           # notebook 1
           'effb2_fix45': 0.1, 'effb3_fix': 0.1}         # notebook 2
d, meta, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = sum(w * np.load(f'{d}/test_{t}.npy') for t, w in WEIGHTS.items())
te = pd.read_csv(meta)
sub = pd.DataFrame({'id': te.id, 'label': LABELS[P.argmax(1)]})
assert len(sub) == 10000 and sub.id.is_unique
sub.to_csv(out, index=False); print(sub.label.value_counts())
