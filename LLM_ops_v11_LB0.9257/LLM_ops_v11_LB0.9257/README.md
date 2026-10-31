# LLM_ops v11: public leaderboard 0.9257 (validation, PKS-like v2: 0.9307)

## Class order (the columns of every probability file)
0 eti_candidate, 1 frb, 2 pulsar, 3 scintillating_star, 4 satellite_pass, 5 radar_chirp, 6 rfi_intermittent, 7 rfi_persistent, 8 noise

Rows follow train_meta.csv (18,000) and test_meta.csv (10,000).

## The 12 models used in v11 (`oof_`, `oofpks_` and `test_<tag>.npy` for each)

| Tags | Run | What it is |
|---|---|---|
| effb2_lvlfast, effb0_lvlfast | run 17 | new CNN with the brightness channel, normal canvas |
| effb2_fixup2, effb0_fixup2 | run 8 | 2x canvas |
| effb2_up2s5, effb0_up2s5 | run 13 | 2x canvas, new pseudo-labels |
| effb2_fix, effb0_fix | run 7 | |
| effb2_fixs2, effb0_fixs2 | run 10 | |
| effb2_fix45 | run 9 | 45 epochs |
| effb3_fix | run 9 | |

## What each file holds
- `oof_` (18000x9): out-of-fold probabilities on clean train. Folds are StratifiedKFold(5, shuffle=True, random_state=0).
- `oofpks_` (18000x9): the same folds with PKS-like noise added (Gamma k=12).
- `test_` (10000x9): test probabilities, averaged over the 5 fold models with flip test-time augmentation.
- `lvl_*` (24 features) and `trk_*` (19 features): stacker features. Use `_trainpks2` for training, which matches the real test's per-panel gain jitter.
- `oofpks_stack_v11` / `test_stack_v11`: the v11 stacker output. Its out-of-fold rows use StratifiedKFold(5, shuffle=True, random_state=1).

## Rebuild v11
```
SPEC=effb2_lvlfast:0.25,effb0_lvlfast:0.15,effb2_fixup2:0.15,effb0_fixup2:0.1,effb2_up2s5:0.15,effb0_up2s5:0.1,effb2_fix45:0.05,effb3_fix:0.05 \
INDIV=effb2_lvlfast,effb0_lvlfast,effb2_fixup2,effb0_fixup2,effb2_up2s5,effb0_up2s5,effb2_fix,effb0_fix,effb2_fixs2,effb0_fixs2,effb2_fix45,effb3_fix \
FSPLIT=trainpks2 FEATS=lvl,trk CACHE=cache/ DATA=<competition data>/ OUT=sub.csv python stack.py
```

## Caveat
The `oofpks` files for the run 17 models (`*_lvlfast`) used the older simulation, which has no gain jitter. The real leaderboard came in 0.005 below validation for v11.
