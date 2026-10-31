# LLM_ops cache for v10 (team-internal)

Leaderboard: v9 = 0.9158, v10 = 0.9162. Validation (PKS-like v2): v9 = 0.9162, v10 = 0.9180.

## Class order (the columns of every probability file)
0 eti_candidate, 1 frb, 2 pulsar, 3 scintillating_star, 4 satellite_pass, 5 radar_chirp, 6 rfi_intermittent, 7 rfi_persistent, 8 noise

Rows follow the order of train_meta.csv (18,000) and test_meta.csv (10,000) exactly.

## Files
- `oof_<model>.npy` (18000x9): out-of-fold probabilities on clean GBT train. CNN folds use StratifiedKFold(5, shuffle=True, random_state=0) on train labels.
- `oofpks_<model>.npy` (18000x9): the same held-out folds after PKS-like noise (Gamma k=12 in the power domain). This is our validation for the test telescope.
- `test_<model>.npy` (10000x9): test probabilities, averaged over the 5 fold models with flip test-time augmentation.
- `lvl_*.npy` (24 features) and `trk_*.npy` (19 features): stacker features. Suffix `_train` = clean train, `_test` = test, `_trainpks` = noisy train (v1 simulation), `_trainpks2` = noisy train plus per-panel gain jitter (v2 simulation, which matches the real test; use this one).
- `oofpks_stack_v10.npy` / `test_stack_v10.npy`: stacker output for v10. Its out-of-fold rows use StratifiedKFold(5, shuffle=True, random_state=1).
- `stack.py`: the stacker script. The v10 command is below.

lvl columns: lvl_p0, lvl_p1, lvl_p2, lvl_p3, lvl_p4, lvl_p5, mx_p0, mx_p1, mx_p2, mx_p3, mx_p4, mx_p5, frac_p0, frac_p1, frac_p2, frac_p3, frac_p4, frac_p5, lvl_on_minus_off, lvl_on_min_minus_off_max, mx_on_minus_off, mx_on_min, frac_on_minus_off, excess_on_off_std

trk columns: trk_score, trk_con, trk_min_on, trk_off, trk_absdrift, con_max, min_on_max, indep_on_sum, indep_min_on, indep_off_max, zero_min_on, zero_off, nz_score, disconnect_gap, nz_minus_zero, E_on_off, E_on_min, Es_patch, Es_max

## Rebuild v10
```
SPEC=effb2_fixup2:0.2,effb0_fixup2:0.2,effb2_up2s5:0.2,effb0_up2s5:0.2,effb2_fix:0.05,effb0_fix:0.05,effb2_fix45:0.05,effb3_fix:0.05 \
INDIV=effb2_fixup2,effb0_fixup2,effb2_up2s5,effb0_up2s5,effb2_fix,effb0_fix,effb2_fixs2,effb0_fixs2,effb2_fix45,effb3_fix \
FSPLIT=trainpks2 FEATS=lvl,trk CACHE=cache/ DATA=<competition data folder>/ OUT=sub.csv python stack.py
```

Not included yet: runs 14–17 (new CNN with the brightness channel). They are still training on Kaggle and will be added when they finish.

| file | shape | dtype |
|---|---|---|
| lvl_test.npy | (10000, 24) | float32 |
| lvl_train.npy | (18000, 24) | float32 |
| lvl_trainpks.npy | (18000, 24) | float32 |
| lvl_trainpks2.npy | (18000, 24) | float32 |
| oof_effb0_fix.npy | (18000, 9) | float64 |
| oof_effb0_fixs2.npy | (18000, 9) | float64 |
| oof_effb0_fixup2.npy | (18000, 9) | float64 |
| oof_effb0_up2s5.npy | (18000, 9) | float64 |
| oof_effb2_fix.npy | (18000, 9) | float64 |
| oof_effb2_fix45.npy | (18000, 9) | float64 |
| oof_effb2_fixs2.npy | (18000, 9) | float64 |
| oof_effb2_fixup2.npy | (18000, 9) | float64 |
| oof_effb2_up2s5.npy | (18000, 9) | float64 |
| oof_effb3_fix.npy | (18000, 9) | float64 |
| oofpks_effb0_fix.npy | (18000, 9) | float64 |
| oofpks_effb0_fixs2.npy | (18000, 9) | float64 |
| oofpks_effb0_fixup2.npy | (18000, 9) | float64 |
| oofpks_effb0_up2s5.npy | (18000, 9) | float64 |
| oofpks_effb2_fix.npy | (18000, 9) | float64 |
| oofpks_effb2_fix45.npy | (18000, 9) | float64 |
| oofpks_effb2_fixs2.npy | (18000, 9) | float64 |
| oofpks_effb2_fixup2.npy | (18000, 9) | float64 |
| oofpks_effb2_up2s5.npy | (18000, 9) | float64 |
| oofpks_effb3_fix.npy | (18000, 9) | float64 |
| oofpks_stack_v10.npy | (18000, 9) | float64 |
| oofpks_stack_v9i.npy | (18000, 9) | float64 |
| test_effb0_fix.npy | (10000, 9) | float64 |
| test_effb0_fixs2.npy | (10000, 9) | float64 |
| test_effb0_fixup2.npy | (10000, 9) | float64 |
| test_effb0_up2s5.npy | (10000, 9) | float64 |
| test_effb2_fix.npy | (10000, 9) | float64 |
| test_effb2_fix45.npy | (10000, 9) | float64 |
| test_effb2_fixs2.npy | (10000, 9) | float64 |
| test_effb2_fixup2.npy | (10000, 9) | float64 |
| test_effb2_up2s5.npy | (10000, 9) | float64 |
| test_effb3_fix.npy | (10000, 9) | float64 |
| test_stack_v10.npy | (10000, 9) | float64 |
| test_stack_v9i.npy | (10000, 9) | float64 |
| trk_test.npy | (10000, 19) | float32 |
| trk_train.npy | (18000, 19) | float32 |
| trk_trainpks.npy | (18000, 19) | float32 |
| trk_trainpks2.npy | (18000, 19) | float32 |
