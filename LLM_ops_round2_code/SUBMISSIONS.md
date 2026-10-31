# How each upload was produced

| Upload | Command | Validation (PKS-like v2) | Leaderboard |
| --- | --- | --- | --- |
| v6 | `python make_submission.py cache data/test_meta.csv submission_r2_v6.csv` | 0.8995 | 0.9003 |
| v8 | `FEATS=lvl,trk OUT=submission_r2_v8.csv python stacker/stack.py` | 0.9092 | 0.9044 |
| v9 | `INDIV=effb2_fixup2,effb0_fixup2,effb2_fix,effb0_fix,effb2_fixs2,effb0_fixs2,effb2_fix45,effb3_fix FSPLIT=trainpks2 FEATS=lvl,trk OUT=submission_r2_v9.csv python stacker/stack.py` | 0.9162 | — |

PKS-like v2 = Gamma k=12 receiver noise + per-panel gain offsets (std 0.13 of the noise scale), matched to the
panel-to-panel level spread measured on the test cadences. v8's stacker was trained on v1 features (no gain offsets).

v10 (adds notebook 5 = run 13, models effb2_up2s5/effb0_up2s5; validation PKS-like v2 0.9180):
`SPEC=effb2_fixup2:0.2,effb0_fixup2:0.2,effb2_up2s5:0.2,effb0_up2s5:0.2,effb2_fix:0.05,effb0_fix:0.05,effb2_fix45:0.05,effb3_fix:0.05 INDIV=effb2_fixup2,effb0_fixup2,effb2_up2s5,effb0_up2s5,effb2_fix,effb0_fix,effb2_fixs2,effb0_fixs2,effb2_fix45,effb3_fix FSPLIT=trainpks2 FEATS=lvl,trk OUT=submission_r2_v10.csv python stacker/stack.py`
(v9 leaderboard: 0.9158)
