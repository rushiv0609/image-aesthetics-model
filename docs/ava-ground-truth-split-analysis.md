# AVA ground-truth and split analysis

Manifest rows: **255,508**

## Findings

- Mean rating: 5.3833 ± 0.7313; range 1.809–8.600.
- Vote entropy: 2.477 bits mean; vote totals median 201.
- Challenge groups: 1,398; size range 7–1,108; largest contiguous annotation run 1,108.
- Exact duplicate hashing examined 178,551 size-collision candidates (23.16 GB); found 8 exact groups (8 crossing challenges).
- Annotation order has lag-1 score correlation 0.0903; no capture chronology or photographer identity is present.

## Candidate split comparison

| Split | Rows | Groups | Mean | Std | Decile L1 error |
|:--|--:|--:|--:|--:|--:|
| naive_random_stratified / train | 204,408 | 204,408 | 5.3832 | 0.7318 | 0.0000 |
| naive_random_stratified / validation | 25,550 | 25,550 | 5.3831 | 0.7274 | 0.0000 |
| naive_random_stratified / test | 25,550 | 25,550 | 5.3843 | 0.7315 | 0.0000 |
| challenge_group_greedy_seed_20260920 / train | 204,408 | 1,275 | 5.3826 | 0.7360 | 0.0070 |
| challenge_group_greedy_seed_20260920 / validation | 25,550 | 62 | 5.3830 | 0.7172 | 0.0232 |
| challenge_group_greedy_seed_20260920 / test | 25,550 | 61 | 5.3891 | 0.7072 | 0.0383 |
| challenge_plus_exact_duplicate_group_greedy_seed_20260920 / train | 204,411 | 1,267 | 5.3831 | 0.7352 | 0.0053 |
| challenge_plus_exact_duplicate_group_greedy_seed_20260920 / validation | 25,549 | 59 | 5.3861 | 0.7160 | 0.0216 |
| challenge_plus_exact_duplicate_group_greedy_seed_20260920 / test | 25,548 | 60 | 5.3825 | 0.7151 | 0.0245 |

## Recommendation

Use the challenge-disjoint greedy grouped split with seed `20260920`, unioning exact duplicate files into the same component. Keep 80% development train, 10% development validation, and 10% locked test. Run five-fold GroupKFold only inside the 90% development portion; never use locked-test labels during selection.

The available `challenge_id` is a leakage-control proxy, not a photographer identifier. AVA metadata does not expose photographer/user identity in this manifest. Challenge IDs are nearly contiguous in AVA.txt (99.45% of adjacent rows share a challenge), so a naive random split can place visually/semantically related challenge material in every partition. Challenge grouping reduces that risk, but cannot prove photographer-level independence because photographer identity is unavailable.
