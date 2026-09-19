# AVA materialized split audit

Algorithm: `ava-group-stratified-v1`; seed: `20260920`.

The full manifest contains 255,508 rows; the completed embedding ledger contributes 255,502. The remaining 6 rows are excluded from all partitions.

| Partition | Rows | Groups | Mean | Std | Decile counts |
|:--|--:|--:|--:|--:|:--|
| train | 204,271 | 1258 | 5.3820 | 0.7348 | `[20525, 20625, 20555, 20488, 20350, 20417, 20334, 20121, 20239, 20617]` |
| validation | 25,619 | 64 | 5.3907 | 0.7139 | `[2500, 2448, 2486, 2551, 2566, 2601, 2603, 2731, 2668, 2465]` |
| test (locked) | 25,612 | 64 | 5.3862 | 0.7202 | `[2526, 2477, 2509, 2511, 2634, 2533, 2613, 2698, 2643, 2468]` |

The locked test requires explicit opt-in and is not available to normal training or hyperparameter-selection code. Five group-disjoint CV fold assignments are materialized only over the development rows.

Dataset/cache/model/split hashes and aggregate CV statistics are in the external `split-metadata.json` and repository JSON audit.
