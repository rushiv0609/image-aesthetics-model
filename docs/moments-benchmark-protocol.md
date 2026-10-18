# Moments Generator Benchmark Protocol

This is the product release gate for the final fusion scorer. Media and absolute
paths remain outside Git. The versioned manifest uses opaque item and event IDs;
the repository may contain only the protocol, scripts, and privacy-safe aggregate
reports.

Human evidence is never synthesized. A benchmark without annotations is reported
as pending and cannot satisfy the release gate.

Required blinded annotations are within-event pairwise preference, keep/reject
labels for severe failures, and event-level top-k selections. Failure tags cover
blur, low light, transition frames, awkward expression, duplicate burst,
screenshot/meme, monochrome, and intentional motion. Frames from one video,
burst, or event must share an event ID.

Predictions are generated only after the architecture, feature contract,
calibration choice, and checkpoint policy are frozen. Each candidate must provide
one `[0,1]` score for every manifest item. The evaluator reports pairwise accuracy,
top-1%/5% event-local selection, overlap with human top-k choices, and severe false
positive rate. F5-G must improve human top-k selection and must not increase severe
false positives versus F0, F1, and F4 before production promotion.

The evaluator can be run before ratings arrive to validate schemas, but null human
metrics are explicitly non-evidence. This separates automatable benchmark
preparation from the real-rater work that remains inherently human.

`scripts/evaluation/prepare_moments_benchmark.py` creates the private external
manifest, a mode-0600 ID salt, an empty annotation stream, and a privacy-safe
summary. It groups chronologically adjacent media into events using a fixed six-hour
gap (configurable before freeze), preferring EXIF capture time and falling back to
file modification time. Only the protocol and aggregate summary may be committed;
the salt, paths, media, and annotations remain external.

`scripts/evaluation/materialize_moments_feature_media.py` then creates one
immutable image-like feature source per item in a new external directory.
JPEG-like images are hard-linked without pixel changes, HEIC images are
converted with macOS `sips` at JPEG quality 95, and videos use one ffmpeg frame
at the exact temporal midpoint. This label-independent choice is recorded in
`materialization-report.json`; changing it requires a new benchmark version
rather than overwriting evidence. The derived manifest is processed by the same
ten-signal cache providers used for training data.
