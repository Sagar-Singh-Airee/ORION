# Architecture

Each study is represented as a variable-length bag of DICOM slices. The reader
orders slices by physical position, preprocesses the selected series, and the
dataset pads only for batching while retaining a valid-slice mask.

```text
DICOM study → series selection → normalization/resize → slice encoder
           → masked MIL attention → multi-label logits → probabilities
```

The image-only path is the required baseline. Report fusion is optional and is
enabled only when `data.multimodal=true`; it cannot silently load an NLP model
for a vision-only experiment. Weak report labels use `-1` for abstention, which
is ignored by both masked losses and metrics.

The inference path applies optional TTA, then performs OOF-only calibration and
ensemble aggregation. Submission writing validates IDs, columns, finiteness,
and `[0, 1]` probability bounds.
