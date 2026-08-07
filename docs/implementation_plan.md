# ORION implementation plan

This repository is now organized around an auditable study-level pipeline rather
than placeholder code. The competition data format can still change, so every
path and column name is configurable; no source file should need editing once
the Kaggle data is attached.

## Completed foundation

- DICOM discovery, geometric slice ordering, rescaling, photometric correction,
  intensity normalization, resizing, uniform selection, and atomic local caching.
- A real `KneeMRIDataset` that reads a metadata CSV and returns `(S, 1, H, W)`
  tensors plus valid-slice masks. Missing weak labels remain `-1`, never `0`.
- Patient-safe grouped and multilabel-aware validation splits.
- MIL image model, optional report fusion, masked losses, OOF metrics,
  calibration, TTA, ensembling, thresholds, and validated submission generation.

## What you need to provide in Kaggle

1. Put the competition directory in `data.root` and set `data.metadata_csv`.
   The CSV must have one of `study_uid`, `study_id`, `StudyInstanceUID`, `uid`,
   or `id`; use `data.image_dir` or a `study_path` column to locate each study.
2. Make target columns match `labels.names` (or override `labels.names` to match
   Kaggle exactly). Expert labels must be `0`/`1`; weak-label abstentions are `-1`.
3. Run preprocessing validation on a small sample, create patient-grouped folds,
   train folds, and save out-of-fold predictions. Fit calibrators and ensemble
   weights only on those OOF predictions.
4. Train final fold models in Kaggle. The repository intentionally does not ship
   weights or run expensive training locally.

## Recommended run order

1. `preprocess.py --config ... --limit 20` to validate paths and DICOM health.
2. `extract_weak_labels.py` for report-only studies; manually review evidence.
3. Train a single image-only MIL baseline, then 5 grouped folds.
4. Add text fusion only after its OOF AUC improves the image baseline.
5. Ensemble OOF-aligned predictions, verify metrics/IDs, then write submission.

## Non-negotiable safeguards

- Never split one patient across folds.
- Never replace an unknown report label with a negative label.
- Never choose thresholds, calibrators, or weights using the leaderboard/test set.
- Always verify a saved submission has unique study IDs, exact label columns, and
  probabilities in `[0, 1]`.
