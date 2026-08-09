# Kaggle runbook

The repository does not train locally. Run the following only in a Kaggle GPU
notebook after attaching the competition dataset and copying/cloning this project
to a writable directory.

## 1. Install and map the competition data

```bash
pip install -e ".[dev]"
```

Set `data.root` to the dataset root. The default code expects
`train.csv`, `test.csv`, `train_images/<study_id>/...`, and
`test_images/<study_id>/...`; use a `study_path` CSV column or override
`data.image_dir` if the final competition layout differs. Read
[`data_format.md`](data_format.md) before changing a path.

## 2. Verify raw studies before allocating GPU time

```bash
python scripts/preprocess.py \
  --config configs/experiment/baseline_resnet50.yaml \
  --data-dir /kaggle/input/<competition>/train_images \
  --limit 20 --output /kaggle/working/preprocess_manifest.json
```

Inspect failures in the manifest. The data path uses header-only ranking of series,
then decodes only the selected sequence; it does not construct random fallback
images. Fix all missing-path and unreadable-series errors before training.

## 3. Create immutable patient-grouped folds

```bash
python scripts/cross_validate.py \
  --metadata /kaggle/input/<competition>/train.csv \
  --group-column patient_id \
  --n-folds 5 \
  --output /kaggle/working/folds.csv
```

`folds.csv` must contain all target labels plus `patient_id` and `fold`. The
dataset selects `fold != data.splits.fold` for training and `fold == ...` for
validation. This prevents the same patient reaching both sides of a fold.

## 4. Train one baseline fold

```bash
python scripts/train.py \
  --config configs/experiment/baseline_resnet50.yaml \
  data.root=/kaggle/input/<competition> \
  data.metadata_csv=/kaggle/working/folds.csv \
  data.splits.fold=0 \
  project.output_dir=/kaggle/working/orion_runs
```

The script writes the resolved config, `last.pth`, `best_model.pth`, and
`history.json` under the experiment output directory. It uses the real
`KneeMRIDataset`, mixed precision only on CUDA, gradient accumulation,
optional EMA, masked labels, and early stopping. Repeat with fold `1` through
`4`; do not combine validation rows from a checkpoint trained on that fold.

## 5. Produce and evaluate out-of-fold predictions

For each fold, run `scripts/predict.py` against its validation CSV/checkpoint,
then concatenate results by `study_id` in the exact fold order. Evaluate only
after every study appears once:

```bash
python scripts/evaluate.py \
  --targets /kaggle/working/folds.csv \
  --predictions /kaggle/working/oof.csv \
  --id-column study_id \
  --output /kaggle/working/oof_metrics.json
```

Inspect macro AUC, each label AUC, calibration, scanner/site slices, and the
highest-error studies. A label with only one observed class in a fold is reported
as undefined and excluded from that fold's macro AUC.

## 6. Train final fold models and predict test studies

Use each saved fold checkpoint with the matching config:

```bash
python scripts/predict.py \
  --config configs/experiment/baseline_resnet50.yaml \
  --checkpoint /kaggle/working/orion_runs/<experiment>/best_model.pth \
  --data-root /kaggle/input/<competition> \
  --metadata-csv /kaggle/input/<competition>/test.csv \
  --output /kaggle/working/fold0_test.csv
```

The predictor uses CUDA autocast when available and only materializes one TTA
logit accumulator, avoiding an unnecessary stack of full prediction tensors.

## 7. Ensemble and calibrate using OOF evidence only

```bash
python scripts/ensemble.py \
  --predictions /kaggle/working/fold0_test.csv /kaggle/working/fold1_test.csv \
  --method rank_mean --output /kaggle/working/test_ensemble.csv

python scripts/calibrate.py \
  --targets /kaggle/working/folds.csv --oof /kaggle/working/oof.csv \
  --test /kaggle/working/test_ensemble.csv \
  --output /kaggle/working/test_calibrated.csv
```

The calibration command fits temperature values on OOF labels/predictions and
applies them to the test file. Never fit it, tune thresholds, or choose weights
using leaderboard feedback.

## 8. Validate the final submission

```bash
python scripts/submit.py \
  --predictions /kaggle/working/test_calibrated.csv \
  --sample-submission /kaggle/input/<competition>/sample_submission.csv \
  --output /kaggle/working/submission.csv
```

The command rejects duplicated study IDs, unexpected shapes, NaNs, and values
outside `[0, 1]`. Upload only the generated `submission.csv`.
