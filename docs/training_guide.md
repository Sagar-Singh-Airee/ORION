# Kaggle runbook

No model is trained locally. In Kaggle, install the project, attach the data,
then set `data.root` to the mounted competition directory.

```bash
# 1. Confirm DICOM readability before spending GPU time.
python scripts/preprocess.py --config configs/experiment/baseline_resnet50.yaml \
  --data-dir /kaggle/input/<competition>/train_images --limit 20

# 2. Build patient-safe folds once and keep the generated CSV immutable.
python scripts/cross_validate.py --metadata /kaggle/input/<competition>/train.csv \
  --group-column patient_id --output /kaggle/working/folds.csv

# 3. Train one fold. The generated config and checkpoints go under project.output_dir.
python scripts/train.py --config configs/experiment/baseline_resnet50.yaml \
  data.metadata_csv=/kaggle/working/folds.csv data.splits.fold=0

# 4. Evaluate OOF predictions and only then ensemble/calibrate.
python scripts/evaluate.py --targets /kaggle/working/folds.csv --predictions oof.csv
```

Never choose ensemble weights, thresholds, calibrators, or architecture changes from
the test leaderboard. Use the held-out patient folds and preserve their IDs/order.

For prediction, point `scripts/predict.py` at a saved checkpoint and test CSV.
`scripts/submit.py` checks unique IDs and probability bounds before writing the
Kaggle upload file.
