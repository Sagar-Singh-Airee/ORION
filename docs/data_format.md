# Data contract

ORION expects one row per MRI study in a CSV. Required fields are one study ID
column—`study_uid`, `study_id`, `StudyInstanceUID`, `uid`, or `id`—and the 12
configured label columns when labels are available.

The default relative layout is:

```text
<data.root>/
  train.csv
  train_images/<study_uid>/**/<dicom files>
  test.csv
  test_images/<study_uid>/**/<dicom files>
```

Use a `study_path` column for any different layout. `data.image_dir` and
`data.test_image_dir` are configurable alternatives.

Expert labels are `0` or `1`. Use `-1` only for weak-label abstention/missing
labels; the losses and metrics intentionally exclude these cells. For cross
validation, add `patient_id` and generate a `fold` column with
`scripts/cross_validate.py` before running `scripts/train.py`.
