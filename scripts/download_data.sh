#!/usr/bin/env bash
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token)
# Verify each dataset slug on its Kaggle page (the part after kaggle.com/datasets/).
set -e
kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset \
  -p ml/datasets/sleep/data --unzip
# Add the other three the same way once you have their slugs, e.g.:
# kaggle datasets download -d <owner>/<student-depression-slug> -p ml/datasets/depression/data --unzip
