import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path("datasets/pcosgen_train")
TEST_DIR = Path("datasets/pcosgen_test/images")

OVERLAP_CSV = (
    ROOT / "splits/high_confidence_test_overlaps.csv"
)

TEST_ZIP = (
    Path.home() / "Downloads/updated test dataset.zip"
)

MODEL_PATH = Path("models/pcosgen_mobilenetv2.keras")

THRESHOLD = 0.60

# --------------------------------
# Load official test labels
# --------------------------------

with ZipFile(TEST_ZIP) as archive:
    with archive.open("class label.csv") as file:
        reader = csv.reader(
            (line.decode("utf-8-sig") for line in file)
        )

        next(reader)

        labels = {}

        for row in reader:
            filename = Path(row[0].strip()).name
            visibility = row[2].strip()

            if visibility not in ("Visible", "Not-visible"):
                raise ValueError(
                    f"Invalid label: {filename}"
                )

            if filename in labels:
                raise ValueError(
                    f"Duplicate label record: {filename}"
                )

            labels[filename] = int(
                visibility == "Visible"
            )

# --------------------------------
# Load known development/test overlaps
# --------------------------------

with OVERLAP_CSV.open(
    newline="", encoding="utf-8"
) as file:
    overlap_names = {
        row["test_image"]
        for row in csv.DictReader(file)
    }

print("Known overlapping test filenames:", len(overlap_names))

# --------------------------------
# Group exact duplicates
# --------------------------------

hash_groups = defaultdict(list)

for filename in sorted(labels):
    path = TEST_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    hash_groups[digest].append(filename)

kept = []
conflicting_groups = 0
overlapping_groups = 0

for filenames in hash_groups.values():
    group_labels = {labels[name] for name in filenames}

    # Exclude contradictory labels.
    if len(group_labels) > 1:
        conflicting_groups += 1
        continue

    # Exclude all identical copies if any member overlaps.
    if any(name in overlap_names for name in filenames):
        overlapping_groups += 1
        continue

    # Retain one representative.
    kept.append(min(filenames))

kept.sort()

print("\n--- SCREENED TEST DATASET ---")
print("Original test images:", len(labels))
print("Conflicting-label groups excluded:", conflicting_groups)
print("Groups excluded for development overlap:", overlapping_groups)
print("Images retained:", len(kept))

# --------------------------------
# Prepare test images
# --------------------------------

paths = [
    str((TEST_DIR / name).resolve())
    for name in kept
]

actual = np.array(
    [labels[name] for name in kept],
    dtype=int
)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(
        image, channels=3
    )
    image = tf.image.resize(
        image, (160, 160)
    )
    return image

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)

# --------------------------------
# Evaluate saved model
# --------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

scores = model.predict(ds, verbose=1).ravel()
predicted = (scores >= THRESHOLD).astype(int)

print("\n========== SCREENED TEST RESULTS ==========")

print("Threshold:", THRESHOLD)
print("Images:", len(actual))
print("PCOM visible:", int(actual.sum()))
print("PCOM not visible:", int(len(actual) - actual.sum()))

print("Accuracy:", round(
    accuracy_score(actual, predicted), 4
))

print("Balanced accuracy:", round(
    balanced_accuracy_score(actual, predicted), 4
))

print("PCOM precision:", round(
    precision_score(actual, predicted, zero_division=0), 4
))

print("PCOM recall:", round(
    recall_score(actual, predicted, zero_division=0), 4
))

print("PCOM F1:", round(
    f1_score(actual, predicted, zero_division=0), 4
))

print("ROC-AUC:", round(
    roc_auc_score(actual, scores), 4
))

print("Average precision:", round(
    average_precision_score(actual, scores), 4
))

print("\nConfusion matrix:")
print(confusion_matrix(
    actual, predicted, labels=[0, 1]
))

print("\nScreened evaluation completed.")
print("Original models and images unchanged.")