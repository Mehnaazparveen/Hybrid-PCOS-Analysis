from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
from io import TextIOWrapper
import csv
import hashlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

ZIP_PATH = Path.home() / "Downloads" / "updated test dataset.zip"
TEST_ROOT = Path("datasets/pcosgen_test")
MODEL_PATH = Path("models/pcosgen_mobilenetv2.keras")

THRESHOLD = 0.60
IMG_SIZE = (160, 160)

TEST_ROOT.mkdir(parents=True, exist_ok=True)
(TEST_ROOT / "images").mkdir(exist_ok=True)

# Read labels, audit duplicate contents, and extract test images.
with ZipFile(ZIP_PATH) as archive:
    with archive.open("class label.csv") as file:
        reader = csv.reader(
            TextIOWrapper(file, encoding="utf-8-sig")
        )
        next(reader)

        label_map = {}

        for row in reader:
            filename = Path(row[0].strip()).name
            visibility = row[2].strip()

            if visibility not in {"Visible", "Not-visible"}:
                raise ValueError(
                    f"Invalid label for {filename}: {visibility}"
                )

            if filename in label_map:
                raise ValueError(
                    f"Repeated filename in labels: {filename}"
                )

            label_map[filename] = int(
                visibility == "Visible"
            )

    image_members = sorted(
        name for name in archive.namelist()
        if name.startswith("images/")
        and name.lower().endswith(".jpg")
    )

    names = []
    hashes = defaultdict(list)

    for member in image_members:
        filename = Path(member).name

        if filename not in label_map:
            raise ValueError(f"Missing label: {filename}")

        data = archive.read(member)
        digest = hashlib.sha256(data).hexdigest()

        hashes[digest].append(
            (filename, label_map[filename])
        )

        (TEST_ROOT / "images" / filename).write_bytes(data)
        names.append(filename)

if len(names) != len(label_map):
    raise ValueError("Test images and labels do not match.")

# Identify exact duplicate groups and label conflicts.
duplicates = [
    items for items in hashes.values()
    if len(items) > 1
]

conflicting_hashes = {
    digest
    for digest, items in hashes.items()
    if len({label for _, label in items}) > 1
}

# Keep one representative per nonconflicting image content.
keep_names = {
    min(items, key=lambda x: x[0])[0]
    for digest, items in hashes.items()
    if digest not in conflicting_hashes
}

print("\n--- TEST DATASET AUDIT ---")
print("Original test images:", len(names))
print("Exact duplicate groups:", len(duplicates))
print("Conflicting-label groups:", len(conflicting_hashes))
print("Clean unique test images:", len(keep_names))

# Predict in the same order as the test filenames.
paths = [
    str((TEST_ROOT / "images" / name).resolve())
    for name in names
]

labels = np.array(
    [label_map[name] for name in names],
    dtype=int
)

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image

dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

print("\nLoading saved MobileNetV2...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Predicting test images...")
probabilities = model.predict(
    dataset, verbose=1
).ravel()

def evaluate(title, actual, scores):
    predicted = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    print(f"\n========== {title} ==========")
    print("Images:", len(actual))
    print("Threshold:", THRESHOLD)

    print(
        "Accuracy:",
        round(accuracy_score(actual, predicted), 4)
    )
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4)
    )
    print(
        "PCOM precision:",
        round(precision_score(actual, predicted, zero_division=0), 4)
    )
    print(
        "PCOM recall:",
        round(recall_score(actual, predicted, zero_division=0), 4)
    )
    print(
        "PCOM F1:",
        round(f1_score(actual, predicted, zero_division=0), 4)
    )
    print(
        "ROC-AUC:",
        round(roc_auc_score(actual, scores), 4)
    )
    print(
        "Average precision:",
        round(average_precision_score(actual, scores), 4)
    )

    print("\nConfusion matrix:")
    print([[int(tn), int(fp)], [int(fn), int(tp)]])

    print(
        "Majority-class baseline accuracy:",
        round(max(np.mean(actual), 1 - np.mean(actual)), 4)
    )

# Report original test results.
evaluate("ORIGINAL TEST SET", labels, probabilities)

# Report sensitivity analysis after duplicate cleanup.
indices = np.array([
    i for i, name in enumerate(names)
    if name in keep_names
], dtype=int)

evaluate(
    "DEDUPLICATED TEST SET",
    labels[indices],
    probabilities[indices]
)

print("\nEvaluation completed.")
print("No model training or threshold tuning performed.")