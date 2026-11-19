from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# -------------------------------
# Configuration
# -------------------------------

tf.keras.utils.set_random_seed(42)

ROOT = Path("datasets/pcosgen_train")
SPLITS = ROOT / "splits"

MODEL_PATH = Path("models/pcosgen_mobilenetv2.keras")

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 12


# -------------------------------
# Read the dataset manifests
# -------------------------------

def load_manifest(filename):
    paths = []
    labels = []

    with (SPLITS / filename).open(
        newline="", encoding="utf-8"
    ) as file:

        for row in csv.DictReader(file):
            image_path = ROOT / "images" / row["filename"]

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            paths.append(str(image_path.resolve()))
            labels.append(int(row["pcom_visible"]))

    return paths, np.array(labels, dtype=np.int32)


train_paths, train_labels = load_manifest("train_manifest.csv")
val_paths, val_labels = load_manifest("val_manifest.csv")

print("\n--- DATASET ---")
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Training PCOM-visible:", int(train_labels.sum()))
print("Validation PCOM-visible:", int(val_labels.sum()))


# -------------------------------
# Image loading
# -------------------------------

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, IMG_SIZE)

    # Keep pixel values in 0–255.
    # The model performs its own preprocessing.
    return image, tf.cast(label, tf.float32)


def create_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=42,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = create_dataset(
    train_paths, train_labels, training=True
)

val_ds = create_dataset(
    val_paths, val_labels, training=False
)


# -------------------------------
# Handle class imbalance
# -------------------------------

n_total = len(train_labels)
n_positive = int(train_labels.sum())
n_negative = n_total - n_positive

class_weights = {
    0: n_total / (2 * n_negative),
    1: n_total / (2 * n_positive),
}

print("\nClass weights:", class_weights)


# -------------------------------
# MobileNetV2 transfer learning
# -------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    alpha=0.35,
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))

# Convert 0–255 pixels into MobileNetV2's -1 to +1 range.
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
)(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
    ],
)

model.summary()


# -------------------------------
# Training
# -------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
]

print("\n--- STARTING TRAINING ---")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)


# -------------------------------
# Save the new model
# -------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

print("\nModel saved:", MODEL_PATH)


# -------------------------------
# Validation evaluation
# -------------------------------

probabilities = model.predict(
    val_ds, verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

print("\n========== VALIDATION RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(val_labels, predictions), 4)
)

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(val_labels, predictions), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(val_labels, probabilities), 4)
)

print(
    "Average precision (PR-AUC summary):",
    round(average_precision_score(val_labels, probabilities), 4)
)

print("\nConfusion matrix:")
print(confusion_matrix(val_labels, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        val_labels,
        predictions,
        labels=[0, 1],
        target_names=["PCOM Not Visible", "PCOM Visible"],
        zero_division=0,
    )
)

print("\nTraining and evaluation completed!")