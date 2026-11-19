"""CNN for experimental ovarian ultrasound classification."""

from pathlib import Path
import numpy as np

IMG_SIZE = (128, 128)
MODEL_PATH = Path("models/cnn_model.keras")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(*IMG_SIZE, 1))

    # Scale pixel values identically for training and prediction.
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Mild augmentation helps with a small training dataset.
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.03)(x)
    x = layers.RandomZoom(0.05)(x)

    for filters in [16, 32, 64]:
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    # Fewer parameters than Flatten() to reduce overfitting.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def preprocess_image(pil_or_path):
    """Resize and convert to grayscale; return raw 0–255 pixels."""
    import cv2
    from PIL import Image

    if isinstance(pil_or_path, (str, Path)):
        img = cv2.imread(str(pil_or_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {pil_or_path}")
    elif isinstance(pil_or_path, Image.Image):
        img = np.array(pil_or_path.convert("L"))
    else:
        raise TypeError("Expected an image path or PIL image.")

    img = cv2.resize(
        img, IMG_SIZE, interpolation=cv2.INTER_LINEAR
    )
    img = img.astype("float32")

    # Model contains the Rescaling layer, so don't normalize here.
    return img.reshape(1, *IMG_SIZE, 1)


def train(epochs=60, batch=16):
    import tensorflow as tf
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
    )

    train_dir = Path("cnn_dataset/train")
    val_dir = Path("cnn_dataset/val")

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Expected cnn_dataset/train and cnn_dataset/val folders."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch,
        label_mode="binary",
        shuffle=False,
    )

    print("Class mapping:", dict(enumerate(train_ds.class_names)))

    # Do not silently train with reversed class labels.
    if train_ds.class_names != ["non_pcos", "pcos"]:
        raise ValueError(
            "Expected folders ordered as ['non_pcos', 'pcos']; "
            f"found {train_ds.class_names}"
        )

    def count_images(folder):
        return sum(
            1
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    n0 = count_images(train_dir / "non_pcos")
    n1 = count_images(train_dir / "pcos")
    total = n0 + n1

    if n0 == 0 or n1 == 0:
        raise ValueError("Both training classes must contain images.")

    # Equalize the total contribution from each class.
    class_weights = {
        0: total / (2.0 * n0),
        1: total / (2.0 * n1),
    }

    print(f"Training images: non_pcos={n0}, pcos={n1}")
    print("Class weights:", class_weights)

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            mode="min",
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Save the best weights restored by EarlyStopping.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Evaluate both classes, not accuracy alone.
    y_true = np.concatenate([
        labels.numpy().reshape(-1)
        for _, labels in val_ds
    ]).astype(int)

    probabilities = model.predict(val_ds, verbose=0).ravel()
    y_pred = (probabilities >= 0.5).astype(int)

    print(
        "\nValidation accuracy:",
        round(accuracy_score(y_true, y_pred), 3),
    )
    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-PCOS", "PCOS"],
        zero_division=0,
    ))


def predict(pil_image) -> dict:
    """Return the model's class and sigmoid score."""
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Ultrasound CNN has not been trained. Run python cnn_model.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    arr = preprocess_image(pil_image)
    probability = float(model.predict(arr, verbose=0)[0, 0])

    return {
        "label": "PCOS" if probability >= 0.5 else "Non-PCOS",
        "probability": probability,
    }


if __name__ == "__main__":
    train()