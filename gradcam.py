from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/pcosgen_mobilenetv2.keras")
SAMPLE_PATH = Path(
    "datasets/pcosgen_train/images/100image65.jpg"
)


def generate_gradcam(image, model, image_bytes=None):
    # Convert ultrasound to RGB.
    original = image.convert("RGB")
    if image_bytes is None:
        image_bytes = SAMPLE_PATH.read_bytes()

    decoded = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    resized = tf.image.resize(decoded, (160, 160))
    batch = tf.expand_dims(resized, axis=0)

    # Access the trained MobileNetV2 components.
    rescaling = model.get_layer("rescaling")
    backbone = model.get_layer("mobilenetv2_0.35_160")
    pooling = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("dense")

    with tf.GradientTape() as tape:
        x = rescaling(batch)

        feature_maps = backbone(x, training=False)
        tape.watch(feature_maps)

        x = pooling(feature_maps)
        x = dropout(x, training=False)
        predictions = classifier(x)

        # Explain the PCOM-visible score.
        target_score = predictions[:, 0]

    gradients = tape.gradient(target_score, feature_maps)

    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(
        gradients, axis=(1, 2), keepdims=True
    )

    heatmap = tf.reduce_sum(
        weights * feature_maps, axis=-1
    )[0]

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = heatmap.numpy()

    # Resize heatmap to original ultrasound dimensions.
    heatmap = cv2.resize(
        heatmap,
        original.size,
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(heatmap, 0, 1)

    # Convert heatmap into a coloured visualization.
    coloured = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    coloured = cv2.cvtColor(
        coloured, cv2.COLOR_BGR2RGB
    )

    coloured_image = Image.fromarray(coloured)

    # Overlay heatmap on the ultrasound.
    overlay = Image.blend(
        original, coloured_image, alpha=0.35
    )

    return overlay, float(target_score.numpy()[0])


if __name__ == "__main__":
    print("Loading MobileNetV2...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    image = Image.open(SAMPLE_PATH)

    overlay, score = generate_gradcam(image, model)

    output = Path("gradcam_preview.png")
    overlay.save(output)

    print("\nGrad-CAM generated successfully!")
    print("PCOM-visible model score:", round(score, 3))
    print("Saved image:", output.resolve())
