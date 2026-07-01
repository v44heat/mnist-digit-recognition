# ============================================================
#  predict.py — Predict a digit from your own image
#  Requires: mnist_model.keras (run train.py first)
#  Usage:    python predict.py path/to/your_digit.png


import sys
import os
import numpy as np
import matplotlib
is_headless = (
    sys.platform.startswith("linux")
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
)
if is_headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from scipy.ndimage import zoom as nd_zoom

def load_model():
    try:
        return keras.models.load_model("mnist_model.keras")
    except Exception:
        print("ERROR: mnist_model.keras not found.")
        print("Please run  python train.py  first to train and save the model.")
        sys.exit(1)


def preprocess_image(image_path: str):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img).astype("float32") / 255.0

    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array
        print("  Note: image inverted (white background detected).")

    # ── NEW: Normalize stroke thickness ──────────────────────
    from skimage.morphology import skeletonize, dilation, disk
    binary = img_array > 0.3
    skeleton = skeletonize(binary)
    # Re-dilate slightly to give the model something to work with
    img_array = dilation(skeleton.astype("float32"), disk(2))
    # ─────────────────────────────────────────────────────────

    # Auto-crop (keep your existing code)
    threshold = 0.1
    rows = np.any(img_array > threshold, axis=1)
    cols = np.any(img_array > threshold, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(img_array.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(img_array.shape[1], cmax + pad)
        img_array = img_array[rmin:rmax, cmin:cmax]

    img_pil = Image.fromarray((img_array * 255).astype("uint8"))
    img_pil = img_pil.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_pil).astype("float32") / 255.0

    return img_array, img_array.reshape(1, 28, 28, 1)


def predict(image_path: str):
    model = load_model()

    print(f"\nProcessing: {image_path}")
    img_array, img_flat = preprocess_image(image_path)

    # Run prediction
    probs  = model.predict(img_flat, verbose=0)[0]
    digit  = int(np.argmax(probs))
    conf   = probs[digit] * 100

    # Print results to terminal
    print(f"\n  Predicted digit : {digit}")
    print(f"  Confidence      : {conf:.1f} %")
    print("\n  All probabilities:")
    for d, p in enumerate(probs):
        bar    = "X" * int(p * 40)
        marker = " <- predicted" if d == digit else ""
        print(f"    {d}: {bar:<40} {p * 100:5.1f} %{marker}")

    # Save result chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    ax1.imshow(img_array, cmap="gray")
    ax1.set_title(f"Processed image (28x28)\nPredicted: {digit}  ({conf:.1f}% confident)")
    ax1.axis("off")

    colors = ["steelblue"] * 10
    colors[digit] = "coral"
    ax2.bar(range(10), probs, color=colors)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit (0-9)")
    ax2.set_ylabel("Probability")
    ax2.set_title("Confidence per digit class")

    plt.tight_layout()
    out_path = "prediction_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Chart saved -> {out_path}")
    plt.show()
    


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your_digit.png")
        print("Example: python predict.py samples/eight.png")
        sys.exit(1)
    predict(sys.argv[1])
