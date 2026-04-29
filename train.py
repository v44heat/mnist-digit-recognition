# ============================================================
#  train.py — Train the MNIST digit recognition model
#  Run this ONCE to train and save the model.
#  Usage: python train.py
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed during training
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load MNIST dataset ────────────────────────────────────
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")

# ── 2. Preprocess ────────────────────────────────────────────
# Flatten 28x28 images → 784-element vectors, normalize to [0,1]
X_train_flat = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test_flat  = X_test.reshape(-1, 784).astype("float32")  / 255.0
print("  Preprocessing complete.")

# ── 3. Build the neural network ──────────────────────────────
# Input (784) → Hidden (128, ReLU) → Hidden (64, ReLU) → Output (10, Softmax)
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64,  activation="relu"),
    layers.Dense(10,  activation="softmax"),
])
model.summary()

# ── 4. Compile ───────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 5. Train ─────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train_flat, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# ── 6. Evaluate ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc * 100:.2f} %")
print(f"Test loss     : {test_loss:.4f}")

# ── 7. Save training history chart ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Training chart saved → training_history.png")

# ── 8. Save the model ────────────────────────────────────────
model.save("mnist_model.keras")
print("Model saved → mnist_model.keras")
print("\nDone! Now run:  python predict.py your_image.png")
