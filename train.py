# ============================================================
#  train.py — Train the MNIST digit recognition model
#  Run this ONCE to train and save the model.
#  Usage: python train.py


import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load MNIST dataset ────────────────────────────────────
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")

# ── 2. Preprocess ────────────────────────────────────────────
# Reshape to (N, 28, 28, 1) for CNN — keep spatial structure, normalize to [0,1]
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32")  / 255.0
print("  Preprocessing complete.")

# ── 3. Build the neural network ──────────────────────────────

model = keras.Sequential([
   
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),

    # ── CNN feature extractor ────────────────────────────────
    # Block 1: detect basic edges and strokes
    layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 2: detect higher-level shapes (curves, corners, loops)
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # ── Dense classifier head ────────────────────────────────
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.summary()

# ── 4. Compile ───────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 5. Callbacks ─────────────────────────────────────────────
callbacks = [
    # Stop early if validation accuracy stops improving — avoids wasted time
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    # Reduce learning rate when training plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── 6. Train ─────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=30,               # EarlyStopping will cut this short when needed
    batch_size=128,          # larger batch = more stable CNN gradients
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# ── 7. Evaluate ──────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy : {test_acc * 100:.2f} %")
print(f"Test loss     : {test_loss:.4f}")

# ── 8. Save training history chart ───────────────────────────
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

# ── 9. Save the model ────────────────────────────────────────
model.save("mnist_model.keras")
print("Model saved → mnist_model.keras")
print("\nDone! Now run:  python predict.py your_image.png")
