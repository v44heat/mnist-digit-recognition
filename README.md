# MNIST Handwritten Digit Recognition

A beginner-friendly convolutional neural network (CNN) that recognises handwritten digits (0–9) using the MNIST dataset and TensorFlow/Keras. Achieves ~**99%+ accuracy** on the test set.

---

## Demo

| Your image | Processed (28×28) | Prediction |
|:-----------:|:-----------------:|:----------:|
| ✍️ Draw a digit | Auto-cropped & resized | ✅ Predicted with confidence % |

---

## Project Structure

```
mnist-digit-recognition/
│
├── train.py          # Train the neural network (run once)
├── predict.py        # Predict a digit from your own image
├── requirements.txt  # Python dependencies
├── samples/          # Example digit images to test with
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/v44heat/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### 2. Create a virtual environment & install dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This will:
- Download the MNIST dataset automatically (~11 MB)
- Train up to 30 epochs with early stopping (typically stops at ~15, takes 5–20 minutes on CPU)
- Save the model as `mnist_model.keras`
- Save a training chart as `training_history.png`

Expected output:
```
Test accuracy : 99.10 %
Test loss     : 0.0312
Model saved → mnist_model.keras
```

### 4. Predict your own digit

Draw a digit (0–9) in Paint or any image editor and save it as a PNG, then run:

```bash
python predict.py your_digit.png
```

Expected output:
```
Predicted digit : 7
Confidence      : 98.6 %
Chart saved → prediction_result.png
```

---

## Neural Network Architecture

The model uses a **Convolutional Neural Network (CNN)** which reads images spatially rather than as a flat list of pixels, making it far more accurate on real-world drawn digits.

```
Input (28×28×1)
   │
   ├── RandomRotation / RandomZoom / RandomTranslation  ← data augmentation
   │
   ├── Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
   │                                                  ↑ detects edges & strokes
   ├── Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
   │                                                  ↑ detects curves & shapes
   ├── Flatten
   ├── Dense(256) → BatchNorm → Dropout(0.5)
   └── Dense(10, Softmax)  ← probability per digit 0–9
```

---

## How It Works

1. **Load** — MNIST dataset: 60,000 training + 10,000 test images
2. **Preprocess** — Reshape to 28×28×1, normalize pixel values to [0, 1]
3. **Augment** — Random rotation, zoom, and translation simulate real drawing variation
4. **Train** — Backpropagation with Adam optimizer; EarlyStopping and ReduceLROnPlateau callbacks prevent overfitting
5. **Predict** — Your image is stroke-normalized, auto-cropped, resized to 28×28, and fed to the model

---

## Tips for Best Prediction Results

| Do this ✅ | Avoid this ❌ |
|---|---|
| Use a canvas between 200×200 and 500×500 px | Full-screen canvas with a tiny digit |
| Draw with a thin-to-medium brush (8–15 px on a 200px canvas) | Very thick strokes that blur when scaled down |
| Center the digit, filling ~60–70% of the canvas | Digit touching the edges |
| Save as **PNG** | JPEG (compression artifacts corrupt pixel values) |
| Draw one clean stroke per part of the digit | Sketchy overlapping lines |
| Use solid black on white background | Low contrast or coloured backgrounds |

The script automatically:
- Detects white-on-black vs black-on-white and inverts if needed
- Normalizes stroke thickness using skeletonization
- Crops out empty borders around the digit
- Resizes to the required 28×28 pixels

---

## Requirements

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | ≥ 2.11 | Neural network framework |
| `numpy` | ≥ 1.23 | Array operations |
| `matplotlib` | ≥ 3.6 | Plotting charts |
| `pillow` | ≥ 9.3 | Image loading & processing |
| `scikit-image` | ≥ 0.19 | Stroke normalization (skeletonize & dilation) |

---

## Results

| Metric | Old (Dense) | New (CNN) |
|---|---|---|
| Training accuracy | ~99.3% | ~99.8% |
| Validation accuracy | ~98.1% | ~99.3% |
| **Test accuracy** | **~97.7%** | **~99.1%** |
| Paint-drawn digit accuracy | Poor | Good |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'matplotlib'`**
→ Your virtual environment is not activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) first.

**`File not found: mnist_model.keras`**
→ You need to train the model first. Run `python train.py`.

**`ValueError: Expected shape (None, 28, 28, 1), but input has incompatible shape (1, 784)`**
→ You have a old model saved from the previous dense architecture. Re-run `python train.py` to generate a new compatible model.

**Low confidence / wrong prediction**
→ Check the drawing tips above. The most common causes are a brush that is too thick, a digit that is too small on the canvas, or saving as JPG instead of PNG.

**CUDA / GPU warnings in terminal**
→ These are harmless. TensorFlow will fall back to CPU automatically if no GPU drivers are found.

**Matplotlib window not appearing**
→ The script auto-detects headless Linux environments and falls back to saving charts as PNG files instead. On desktop Linux, macOS, and Windows a window will open automatically.

---

## License

MIT License — free to use, modify, and share.

---

*Built with TensorFlow/Keras | MNIST Dataset*