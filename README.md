# MNIST Handwritten Digit Recognition

A beginner-friendly neural network that recognises handwritten digits (0–9) using the MNIST dataset and TensorFlow/Keras. Achieves ~**97–98% accuracy** on the test set.

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
- Train for 10 epochs (~1–2 minutes on CPU)
- Save the model as `mnist_model.keras`
- Save a training chart as `training_history.png`

Expected output:
```
Test accuracy : 97.74 %
Test loss     : 0.0841
Model saved → mnist_model.keras
```

### 4. Predict your own digit

Draw a digit (0–9) and save it as a PNG or JPG, then run:

```bash
python predict.py your_digit.png
```

Expected output:
```
Predicted digit : 8
Confidence      : 94.3 %
Chart saved → prediction_result.png
```

---

## Neural Network Architecture

```
Input Layer     →  784 neurons  (one per pixel in a 28×28 image)
Hidden Layer 1  →  128 neurons  (ReLU activation)
Hidden Layer 2  →   64 neurons  (ReLU activation)
Output Layer    →   10 neurons  (Softmax → probability per digit 0–9)

Total parameters: 109,386
```

---

## How It Works

1. **Load** — MNIST dataset: 60,000 training + 10,000 test images
2. **Preprocess** — Flatten 28×28 → 784 pixels, normalize values to [0, 1]
3. **Train** — Backpropagation with Adam optimizer
4. **Predict** — Your image is auto-cropped, resized to 28×28, and fed to the model

---

## Tips for Best Prediction Results

| Do this ✅ | Avoid this ❌ |
|---|---|
| Draw a large digit that fills the frame | Tiny digit in a huge image |
| Use thick strokes | Thin or wispy lines |
| Keep both loops of an 8 equal in size | Uneven loops (looks like a 2) |
| Save as PNG | JPEG (adds compression noise) |
| Good lighting if photographing paper | Shadows or uneven lighting |

The script automatically:
- Detects white-on-black vs black-on-white and inverts if needed
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

---

## Results

After 10 epochs of training:

| Metric | Value |
|---|---|
| Training accuracy | ~99.3% |
| Validation accuracy | ~98.1% |
| **Test accuracy** | **~97.7%** |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'matplotlib'`**
→ Your virtual environment is not activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) first.

**`File not found: mnist_model.keras`**
→ You need to train the model first. Run `python train.py`.

**Low confidence / wrong prediction**
→ Make sure the digit is large, centered, and drawn with thick strokes. See tips above.

**Matplotlib windows not appearing on Windows**
→ Charts are saved as PNG files in the project folder — just open them directly.

---

## License

MIT License — free to use, modify, and share.

---

*Built with TensorFlow/Keras | MNIST Dataset*
