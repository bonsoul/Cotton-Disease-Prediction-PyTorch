# 🌿 Cotton Disease Prediction — Deep Learning Classifier

> CNN-based image classifier that detects disease in cotton plants with **~91% validation accuracy**.

---

## What It Does

Classifies cotton plant images into four categories:

| Class | Description |
|---|---|
| 🌱 Fresh Cotton Leaf | Healthy leaf, no disease |
| 🌿 Fresh Cotton Plant | Healthy full plant |
| 🍂 Diseased Cotton Leaf | Leaf showing disease symptoms |
| 🪴 Diseased Cotton Plant | Full plant showing disease symptoms |

---

## Model Architecture

Built with **PyTorch** — custom CNN with:
- Multiple convolutional layers + ReLU activation + max-pooling
- Fully connected output layers
- **Adam optimizer** | **Cross-Entropy loss**
- GPU acceleration supported

**Preprocessing:** 32×32 resize · ImageNet normalization · augmentation via `torchvision.transforms`

---

## Performance

| Metric | Result |
|---|---|
| Validation Accuracy | ~91% |
| Validation Loss | Steadily decreasing across epochs |
| Generalization | Strong on unseen test data |

---

## Quickstart

**1. Clone**
```bash
git clone https://github.com/yourusername/cotton-disease-prediction.git
cd cotton-disease-prediction
```

**2. Install dependencies**
```bash
pip install torch torchvision matplotlib numpy
```

**3. Organize dataset**
```
dataset/
├── train/
│   ├── fresh cotton leaf/
│   ├── fresh cotton plant/
│   ├── diseased cotton leaf/
│   └── diseased cotton plant/
├── val/
└── test/
```

**4. Train & evaluate**

Open `notebook.ipynb` in Google Colab or Jupyter and run all cells.

**5. Run inference**

Load saved weights and predict on new images:
```python
model.load_state_dict(torch.load('cotton_disease_model.pth'))
```

---

## Repo Structure

```
├── notebook.ipynb               # Full pipeline — training, evaluation, testing
├── cotton_disease_model.pth     # Saved model weights
└── dataset/                     # Train / val / test image directories
```

---

## Roadmap

- [ ] Expand classes to cover more cotton diseases
- [ ] Transfer learning with pre-trained backbones (ResNet, EfficientNet)
- [ ] Deploy as a web or mobile application for field use

---

## Stack

`Python 3.7+` · `PyTorch` · `Torchvision` · `NumPy` · `Matplotlib`
