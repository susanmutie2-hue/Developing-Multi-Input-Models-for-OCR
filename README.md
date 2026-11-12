# Developing-Multi-Input-Models-for-OCR
A project  using multi-modal learning to train an Optical Character Recognition (OCR) model. 

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📌 Project Overview

**DigiNsure Inc.** is an innovative insurance company focused on modernizing claims processing and elevating customer service interactions through digital transformation. As part of their newest initiative to digitize historical insurance claim documents, this project tackles the automated labeling and classification of scanned identification documents. 

Specifically, the model distinguishes between **Primary IDs** and **Secondary IDs** extracted from historical paper archives. To maximize accuracy in complex scenarios, this project leverages a **multi-modal learning approach**—combining visual features from scanned document images with metadata representing the insurance category (home, life, auto, health, or other).

---

## 🏗️ Model Architecture

The `OCRModel` is built using **PyTorch** and implements a multi-input architecture that fuses visual and categorical modalities before final classification:

1. **Image Branch:** 
   - Processes $64 \times 64$ grayscale document crops.
   - Consists of a 2D Convolutional layer (`nn.Conv2d`: 1 input channel, 16 output channels, $3 \times 3$ kernel, padding 1), followed by ReLU activation, Max Pooling, Flattening, and a Fully Connected layer yielding 128 features.
2. **Type Branch:** 
   - Processes the one-hot encoded insurance type vector (5 categories: home, life, auto, health, other).
   - Passes through a Linear layer mapping to 10 features, followed by a ReLU activation.
3. **Fusion & Classification Layer:** 
   - Concatenates the image features and type features ($128 + 10 = 138$ combined dimensions).
   - Feeds the combined representation into a final linear classifier (`nn.Linear`) to predict the ID label class (Primary vs. Secondary).

---

## 📂 Project Structure

```text
├── project_utils.py             # Custom dataset utilities and helper classes
├── ocr_insurance_dataset.pkl    # Serialized dataset containing image-type-label pairs
├── model.py / script.py         # Model definition, data loading, and training loop
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and the following required libraries installed:

```bash
pip install torch torchvision matplotlib numpy
```

### Dataset Inspection

The dataset can be loaded via `pickle` and inspected using the provided visualization utility:

```python
import pickle
from project_utils import ProjectDataset

# Load dataset
dataset = pickle.load(open('ocr_insurance_dataset.pkl', 'rb'))

# Inspect dataset samples
show_dataset_images(dataset, num_images=5)
```

---

## 💻 Training the Model

The training script initializes the multi-modal architecture, sets up the Adam optimizer and Cross-Entropy loss function, and trains the network for 10 epochs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Instantiate Model, Loss, and Optimizer
num_classes = len(dataset.label_mapping) if hasattr(dataset, 'label_mapping') else 2
model = OCRModel(num_classes=num_classes)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for (batch_images, batch_types), batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_images, batch_types)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
```

---

## 📈 Training Results

The model successfully converges over 10 epochs, demonstrating a steady reduction in cross-entropy loss:

| Epoch | Loss | Status |
| :---: | :---: | :---: |
| 1 / 10 | 7.9460 | Initializing |
| 2 / 10 | 3.4253 | Optimizing |
| 3 / 10 | 2.6708 | Optimizing |
| 4 / 10 | 1.1700 | Learning |
| 5 / 10 | 0.6288 | Refining |
| 6 / 10 | 0.6036 | Refining |
| 7 / 10 | 0.5507 | Converging |
| 8 / 10 | 0.5909 | Converging |
| 9 / 10 | 0.5006 | Stable |
| 10 / 10 | **0.4256** | **Converged** |

---

## 🔮 Future Enhancements

- **Deeper Backbones:** Integrate deeper convolutional feature extractors (e.g., ResNet-18 variants) for higher-resolution document scans.
- **Attention Mechanisms:** Implement multi-modal attention gates to dynamically weigh the importance of image regions versus insurance context.
- **Data Augmentation:** Apply rotation, noise injection, and scaling to improve model robustness against degraded historical paper scans.

---

## 📄 License

This project is proprietary to **DigiNsure Inc.** All rights reserved.
