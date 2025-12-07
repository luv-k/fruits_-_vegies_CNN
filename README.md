# Food Image Classification using CNN

## Overview

This project implements a deep learning–based food image classification system trained on a multi-class dataset of fruits and vegetables. It uses a custom Convolutional Neural Network (CNN) architecture built with PyTorch and evaluates the model on separate training, validation, and test splits.

The goal of this project is to accurately identify food categories from images, making it useful for applications such as dietary tracking, automated checkout systems, or educational tools.

## Dataset

* **Train samples:** 46,187
* **Validation samples:** 12,544
* **Test samples:** 13,353
* **Number of classes:** 50
* Input images are preprocessed to **224×224×3**.

## Model Architecture

A custom CNN model (`netCNN`) composed of:

* Convolutional layers with ReLU activations
* Max pooling layers
* Dropout for regularization
* Fully connected layers for classification

The model is trained using:

* **Loss function:** CrossEntropyLoss
* **Optimizer:** SGD with momentum
* **Learning rate scheduling:** StepLR
* **Epochs:** 20

## Training Procedure

* Training and validation are performed each epoch.
* The model with the highest validation accuracy is saved as **`best_model.pth`**.
* Final evaluation is performed on the test dataset after loading the best checkpoint.

## Results

* **Test Accuracy:** **50%**
* **Device used:** CUDA (GPU)

These results reflect the current state of training in the provided notebook. Accuracy may improve with hyperparameter tuning, data augmentation, or a more advanced model architecture (e.g., transfer learning).

## Project Structure

```
├── fooddetection.ipynb       # Training, validation, and testing pipeline
├── best_model.pth            # Saved model weights (if included)
├── README.md                 # Project documentation
└── data/                     # Dataset (not included in repository)
```

## How to Run

1. Install dependencies:

   ```
   pip install torch torchvision
   ```
2. Open the notebook:

   ```
   jupyter notebook fooddetection.ipynb
   ```
3. Run all cells to train or evaluate the model.
