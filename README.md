# Peptide-based Anti-aging Compound Prediction

This project focuses on predicting the anti-aging properties of peptides using various machine learning and deep learning models. The models are trained on ESM-2 protein embeddings.

## Project Structure

```
.
├── data/
│   ├── esm2_meanpool.npy
│   ├── labels.npy
│   └── peptide.csv
├── models/
│   ├── baseline_mlp_model.py
│   ├── random_forest_model.py
│   ├── svm_model.py
│   ├── AgeAttnMLP_model.py
│   ├── transformer_model.py
│   └── xgboost_model.py
├── src/
│   └── 00_extract_esm2_features.py
├── weights/
│   └── AgeAttnMLP_best.pth
└── README.md
```

- **data/**: Contains the feature embeddings and labels for training and prediction.
- **models/**: Contains the Python scripts defining the model architectures.
- **src/**: Contains scripts for data preprocessing and feature extraction.
- **weights/**: Contains the pre-trained weights for the best performing model.
- **README.md**: This file.

## Models

This project explores several models for the prediction task:

- **baseline_mlp_model.py**: A simple Multi-Layer Perceptron (MLP) model that serves as a baseline.
- **random_forest_model.py**: A Random Forest classifier.
- **svm_model.py**: A Support Vector Machine (SVM) classifier with an RBF kernel.
- **AgeAttnMLP_model.py**: An optimized MLP model incorporating attention mechanisms (ECA, CoordAttention), DropPath regularization, and residual connections.
- **transformer_model.py**: A transformer-based model for feature classification.
- **xgboost_model.py**: An XGBoost classifier.

## Usage

Each script in the `models/` directory defines a function or class to create the respective model. These models can be imported and used in a training and evaluation pipeline.
