# TM10007_groep11_dataset3
# Radiomic Feature-Based Classification of GIST Tumors
This repository contains a reproducible machine learning pipeline for classifying gastrointestinal stromal tumors (GIST) using radiomic features. The workflow includes data preprocessing, feature selection, model training and tuning, and performance evaluation using standard classifiers.

## Objective
To evaluate the discriminative power of radiomic features in distinguishing between **GIST** and **non-GIST** lesions using multiple machine learning algorithms and feature selection techniques.

### Run in Google Colab
Click the badge below to launch the full pipeline in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skwakernaat/TM10007_groep11_dataset3/blob/main/TM10007_groep11_dataset3.ipynb)

TM10007_groep11_dataset3/
├── main.py                          # Orchestrates the full analysis pipeline
├── GIST_radiomicFeatures.csv        # Dataset used for classification
├── preprocessing/                   # Data cleaning, splitting, scaling, feature selection
├── classifiers/                     # ML models with hyperparameter tuning using grid search
├── results/                         # Learning curves, evaluation metrics
└── notebooks/
    └── TM10007_groep11_dataset3.ipynb           # Colab-compatible entry point
