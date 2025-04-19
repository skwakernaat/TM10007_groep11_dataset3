# TM10007_groep11_dataset3
# Radiomic Feature-Based Classification of GIST Tumors
This repository contains a reproducible machine learning pipeline for classifying gastrointestinal stromal tumors (GIST) using radiomic features. The workflow includes data preprocessing, feature selection, model training and tuning, and performance evaluation using standard classifiers.

## Objective
To evaluate the discriminative power of radiomic features in distinguishing between **GIST** and **non-GIST** lesions using multiple machine learning algorithms and feature selection techniques.

### Run in Google Colab
Click the badge below to launch the full pipeline in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skwakernaat/TM10007_groep11_dataset3/blob/main/TM10007_groep11_dataset3.ipynb)

## Project Structure

- `main.py`
  Entry point that runs the full pipeline: preprocessing, feature selection, model training, evaluation.

- `GIST_radiomicFeatures.csv`
  Dataset containing radiomic features for GIST and non-GIST cases.

- `requirements.txt`
  List of Python dependencies needed to run the project.

- `preprocessing/`
  Modules for preparing the data:
  - `load_data.py` – Loads the dataset from CSV.
  - `clean_data.py` – Cleans and encodes the labels.
  - `split_data.py` – Splits the data into training and testing sets.
  - `scale_data.py` – Scales feature values.
  - `check_data_balance.py` – Checks for class imbalance.
  - `forward_feature_selection.py` – Applies sequential feature selection.

- `classifiers/`
  Contains implementations and grid search tuning for different classifiers:
  - `qda_classifier.py`
  - `random_forest.py`
  - `SVM_classifier.py`
  - `linear_classifiers.py`
  - `grid_search.py` – Shared utility for grid search cross-validation.

- `results/`
  Functions to visualize and evaluate model performance:
  - `plot_learning_curve.py`
  - `evaluate_model.py`

- `notebooks/`
  Jupyter/Colab-compatible notebooks for running the pipeline in a cloud environment.
  - `TM10007_groep11_dataset3.ipynb`
