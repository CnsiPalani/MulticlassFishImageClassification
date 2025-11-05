# Multiclass Fish Image Classification

## Overview
Classify fish images into multiple categories using deep learning (CNN and transfer learning). Includes model training, evaluation, and deployment via Streamlit.

## Project Structure
- `scripts/` : Python scripts for preprocessing, training, evaluation
- `models/` : Saved trained models (.h5)
- `notebooks/` : Jupyter notebooks for exploration
- `docs/` : Documentation and reports
- `requirements.txt` : Dependencies

## Steps
1. Data Preprocessing & Augmentation
2. Model Training (CNN & Transfer Learning)
3. Model Evaluation
4. Streamlit Deployment

## Usage
- Run preprocessing: `python scripts/data_preprocessing.py`
- Train models: `python scripts/train_cnn.py` and `python scripts/train_transfer.py`
- Evaluate: `python scripts/evaluate.py`
- Launch app: `streamlit run scripts/app.py`

## Dataset
- Fish images categorized by species (see data folder)

## Deliverables
- Trained models (.h5)
- Streamlit app
- Python scripts
- Comparison report

## Skills
- Deep Learning, Python, TensorFlow/Keras, Streamlit, Data Preprocessing, Transfer Learning, Model Evaluation, Visualization, Model Deployment
