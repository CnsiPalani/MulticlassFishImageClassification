
# Multiclass Fish Image Classification

## Overview
This project uses deep learning to classify fish species from images. It features custom CNN and multiple transfer learning models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0), with a Streamlit web app for interactive predictions and model comparison.

## Features
- **Data Preprocessing & Augmentation**: Automated scripts for preparing and augmenting fish image datasets.
- **Model Training**: Train both custom CNN and state-of-the-art transfer learning models.
- **Model Evaluation**: Automated evaluation, metrics comparison, and confusion matrix visualization.
- **Streamlit App**: User-friendly interface for model metrics, summaries, and image classification.

## Project Structure
```
├── scripts/                # Python scripts for training, evaluation, and app
├── models/                 # Saved models and training histories
├── Dataset/                # Organized fish image data (train/val/test)
├── docs/                   # Documentation and class indices
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

## Quick Start

1. **Install dependencies**
	```sh
	pip install -r requirements.txt
	```

2. **Preprocess data**
	```sh
	python scripts/data_preprocessing.py
	```

3. **Train models**
	- Custom CNN:
	  ```sh
	  python scripts/train_cnn.py
	  ```
	- Transfer Learning:
	  ```sh
	  python scripts/train_transfer.py
	  ```

4. **Evaluate models**
	```sh
	python scripts/evaluate.py
	```

5. **Launch Streamlit app**
	```sh
	streamlit run scripts/app.py
	```

## Streamlit App
- **Model Metrics**: Compare accuracy, precision, recall, and F1 across models.
- **Model Summary**: Visual metric grid for all models.
- **Classify Image**: Upload a fish image and get species prediction with confidence scores.

## Dataset
- Images are organized by species in `Dataset/images/data/{train,val,test}/`.
- See `docs/class_indices.csv` for class labels.

## Model Outputs
- Trained models: `.keras` and `.h5` files in `models/`
- Training histories: CSV files for each model
- Confusion matrices: PNG images for each model
- Model comparison: `model_comparison.csv`

## Technologies Used
- Python, TensorFlow, Keras, scikit-learn, Streamlit, Matplotlib, Pandas, NumPy, Pillow

## References
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Keras Applications](https://keras.io/api/applications/)
