# Usage Guide

## 1. Install Dependencies
```sh
pip install -r requirements.txt
```

## 2. Data Preprocessing
```sh
python scripts/data_preprocessing.py
```

## 3. Model Training
- Custom CNN:
  ```sh
  python scripts/train_cnn.py
  ```
- Transfer Learning:
  ```sh
  python scripts/train_transfer.py
  ```

## 4. Model Evaluation
```sh
python scripts/evaluate.py
```

## 5. Launch Streamlit App
```sh
streamlit run scripts/app.py
```

## 6. Dataset Structure
- Images are organized by species in `Dataset/images/data/{train,val,test}/`
- Class labels: see `docs/class_indices.csv`

## 7. Outputs
- Trained models: `models/*.keras`, `models/*.h5`
- Training histories: `models/*_history.csv`
- Confusion matrices: `models/*_confusion_matrix.png`
- Model comparison: `models/model_comparison.csv`

For more details, see `project_overview.md` and the main `README.md`.