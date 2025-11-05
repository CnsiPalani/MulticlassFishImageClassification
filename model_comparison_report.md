# Model Comparison Report

This report summarizes the performance of all trained models for multiclass fish image classification.

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results
See `models/model_comparison.csv` for detailed metrics.
Confusion matrices are saved as PNG files in the `models/` folder for each model.

## Best Model
Select the model with the highest F1-score and accuracy for deployment in the Streamlit app.

## Notes
- All models trained and evaluated using the same validation split.
- Data augmentation applied during training.
- For further details, see scripts and README.
