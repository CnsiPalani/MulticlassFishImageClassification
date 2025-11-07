# Model Details

## Models Used
- **Custom CNN**: Sequential convolutional layers, trained from scratch.
- **Transfer Learning Models**:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0

All models use softmax output for multiclass classification.

## Training
- Data augmentation: rotation, shift, zoom, flip
- Input size: 224x224 pixels
- Optimizer: Adam
- Loss: Categorical crossentropy
- Metrics: Accuracy

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1
- Confusion matrices for each model
- Results saved in `models/model_comparison.csv`

## Files
- Model weights: `models/*.keras`, `models/*.h5`
- Training history: `models/*_history.csv`
- Confusion matrices: `models/*_confusion_matrix.png`

See `usage_guide.md` for how to train and evaluate models.