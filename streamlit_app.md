# Streamlit App Guide

## Features
- **Model Metrics**: Compare accuracy, precision, recall, and F1 for all models
- **Model Summary**: Visual metric grid for all models
- **Classify Image**: Upload a fish image and get species prediction with confidence scores

## How to Run
```sh
streamlit run scripts/app.py
```

## Pages
- **Model Metrics**: Interactive bar charts and training history plots
- **Model Summary**: Metric grid with best values highlighted
- **Classify Image**: Upload image, view prediction and confidence scores

## Requirements
- All model files and metrics must be present in `models/`
- Class indices in `models/class_indices.csv`

## Tips
- For best results, use images similar to those in the training set
- App UI is styled for clarity and ease of use

See `usage_guide.md` for setup steps.