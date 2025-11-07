import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\val'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELS_DIR = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models'

# Load validation data
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
results = []

for model_file in model_files:
    print(f'Evaluating {model_file}...')
    model = load_model(os.path.join(MODELS_DIR, model_file))
    preds = model.predict(val_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))
    results.append([model_file, acc, prec, rec, f1])
    plt.figure(figsize=(6,6))
    plt.title(f'Confusion Matrix: {model_file}')
    plt.imshow(cm, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig(f'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models\\{model_file}_confusion_matrix.png')
    plt.close()

# Save results
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
results_df.to_csv('C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models\\model_comparison.csv', index=False)
print('Evaluation complete. Results saved.')
