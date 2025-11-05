import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\train'
validation_dir = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\val'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
pd.DataFrame(list(train_generator.class_indices.items()), columns=['Class', 'Index']).to_csv('C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models\\class_indices.csv', index=False)

print('Data generators created and class indices saved.')
