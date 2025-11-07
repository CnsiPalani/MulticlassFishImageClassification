import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
from tensorflow.keras import layers, models
import os

DATA_DIR = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\train'
validation_dir = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\val'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2

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
    DATA_DIR,
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

num_classes = train_generator.num_classes

base_models = {
    'VGG16': VGG16,
    'ResNet50': ResNet50,
    'MobileNet': MobileNet,
    'InceptionV3': InceptionV3,
    'EfficientNetB0': EfficientNetB0
}

for name, base in base_models.items():
    print(f'Training {name}...')
    base_model = base(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
    # Ensure the models directory exists
    models_dir = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models'
    os.makedirs(models_dir, exist_ok=True)
    # Save training history to CSV
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{models_dir}\{name}_fish_classifier_history.csv', index=False)
    print(f'{name} training history saved.')
    try:
        model.save(f'{models_dir}\{name}_fish_classifier.keras')
        print(f'{name} model trained and saved.')
    except Exception as e:
        print(f'Error saving {name} model:', e)