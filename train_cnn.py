import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\train'
validation_dir = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\Dataset\\images\\data\\val'


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

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

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

model.save('C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models\\cnn_fish_classifier.h5')
print('CNN model trained and saved.')
# Save training history to CSV
import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv('C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models\\cnn_fish_classifier_history.csv', index=False)
print('Training history saved.')
