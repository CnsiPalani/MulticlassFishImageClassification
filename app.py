import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

MODELS_DIR = '../models'
IMG_SIZE = (224, 224)

# Load class indices
class_indices = pd.read_csv(os.path.join(MODELS_DIR, 'class_indices.csv'))
class_names = class_indices['Class'].tolist()

# Load best model (change filename as needed)
MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_fish_classifier.h5')
model = load_model(MODEL_PATH)

st.title('Fish Species Image Classification')

uploaded_file = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Species: **{class_names[pred_idx]}**')
    st.write('Confidence Scores:')
    for i, score in enumerate(preds):
        st.write(f'{class_names[i]}: {score:.2f}')
