import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import os
import matplotlib.pyplot as plt


MODELS_DIR = 'C:\\WA\\POC\\Python\\MulticlassFishImageClassification\\models'
IMG_SIZE = (224, 224)
# Load class indices
class_indices = pd.read_csv(os.path.join(MODELS_DIR, 'class_indices.csv'))
class_names = class_indices['Class'].tolist()

# Load model comparison metrics
metrics_df = pd.read_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'))

# Load best model (MobileNet)
MODEL_PATH = os.path.join(MODELS_DIR, 'MobileNet_fish_classifier.keras')
model = load_model(MODEL_PATH)

## Modern header with background color and logo
custom_theme_css = """
<style>
.modern-header {
    background: linear-gradient(90deg, #4F8BF9 60%, #2E8B57 100%);
    padding: 22px 0 16px 0;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 2px 12px rgba(79,139,249,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 18px;
}
.modern-header-logo {
    width: 54px;
    height: 54px;
    margin-right: 18px;
    border-radius: 50%;
    background: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(46,139,87,0.10);
}
.modern-header-title {
    color: #fff;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 0;
    margin-top: 0;
    text-shadow: 0 2px 8px rgba(46,139,87,0.10);
}
.modern-header-sub {
    color: #e3f0ff;
    font-size: 1.08rem;
    font-weight: 400;
    margin-top: 2px;
    margin-bottom: 0;
    letter-spacing: 0.5px;
}
/* Custom theme for buttons and text */
button, .stButton > button {
    background: linear-gradient(90deg, #4F8BF9 60%, #2E8B57 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1.08rem !important;
    box-shadow: 0 2px 8px rgba(79,139,249,0.10);
    transition: box-shadow 0.2s, transform 0.2s;
    margin-bottom: 6px !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(46,139,87,0.18);
    transform: translateY(-2px) scale(1.04);
    background: linear-gradient(90deg, #2E8B57 60%, #4F8BF9 100%) !important;
}
.stRadio > div {
    color: #4F8BF9 !important;
    font-weight: 600 !important;
    font-size: 1.08rem !important;
}
.stMarkdown, .stText, .stDataFrame, .stExpander {
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
}
.stMarkdown h5, .stMarkdown h4, .stMarkdown h3, .stMarkdown h6 {
    color: #2E8B57 !important;
    font-weight: 700 !important;
}
.stSuccess {
    background: #e6ffe6 !important;
    color: #2E8B57 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stInfo {
    background: #e3f0ff !important;
    color: #4F8BF9 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
"""
st.markdown(custom_theme_css, unsafe_allow_html=True)
st.markdown(
    '''<div class="modern-header">
        <div class="modern-header-logo">
            <img src="https://img.icons8.com/color/96/000000/fish.png" width="40" height="40" style="margin:0;" />
        </div>
        <div>
            <div class="modern-header-title">Fish Species Image Classification</div>
            <div class="modern-header-sub">AI-powered multiclass fish recognition</div>
        </div>
    </div>''', unsafe_allow_html=True
)

# Sidebar with logo and info
st.sidebar.markdown('<img src="https://img.icons8.com/color/96/000000/fish.png" width="40" style="margin-bottom:-6px;"> <span style="color:#4F8BF9;font-size:13px;font-weight:bold;">Fish Classifier</span>', unsafe_allow_html=True)
st.sidebar.markdown('<hr style="margin:4px 0 8px 0;">', unsafe_allow_html=True)

# Sidebar navigation for page selection
page = st.sidebar.radio(
    "Navigate",
    ["📊 Model Metrics", "🖼️ Model Summary", "🖼️ Classify Image"],
    index=0
)

if page == "📊 Model Metrics":
    st.markdown('<h5 style="color:#2E8B57;margin-bottom:0.2rem;">Model Metrics Comparison</h5>', unsafe_allow_html=True)
    available_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    import plotly.graph_objects as go
    fig = go.Figure()
    for metric in available_metrics:
        fig.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df[metric],
            name=metric,
            text=[f'{v:.2f}' for v in metrics_df[metric]],
            textposition='auto',
        ))
    fig.update_layout(
        barmode='group',
        xaxis_title='Model',
        yaxis_title='Score',
        title='',
        legend_title='Metric',
        template='plotly_white',
        height=320,
        margin={'l':10, 'r':10, 't':10, 'b':10}
    )
    st.plotly_chart(fig, use_container_width=True)
    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
    st.success(f'{best_model} is currently the best performing model for fish image classification!')

    with st.expander("Training History Charts", expanded=False):
        history_files = [
            ('CNN', 'models/cnn_fish_classifier_history.csv'),
            ('EfficientNetB0', 'models/EfficientNetB0_fish_classifier_history.csv'),
            ('InceptionV3', 'models/InceptionV3_fish_classifier_history.csv'),
            ('MobileNet', 'models/MobileNet_fish_classifier_history.csv'),
            ('ResNet50', 'models/ResNet50_fish_classifier_history.csv'),
            ('VGG16', 'models/VGG16_fish_classifier_history.csv'),
        ]
        import plotly.graph_objects as go
        for model_name, file_path in history_files:
            if os.path.exists(file_path):
                st.markdown(f'<span style="font-size:13px;"><b>{model_name}</b></span>', unsafe_allow_html=True)
                hist_df = pd.read_csv(file_path)
                col1, col2 = st.columns(2)
                title_style = "font-size:15px;color:#2E8B57;font-weight:600;margin-bottom:12px;letter-spacing:0.5px;"
                with col1:
                    st.markdown(f'<div style="{title_style}">Accuracy</div>', unsafe_allow_html=True)
                    acc_fig = go.Figure()
                    acc_fig.add_trace(go.Scatter(y=hist_df['accuracy'], mode='lines', name='Train Accuracy'))
                    if 'val_accuracy' in hist_df:
                        acc_fig.add_trace(go.Scatter(y=hist_df['val_accuracy'], mode='lines', name='Val Accuracy'))
                    acc_fig.update_layout(title='', xaxis_title='Epoch', yaxis_title='Accuracy', template='plotly_white', height=340)
                    st.plotly_chart(acc_fig, use_container_width=True)
                with col2:
                    st.markdown(f'<div style="{title_style}">Loss</div>', unsafe_allow_html=True)
                    loss_fig = go.Figure()
                    loss_fig.add_trace(go.Scatter(y=hist_df['loss'], mode='lines', name='Train Loss'))
                    if 'val_loss' in hist_df:
                        loss_fig.add_trace(go.Scatter(y=hist_df['val_loss'], mode='lines', name='Val Loss'))
                    loss_fig.update_layout(title='', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_white', height=340)
                    st.plotly_chart(loss_fig, use_container_width=True)

    with st.expander("Confusion Matrices", expanded=False):
        col1, col2 = st.columns([1,1], gap="small")
        with col1:
            st.image('models/EfficientNetB0_fish_classifier.keras_confusion_matrix.png', caption='EfficientNetB0', use_container_width=True)
            st.image('models/InceptionV3_fish_classifier.keras_confusion_matrix.png', caption='InceptionV3', use_container_width=True)
            st.image('models/MobileNet_fish_classifier.keras_confusion_matrix.png', caption='MobileNet', use_container_width=True)
        with col2:
            st.image('models/ResNet50_fish_classifier.keras_confusion_matrix.png', caption='ResNet50', use_container_width=True)
            st.image('models/VGG16_fish_classifier.keras_confusion_matrix.png', caption='VGG16', use_container_width=True)

elif page == "🖼️ Model Summary":
    st.markdown('<h4 style="color:#2E8B57;margin-top:10px;margin-bottom:18px;">Model Summary</h4>', unsafe_allow_html=True)
    available_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    # Metric grid: each metric is a bordered widget, showing all models' values in a horizontal row
    # Enhanced metric grid UI
    enhanced_css = """
    <style>
    .metric-grid {
        display: flex;
        flex-direction: column;
        gap: 14px;
        margin-bottom: 16px;
        align-items: center;
    }
    .metric-row {
        display: flex;
        flex-direction: row;
        gap: 14px;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .metric-label {
        font-size:16px;
        color:#2E8B57;
        font-weight:bold;
        margin-right:10px;
        display:flex;
        align-items:center;
    }
    .metric-label i {margin-right:6px;}
    .metric-widget {
        border:1.5px solid #4F8BF9;
        border-radius:12px;
        padding:12px 18px;
        background:linear-gradient(135deg,#f7fbff 80%,#e3f0ff 100%);
        min-width:140px;
        max-width:140px;
        min-height:90px;
        max-height:90px;
        text-align:center;
        box-shadow:0 2px 8px rgba(79,139,249,0.08);
        transition:box-shadow 0.2s, transform 0.2s;
        position:relative;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    }
    .metric-widget:hover {box-shadow:0 4px 16px rgba(79,139,249,0.18); transform:translateY(-2px) scale(1.03);}
    .model-label {font-size:13px; color:#4F8BF9; font-weight:bold; margin-bottom:2px;}
    .metric-value {font-size:18px; color:#222; font-weight:bold;}
    .best-value {color:#2E8B57; background:#e6ffe6; border-radius:6px; padding:2px 6px; font-size:17px; font-weight:bold; box-shadow:0 1px 4px rgba(46,139,87,0.08);}
    .metric-widget .star {position:absolute; top:8px; right:8px; font-size:18px; color:#FFD700;}
    </style>
    """
    st.markdown(enhanced_css, unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    # Metric icons
    metric_icons = {
        'Accuracy': '✅',
        'Precision': '🎯',
        'Recall': '🔄',
        'F1': '⭐',
    }
    grid_html = "<div class='metric-grid'>"
    for metric in available_metrics:
        # Find best value for highlight
        best_val = metrics_df[metric].max()
        best_idxs = metrics_df.index[metrics_df[metric] == best_val].tolist()
        grid_html += f"<div class='metric-row'><div class='metric-label'><i>{metric_icons.get(metric,'')}</i>{metric}</div>"
        for idx, row in metrics_df.iterrows():
            is_best = idx in best_idxs
            value_html = f"<span class='metric-value'>{row[metric]:.2f}</span>"
            star_html = "<span class='star'>★</span>" if is_best else ""
            if is_best:
                value_html = f"<span class='best-value'>{row[metric]:.2f}</span>"
            # Split model name on '_' and show only the first part
            model_display = row['Model'].split('_')[0] if '_' in row['Model'] else row['Model']
            grid_html += f"<div class='metric-widget'>{star_html}<div class='model-label'>{model_display}</div>{value_html}</div>"
        grid_html += "</div>"
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

elif page == "🖼️ Classify Image":
    st.markdown('<h5 style="color:#2E8B57;margin-bottom:0.2rem;">Classify Fish Image</h5>', unsafe_allow_html=True)
    st.write('Upload a fish image to get a prediction using the best model.')
    uploaded_file = st.file_uploader('Upload a fish image', type=['jpg', 'jpeg', 'png'], help='Supported formats: jpg, jpeg, png')
    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner('Predicting...'):
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
        st.image(img, caption='Uploaded Image', width=220)
        st.markdown(f'<h6 style="color:#4F8BF9;margin-bottom:0.1rem;">Predicted Species: <b>{class_names[pred_idx]}</b> 🐟</h6>', unsafe_allow_html=True)
        st.info(f"Confidence: {preds[pred_idx]:.2f}")
        st.write('Confidence Scores:')
        import plotly.express as px
        conf_df = pd.DataFrame({
            'Species': class_names,
            'Confidence': preds
        })
        fig = px.bar(conf_df, x='Species', y='Confidence', text=conf_df['Confidence'].apply(lambda x: f'{x:.2f}'))
        fig.update_traces(textposition='auto', marker_color='rgba(79,139,249,0.8)')
        fig.update_layout(
            title='Confidence Scores for All Classes',
            xaxis_title='Species',
            yaxis_title='Confidence',
            template='plotly_white',
            height=400,
            margin={'l':10, 'r':10, 't':30, 'b':10}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(conf_df.style.format({'Confidence': '{:.2f}'}), use_container_width=True)
        
        