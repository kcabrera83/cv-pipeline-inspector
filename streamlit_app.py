import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="CV Pipeline Inspector", layout="wide")
st.title("CV Pipeline Inspector")
st.markdown("Detect pipeline defects using computer vision.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'defect': joblib.load(d / 'defect_classifier.pkl'), 'severity': joblib.load(d / 'severity_estimator.pkl')}

st.sidebar.header("Input Parameters")
mean_intensity = st.sidebar.slider('Mean Intensity', 0, 255, 127)
std_intensity = st.sidebar.slider('Std Intensity', 0, 100, 50)
edge_density = st.sidebar.slider('Edge Density', 0, 1, 0)
entropy = st.sidebar.slider('Entropy', 0, 8, 4)
red_mean = st.sidebar.slider('Red Mean', 0, 255, 127)
green_mean = st.sidebar.slider('Green Mean', 0, 255, 127)
blue_mean = st.sidebar.slider('Blue Mean', 0, 255, 127)
hog_energy = st.sidebar.slider('Hog Energy', 0, 1000, 500)
laplacian_var = st.sidebar.slider('Laplacian Var', 0, 100, 50)
gradient_mag = st.sidebar.slider('Gradient Mag', 0, 100, 50)
contrast = st.sidebar.slider('Contrast', 0, 100, 50)
homogeneity = st.sidebar.slider('Homogeneity', 0, 1, 0)
energy = st.sidebar.slider('Energy', 0, 1, 0)
correlation = st.sidebar.slider('Correlation', -1, 1, 0)

if st.sidebar.button("Run"):
    try:
        x = np.array([[mean_intensity, std_intensity, edge_density, entropy, red_mean, green_mean, blue_mean, hog_energy, laplacian_var, gradient_mag, contrast, homogeneity, energy, correlation]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))