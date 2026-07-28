import streamlit as st
import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="CV Pipeline Inspector", layout="wide")
st.title("CV Pipeline Inspector")
st.markdown("Detect pipeline defects (corrosion, cracks, dents, leaks) using computer vision features.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: pickle.load(open(d / v, "rb")) for k, v in [("defect", "defect_classifier.pkl"), ("severity", "severity_estimator.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
mean_intensity = st.sidebar.slider("Mean Intensity", 0, 255, 127)
std_intensity = st.sidebar.slider("Std Intensity", 0, 100, 50)
edge_density = st.sidebar.slider("Edge Density", 0, 1, 0)
texture_entropy = st.sidebar.slider("Texture Entropy", 0, 8, 4)
red_mean = st.sidebar.slider("Red Mean", 0, 255, 127)
green_mean = st.sidebar.slider("Green Mean", 0, 255, 127)
blue_mean = st.sidebar.slider("Blue Mean", 0, 255, 127)
hog_energy = st.sidebar.slider("Hog Energy", 0, 1000, 500)
laplacian_variance = st.sidebar.slider("Laplacian Variance", 0, 100, 50)
gradient_magnitude = st.sidebar.slider("Gradient Magnitude", 0, 100, 50)
glcm_contrast = st.sidebar.slider("Glcm Contrast", 0, 100, 50)
glcm_homogeneity = st.sidebar.slider("Glcm Homogeneity", 0, 1, 0)
glcm_energy = st.sidebar.slider("Glcm Energy", 0, 1, 0)
glcm_correlation = st.sidebar.slider("Glcm Correlation", -1, 1, 0)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[mean_intensity, std_intensity, edge_density, texture_entropy, red_mean, green_mean, blue_mean, hog_energy, laplacian_variance, gradient_magnitude, glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation]])
        m = models["defect"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Defect", result if isinstance(result, str) else f"{result:.4f}")
        m = models["severity"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Severity", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")