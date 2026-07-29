import streamlit as st, joblib, numpy as np
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Pipeline Inspector", page_icon="\U0001f4ca")
st.header("Pipeline Inspector")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'defect': joblib.load(p / 'defect_classifier.pkl'), 'severity': joblib.load(p / 'severity_estimator.pkl')}

with st.sidebar:
    st.write('Configure parameters below')
    c = st.columns(2)
    mean = c[0].slider('Mean', 0, 255, 127)
    std = c[1].slider('Std', 0, 100, 50)
    c = st.columns(2)
    edge = c[0].slider('Edge', 0, 1, 0)
    entropy = c[1].slider('Entropy', 0, 8, 4)
    c = st.columns(2)
    red = c[0].slider('Red', 0, 255, 127)
    green = c[1].slider('Green', 0, 255, 127)
    c = st.columns(2)
    blue = c[0].slider('Blue', 0, 255, 127)
    hog = c[1].slider('Hog', 0, 1000, 500)
    c = st.columns(2)
    lap = c[0].slider('Lap', 0, 100, 50)
    grad = c[1].slider('Grad', 0, 100, 50)
    run = st.button('Analyze', use_container_width=True)

if run:
    x = np.array([[mean, std, edge, entropy, red, green, blue, hog, lap, grad]])
    st.divider()
    m = models['defect']
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else f'{p[0]:.2f}'
    else:
        v = f'{m.predict(x)[0]:.2f}'
    st.metric('Defect', v)
    m = models['severity']
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else f'{p[0]:.2f}'
    else:
        v = f'{m.predict(x)[0]:.2f}'
    st.metric('Severity', v)