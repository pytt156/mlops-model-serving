import streamlit as st
import requests

PREDICT_URL = "http://localhost:8000/predict"

st.title("CIFAR-10 Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    content = uploaded_file.getvalue()
    try:
        st.image(content)
        if st.button("Predict"):
            try:
                response = requests.post(
                    PREDICT_URL,
                    files={"file": (uploaded_file.name, content, uploaded_file.type or "image/jpeg")},
                    timeout=30,
                )
                response.raise_for_status()
                out = response.json()
                st.success(f"**{out['predicted_label']}**")
                st.metric("Confidence", f"{out['confidence']:.2%}")
                with st.expander("Raw response"):
                    st.json(out)
            except requests.RequestException as e:
                st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
