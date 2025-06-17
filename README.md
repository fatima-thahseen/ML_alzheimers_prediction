# 🧠 ML Alzheimer's Prediction

This project is a machine learning application that predicts Alzheimer's diagnosis using various health and lifestyle factors. The model is built using XGBoost and deployed as a user-friendly web app using Streamlit.

🔗 **Live Demo**: [Click here to try the app](https://thahsee-ml-alzheimers-prediction.streamlit.app/)

---

## Features

- **Model**: XGBoost (optimized using `RandomizedSearchCV`)
- **Deployment**: Streamlit web application
- **Preprocessing**:
  - Label Encoding: 15 categorical features
  - One-Hot Encoding: 2 categorical features
  - Feature Scaling: StandardScaler
- **Model Serialization**: `.sav` format for quick loading in production

---

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
