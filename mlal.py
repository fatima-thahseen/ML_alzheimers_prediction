import streamlit as st
import pickle
import numpy as np
# from  PIL import Image

# Function to load pickle files
def load_pickle(file_name):
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"❌ Error: {file_name} not found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {file_name}: {e}")
        return None

# Load Model & Preprocessing Objects
xgb_model = load_pickle('xgboostmodel.sav')
scaler = load_pickle('sd.sav')
leout = load_pickle('leout.sav')

# Load Label Encoders
encoders = {
    "Gender": load_pickle('legen.sav'),
    "Physical Activity Level": load_pickle('leph.sav'),
    "Smoking Status": load_pickle('lesmo.sav'),
    "Alcohol Consumption": load_pickle('lealc.sav'),
    "Diabetes": load_pickle('ledia.sav'),
    "Hypertension": load_pickle('lehyp.sav'),
    "Cholesterol Level": load_pickle('lecho.sav'),
    "Family History of Alzheimers": load_pickle('lehis.sav'),
    "Depression Level": load_pickle('ledep.sav'),
    "Sleep Quality": load_pickle('lesl.sav'),
    "Air Pollution Exposure": load_pickle('leair.sav'),
    "Genetic Risk Factor (APOE-ε4 allele)": load_pickle('leris.sav'),
    "Social Engagement Level": load_pickle('lesoc.sav'),
    "Stress Levels": load_pickle('lestr.sav'),
    "Urban vs Rural Living": load_pickle('leur.sav'),
    "Dietary Habits": load_pickle('encdie.sav'),
    "Marital Status": load_pickle('encmari.sav')
}

if not xgb_model or not scaler or not leout or any(v is None for v in encoders.values()):
    st.error("Missing model or encoder files. Please check that all necessary `.sav` files are present.")
    st.stop()

# Sidebar
st.sidebar.title("🧠 Alzheimer's Prediction")
st.sidebar.write("Hey there! I assess your risk of Alzheimer's based on your inputs")
st.sidebar.image("ai-alzheimers-feat.jpg", use_container_width=True) 
st.sidebar.markdown("---")

# Main Title
st.title("🔬 Alzheimer's Disease Risk Prediction")
st.write("Fill out the form below to receive a prediction.")

# User Input Form
with st.form("alzheimers_form"):
    age = st.number_input("Age", min_value=15, max_value=100, step=1, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.number_input("Education Level (Years)", min_value=0, max_value=25, step=1, value=12)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=22.5)
    physical_activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])
    family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
    cognitive_test = st.number_input("Cognitive Test Score", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    depression = st.selectbox("Depression Level", ["Low", "Medium", "High"])
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Average", "Unhealthy"])
    air_pollution = st.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
    marital_status = st.selectbox("Marital Status", ["Single", "Widowed", "Married"])
    genetic_risk = st.selectbox("Genetic Risk Factor (APOE-ε4 allele)", ["No", "Yes"])
    social_engagement = st.selectbox("Social Engagement Level", ["Low", "Medium", "High"])
    stress = st.selectbox("Stress Levels", ["High", "Medium", "Low"])
    urban_rural = st.selectbox("Urban vs Rural Living", ["Urban", "Rural"])
    submit_button = st.form_submit_button("Predict")

# Encoding Function
def encode_inputs():
    try:
        categorical_features = [
            encoders["Gender"].transform([gender])[0],
            encoders["Physical Activity Level"].transform([physical_activity])[0],
            encoders["Smoking Status"].transform([smoking])[0],
            encoders["Alcohol Consumption"].transform([alcohol])[0],
            encoders["Diabetes"].transform([diabetes])[0],
            encoders["Hypertension"].transform([hypertension])[0],
            encoders["Cholesterol Level"].transform([cholesterol])[0],
            encoders["Family History of Alzheimers"].transform([family_history])[0],
            encoders["Depression Level"].transform([depression])[0],
            encoders["Sleep Quality"].transform([sleep_quality])[0],
            encoders["Air Pollution Exposure"].transform([air_pollution])[0],
            encoders["Genetic Risk Factor (APOE-ε4 allele)"].transform([genetic_risk])[0],
            encoders["Social Engagement Level"].transform([social_engagement])[0],
            encoders["Stress Levels"].transform([stress])[0],
            encoders["Urban vs Rural Living"].transform([urban_rural])[0]
        ]
        one_hot_features = np.concatenate([
            encoders["Dietary Habits"].transform([[dietary_habits]]).flatten(),
            encoders["Marital Status"].transform([[marital_status]]).flatten()
        ])
        features = np.concatenate([[age, education, bmi, cognitive_test], categorical_features, one_hot_features]).reshape(1, -1)
        return scaler.transform(features)
    except Exception as e:
        st.error(f"Encoding Error: {e}")
        return None

def predict_alzheimers():
    features = encode_inputs()
    if features is None:
        return None
    prediction = xgb_model.predict(features)
    return leout.inverse_transform([prediction[0]])[0]

if submit_button:
    result = predict_alzheimers()
    if result:
        st.success("🎯 **Prediction Result**")
        
        # Define color based on prediction result
        color = "grey" if result == "No" else "lightpink"
        
        styled_result = f"""
        <div style="padding: 15px; border-radius: 12px; background-color: {color}; color: black; 
                    text-align: center; font-size: 24px; font-weight: bold;">
            🧠 Alzheimer's Diagnosis: {result}
        </div>
        """
        st.warning(
            "⚠️ **Disclaimer:** This prediction is for informational purposes only. "
            "If you have any doubts or concerns, please consult a doctor for your safety and well-being."
        )

        st.markdown(styled_result, unsafe_allow_html=True)
