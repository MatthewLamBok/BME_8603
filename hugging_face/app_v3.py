import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model('abalone_model')

st.title('Abalone Sex Classification')

# Text inputs
length = st.text_input('Length', value=st.session_state.get('length', 0.5))
diameter = st.text_input('Diameter', value=st.session_state.get('diameter', 0.4))
height = st.text_input('Height', value=st.session_state.get('height', 0.1))

# Convert text inputs to float
length = float(length)
diameter = float(diameter)
height = float(height)

# Number inputs
whole_weight = st.number_input('Whole Weight', min_value=0.0, max_value=3.0, value=st.session_state.get('whole_weight', 1.5), step=0.1)
shucked_weight = st.number_input('Shucked Weight', min_value=0.0, max_value=2.0, value=st.session_state.get('shucked_weight', 0.6), step=0.1)
viscera_weight = st.number_input('Viscera Weight', min_value=0.0, max_value=1.0, value=st.session_state.get('viscera_weight', 0.3), step=0.1)
shell_weight = st.number_input('Shell Weight', min_value=0.0, max_value=1.0, value=st.session_state.get('shell_weight', 0.4), step=0.1)
rings = st.number_input('Rings', min_value=1, max_value=30, value=st.session_state.get('rings', 10), step=1)

# File Uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# When 'Predict Sex' is clicked
if st.button('Predict Sex'):
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_data = pd.read_csv(uploaded_file)
    else:
        # Create DataFrame from inputs
        input_data = pd.DataFrame([[length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings]],
                                  columns=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
    
    # Prediction
    prediction = predict_model(model, data=input_data)
    predicted_label = prediction['Label'][0]
    predicted_probability = prediction['Score'][0]
    
    # Show the prediction
    st.write(f"The predicted sex of the abalone is: {predicted_label} with a probability of {predicted_probability:.2f}")
