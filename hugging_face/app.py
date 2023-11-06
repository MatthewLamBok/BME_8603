import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model('abalone_model')

# Define the UI layout
st.title('Abalone Sex Classification')


# Define sliders for feature input
length = st.slider('Length', min_value=0.0, max_value=1.0, step=0.01)
diameter = st.slider('Diameter', min_value=0.0, max_value=1.0, step=0.01)
height = st.slider('Height', min_value=0.0, max_value=1.0, step=0.01)
whole_weight = st.slider('Whole Weight', min_value=0.0, max_value=3.0, step=0.01)
shucked_weight = st.slider('Shucked Weight', min_value=0.0, max_value=2.0, step=0.01)
viscera_weight = st.slider('Viscera Weight', min_value=0.0, max_value=1.0, step=0.01)
shell_weight = st.slider('Shell Weight', min_value=0.0, max_value=1.0, step=0.01)
rings = st.slider('Rings', min_value=1, max_value=30, step=1)

# Create a button to make predictions
if st.button('Predict Sex'):
    # Create a DataFrame with the input features and correct column names
    input_data = pd.DataFrame([[length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings]],
                              columns=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
    
    # Make predictions
    prediction = predict_model(model, data=input_data)
    
    # Display prediction
    st.write(f"The predicted sex of the abalone is: {prediction['prediction_label'][0]}")
