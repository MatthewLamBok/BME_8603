import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model('abalone_model')

st.title('Abalone Sex Classification')

# Define initial state for sliders if not already set
if 'length' not in st.session_state:
    st.session_state['length'] = 0.5
if 'diameter' not in st.session_state:
    st.session_state['diameter'] = 0.4
if 'height' not in st.session_state:
    st.session_state['height'] = 0.1
if 'whole_weight' not in st.session_state:
    st.session_state['whole_weight'] = 1.5
if 'shucked_weight' not in st.session_state:
    st.session_state['shucked_weight'] = 0.6
if 'viscera_weight' not in st.session_state:
    st.session_state['viscera_weight'] = 0.3
if 'shell_weight' not in st.session_state:
    st.session_state['shell_weight'] = 0.4
if 'rings' not in st.session_state:
    st.session_state['rings'] = 10

# Example values for Male, Female, and Infant
example_values = {
    'Male': [0.475,0.37,0.125,0.5095,0.2165,0.1125,0.165,9],
    'Female': [0.53,0.42,0.135,0.677,0.2565, 0.1415, 0.21,9],
    'Infant': [0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7]
}

# Buttons to set example values
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Example Male'):
        for key, value in zip(['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'], example_values['Male']):
            st.session_state[key] = value

with col2:
    if st.button('Example Female'):
        for key, value in zip(['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'], example_values['Female']):
            st.session_state[key] = value

with col3:
    if st.button('Example Infant'):
        for key, value in zip(['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'], example_values['Infant']):
            st.session_state[key] = value

# Sliders for feature input with session_state
length = st.slider('Length', min_value=0.0, max_value=1.0, value=st.session_state['length'], key='length')
diameter = st.slider('Diameter', min_value=0.0, max_value=1.0, value=st.session_state['diameter'], key='diameter')
height = st.slider('Height', min_value=0.0, max_value=1.0, value=st.session_state['height'], key='height')
whole_weight = st.slider('Whole Weight', min_value=0.0, max_value=3.0, value=st.session_state['whole_weight'], key='whole_weight')
shucked_weight = st.slider('Shucked Weight', min_value=0.0, max_value=2.0, value=st.session_state['shucked_weight'], key='shucked_weight')
viscera_weight = st.slider('Viscera Weight', min_value=0.0, max_value=1.0, value=st.session_state['viscera_weight'], key='viscera_weight')
shell_weight = st.slider('Shell Weight', min_value=0.0, max_value=1.0, value=st.session_state['shell_weight'], key='shell_weight')
rings = st.slider('Rings', min_value=1, max_value=30, value=st.session_state['rings'], key='rings')




# Predict button
if st.button('Predict Sex'):
   
    input_data = pd.DataFrame([[length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings]],
                              columns=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
    

    prediction = predict_model(model, data=input_data)
    
    predicted_label = prediction['prediction_label'][0]
    predicted_probability = prediction['prediction_score'][0]  

    st.write(f"The predicted sex of the abalone is: {predicted_label} with a probability of {predicted_probability:.2f}")
