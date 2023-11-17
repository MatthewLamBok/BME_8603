import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the model
model = load_model('./abalone_model')

st.title('Abalone Gender/Sex Classification')
st.markdown('<p style="font-size: 20px;">Matthew Lam 500959262 || EE8603 701E - Sel Topics: Computer Engr I - F2023</p>', unsafe_allow_html=True)

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


# Function to create either a slider or a number input based on user choice
def input_method(col, key, label, min_value, max_value, value, step):
    # Create two columns for the radio button and the input widget
    col1, col2 = col.columns([3, 1])  # Adjust the ratio as needed
    
    # In the second column, place the radio button
    with col2:
        method = st.radio("Select Input Method", ['Slider', 'Value'], key=f"method_{key}", label_visibility="collapsed")

    # In the first column, place the appropriate input widget based on the radio button selection
    with col1:
        if method == 'Slider':
            return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step)
        else:
            return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step)

# Create a container for inputs
input_container = st.container()

# Create input methods for each feature within the container
length = input_method(input_container, 'length', 'Length', 0.0, 1.0, st.session_state.get('length', 0.5), 0.01)
diameter = input_method(input_container, 'diameter', 'Diameter', 0.0, 1.0, st.session_state.get('diameter', 0.4), 0.01)
height = input_method(input_container, 'height', 'Height', 0.0, 1.0, st.session_state.get('height', 0.1), 0.01)
whole_weight = input_method(input_container, 'whole_weight', 'Whole Weight', 0.0, 3.0, st.session_state.get('whole_weight', 1.5), 0.01)
shucked_weight = input_method(input_container, 'shucked_weight', 'Shucked Weight', 0.0, 2.0, st.session_state.get('shucked_weight', 0.6), 0.01)
viscera_weight = input_method(input_container, 'viscera_weight', 'Viscera Weight', 0.0, 1.0, st.session_state.get('viscera_weight', 0.3), 0.01)
shell_weight = input_method(input_container, 'shell_weight', 'Shell Weight', 0.0, 1.0, st.session_state.get('shell_weight', 0.4), 0.01)
rings = input_method(input_container, 'rings', 'Rings', 1, 30, st.session_state.get('rings', 10), 1)



# Predict button
if st.button('Predict Sex'):
    input_data = pd.DataFrame([[length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings]],
                              columns=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])
    
    prediction = predict_model(model, data=input_data)
    predicted_label = prediction['prediction_label'][0]
    predicted_probability = prediction['prediction_score'][0]  
    
    st.write(f"The predicted sex of the abalone is: {predicted_label} with a probability of {predicted_probability:.2f}")



github_file_url = 'https://github.com/your_username/your_repository/blob/main/your_file.py'

st.markdown(f'Check out the [code on GitHub]({github_file_url})')
