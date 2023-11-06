import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import joblib


# Function to make predictions using the loaded model and new data
def make_predictions(loaded_model, new_data):
    predictions = loaded_model.predict(new_data)
    return predictions

# Replace 'model.pkl' with the path to your model file
def load_model(joblib_file):
    loaded_model = joblib.load(joblib_file)
    return loaded_model
# Replace 'best_abalone_model.pkl' with the path to your model file
model = load_model('abalone_model.pkl')

# Replace this with your actual new data
# Ensure that the column names match those that the model was trained on
new_data = {
    'Length': [0.455, 0.350, 0.530],
    'Diameter': [0.365, 0.265, 0.420],
    'Height': [0.095, 0.090, 0.135],
    'Whole_weight': [0.5140, 0.2255, 0.6770],
    'Shucked_weight': [0.2245, 0.0995, 0.2565],
    'Viscera_weight': [0.1010, 0.0485, 0.1415],
    'Shell_weight': [0.150, 0.070, 0.210],
    'Rings': [15, 7, 9],
}
df = pd.DataFrame(new_data)

# If you have new data in a CSV, you could load it like this:
# df = pd.read_csv('new_abalone_data.csv')

# Making predictions
predictions = make_predictions(model, df)

# Output the predictions
print("Predicted classes:", predictions)

# Interpret the predictions (based on your model's encoding of classes, if applicable)
class_mapping = {0: 'Male', 1: 'Female', 2: 'Infant'}
interpreted_predictions = [class_mapping[pred] if pred in class_mapping else pred for pred in predictions]

print("Interpreted classes:", interpreted_predictions)