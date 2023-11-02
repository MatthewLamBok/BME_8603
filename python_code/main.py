import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from pycaret.classification import *

def normalize_data(df, ground_truth_col, normalization_type='z-score'):

    features = df.drop(ground_truth_col, axis=1)
    ground_truth = df[ground_truth_col]

    # Apply normalization
    if normalization_type == 'z-score':
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    elif normalization_type == 'min-max':
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        raise ValueError("Invalid normalization type. Use 'z-score' or 'min-max'.")

    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_df[ground_truth_col] = ground_truth

    return normalized_df


def balance_classes_with_smote(df, ground_truth_col, random_state=None, plot_distribution=True):
    X = df.drop(ground_truth_col, axis=1)
    y = df[ground_truth_col]
    
    # Check and plot the distribution of classes before SMOTE
    if plot_distribution:
        print("Class distribution before SMOTE:")
        print(y.value_counts())
        y.value_counts().plot(kind='bar', title='Class Distribution Before SMOTE')
        plt.show()

    # Apply SMOTE
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Convert the result back to a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[ground_truth_col] = y_resampled

    # Check and plot the distribution of classes after SMOTE
    if plot_distribution:
        print("\nClass distribution after SMOTE:")
        print(y_resampled.value_counts())
        y_resampled.value_counts().plot(kind='bar', title='Class Distribution After SMOTE')
        plt.show()
        
    return df_resampled

if __name__ == "__main__":
    #Parameter
    target_csv  = 'fill'#'50_50' 
    normalization_bool = True
    normalization_type =  'z-score' #'min-max'
    
    display_bool = False
    rand_seed = 42
    #1.1 Load
    if target_csv == 'fill':
        df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    if target_csv == '50_50':
        df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

    #2.1 Preprocessing
    if normalization_bool== True:
        df = normalize_data(df, ground_truth_col= 'Diabetes_binary', normalization_type= normalization_type)

    if target_csv == 'fill':
        df = balance_classes_with_smote(df,  ground_truth_col= 'Diabetes_binary', random_state=None, plot_distribution=display_bool)

    #3.1 Machine learning
    #3.2 Splitting 60/20/20
    train_val_data, test_data = train_test_split(df, test_size=0.2, random_state=rand_seed)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=rand_seed)
 
    #3.2 Model Creation
    exp = ClassificationExperiment()
    clf1 = exp.setup(train_data, target='Diabetes_binary',  use_gpu=False, session_id=rand_seed)
    print(clf1)

    best = compare_models()
    exp.compare_models()
    # Create and tune a model on the training data
    model = create_model('rf')  # 'rf' is the ID for Random Forest
    tuned_model = tune_model(model)

    # Evaluate the model on the validation data
    val_predictions = predict_model(tuned_model, data=val_data)
    print(val_predictions)

    # Finalize the model on the combined train and validation sets
    final_model = finalize_model(tuned_model)

    # Save the final model
    save_model(final_model, 'final_diabetes_model')
    # Evaluate the final model on the test data
    test_predictions = predict_model(final_model, data=test_data)
    print(test_predictions)
