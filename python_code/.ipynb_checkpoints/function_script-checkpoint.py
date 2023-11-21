
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from pycaret.classification import *
import seaborn as sns
import os

def info_display(data):
    print('>>First 5 Data Points')   
    print(data.head(), '\n\n')
    
    print('>>Data Information')    
    print(data.describe(), '\n\n')
    
    print('>>Missing Data')
    print(data.isnull().sum(), '\n\n')
    
    print('>>Univariate Analysis: Sex distribution')
    print(data['Sex'].value_counts(), '\n\n')
    sns.countplot(x='Sex', data=data)
    plt.title('Distribution of Sex in Abalones')
    plt.show()
    
    print('>>Bivariate Analysis: Relationships between features and the Sex')
    sns.pairplot(data, hue='Sex')
    plt.title('Pairwise Relationships by Sex')
    plt.show()

    
    # Encoding the 'Sex' attribute
    sex_encoding = {'F': 0, 'I': 1, 'M': 2}
    data_encode['Sex_encoded'] = data['Sex'].map(sex_encoding)
    
    # Calculate the correlation matrix
    correlation_matrix = data_encode.corr()
    
    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Abalone Dataset')
    plt.show()






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