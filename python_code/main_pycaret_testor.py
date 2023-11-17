#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from pycaret.classification import *
from pycaret.classification import *
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import function_script
from sklearn.tree import plot_tree
import xgboost 
import catboost
from pycaret.classification import setup, compare_models


# In[ ]:


from pycaret.classification import setup, compare_models
if __name__ == "__main__":
    #Parameter
    target_csv  = 'abalone'#['abalone','50_50', 'fill']
    ground_truth = 'Sex' #['Sex', 'Diabetes_binary']
    Prediction_label = 'All' #['M_F', 'All', 'M', 'F', 'I'] 
    split = '60_20_20' #['60_20_20', '80_10_10'] 
    display_bool = False
    rand_seed = 29
    num_of_models = 3
    
    #1.1 Load ============================================================
    if target_csv == 'fill':
        df = pd.read_csv('../data/diabetes_binary_health_indicators_BRFSS2015.csv')
    if target_csv == '50_50':
        df = pd.read_csv('../data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    if target_csv == 'abalone':
        df = pd.read_csv('../data/abalone.csv')
    
    
    #2.1 Set Label ============================================================        
    if target_csv == 'abalone' and Prediction_label == 'M_F':  #If classifing Male or Female
        df = df[df['Sex'] != 'I']
    if target_csv == 'abalone' and Prediction_label == 'M': #If classifing Male or not Male
        df['Sex'] = df['Sex'].replace(['F','I'],"N")
    if target_csv == 'abalone' and Prediction_label == 'F': #If classifing Female or not Female
        df['Sex'] = df['Sex'].replace(['M','I'],"N")
    if target_csv == 'abalone' and Prediction_label == 'I': #If classifing Infant or not Infant
        df['Sex'] = df['Sex'].replace(['M','F'],"N")
        
    #3.1 Machine learning ============================================================
    #3.2 Splitting 60/20/20
    if split =='60_20_20':
        train_val_data, test_data = train_test_split(df, test_size=0.2, random_state=rand_seed) #split 80/20
        val_split = 0.75 #split 80 to 60/20
    if split =='80_10_10':
        train_val_data, test_data = train_test_split(df, test_size=0.1, random_state=rand_seed) #split 90/10
        val_split = 0.8888 #split 80 to 60/20
    print(train_val_data.shape, test_data.shape)  
    
    # Define the range for polynomial degrees
    polynomial_degrees = [1, 2,3]
    
    # Seed value
    rand_seed = 10
    
    # List of normalization methods
    normalize_methods = ['zscore', 'minmax', 'maxabs', 'robust']
    
    # Options for preprocess, normalize, polynomial features, remove outliers, and feature selection
    preprocess_options = [True, False]
    normalize_options = [True, False]
    polynomial_features_options = [True, False]
    remove_outliers_options = [True, False]
    feature_selection_options = [True, False]
    
    # Dictionary to store results
    results = {}
    
    # Initialize variables to track the best accuracy and configuration
    best_accuracy = 0
    best_config = None
    
    # [Your existing loop setup...]
    
    # Looping through different configurations
    for preprocess in preprocess_options:
        for normalize in normalize_options:
            for poly_features in polynomial_features_options:
                for remove_outliers in remove_outliers_options:
                    for feature_selection in feature_selection_options:
                        for degree in polynomial_degrees:
                            for method in normalize_methods:
                                # [Your existing setup and model training code...]
                                # Setup your model
                                clf = setup(data=train_val_data, target=ground_truth, session_id=rand_seed,
                                            train_size=val_split, preprocess=preprocess, normalize=normalize,
                                            normalize_method=method, fix_imbalance=True, fix_imbalance_method='smote',
                                            polynomial_features=poly_features, polynomial_degree=degree,
                                            remove_outliers=remove_outliers, feature_selection=feature_selection,
                                            fold=2, use_gpu=False)
                                
                                # Evaluate the model and store results
                                best_model = compare_models(sort = 'AUC', n_select = num_of_models)
                                print('===================================')
                                model_accuracy = pull().iloc[0]['Accuracy']  # Assuming 'pull()' retrieves the latest model performance metrics
    
                                # Update best accuracy and configuration if current model is better
                                if model_accuracy > best_accuracy:
                                    best_accuracy = model_accuracy
                                    best_config = (preprocess, normalize, poly_features, remove_outliers, feature_selection, degree, method, rand_seed)
                                print(f"Best Configuration: Preprocess: {best_config[0]}, Normalize: {best_config[1]}, Poly Features: {best_config[2]}, Remove Outliers: {best_config[3]}, Feature Selection: {best_config[4]}, Degree: {best_config[5]}, Method: {best_config[6]}, Seed: {best_config[7]}")

                                del clf, best_model, model_accuracy
                                
    # Print the best configuration and its accuracy
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Configuration: Preprocess: {best_config[0]}, Normalize: {best_config[1]}, Poly Features: {best_config[2]}, Remove Outliers: {best_config[3]}, Feature Selection: {best_config[4]}, Degree: {best_config[5]}, Method: {best_config[6]}, Seed: {best_config[7]}")


# In[ ]:




