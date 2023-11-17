import pandas as pd

target_csv  = '50_50'
#target_csv  = 'fill'

if target_csv == 'fill':
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
if target_csv == '50_50':
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
# Display the first few rows of the DataFrame to ensure it loaded correctly
print(df.head())

print(df.isnull().sum())

# Count the number of instances for each class
class_counts = df['Diabetes_binary'].value_counts()
print (class_counts)

df_info = df.drop('Diabetes_binary', axis=1)
ground_truth = df['Diabetes_binary']

# Calculating the correlation matrix
correlation_matrix = df_info.corr()


print(correlation_matrix)



