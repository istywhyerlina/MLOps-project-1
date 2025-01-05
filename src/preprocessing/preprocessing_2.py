import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

dataset_path = "../../data/raw/data.csv"
dataset_ready_path = "../../data/processed/data.csv"

def main():
    try:
        data = pd.read_csv(dataset_path, sep = "\t")
        #Features 'RowNumber', 'CustomerId', and 'Surname' are specific to each customer and can be dropped
        data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
        #EstimatedSalary' displays a uniform distribution for both types of customers and can be dropped.
        #The categories in 'Tenure' and 'HasCrCard' have a similar churn rate and are deemed redundant. This can be confirmed from a chi-square test 
        features_drop = ['Tenure', 'HasCrCard', 'EstimatedSalary']
        data.drop(features_drop, axis=1)
        
        '''Our dataset contains two features that require encoding.
        For 'Gender', we will use scikit-learn's LabelEncoder() which maps each unique label to an integer (Male --> 1 and Female --> 0).
        For 'Geography', we will manually map values so that customers in Germany have the value of one (1) and all other customers (France and Spain) have zero (0). I chose this method since the churn rate for customers in the other two countries is almost equal and considerably lower than in Germany. Therefore, it makes sense to encode this feature so that it differentiates between German and non-German customers. Additionally, I tried one-hot encoding (get_dummies()) this feature, and the two new features for France and Spain had small feature importance.'''
        
        data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

        data['Geography'] = data['Geography'].map({
        'Germany': 1,
        'Spain': 0,
        'France': 0
        })
        
        scaler = StandardScaler()

        scl_columns = ['CreditScore', 'Age', 'Balance']
        data[scl_columns] = scaler.fit_transform(data[scl_columns])

        data.to_csv(
                dataset_ready_path,
                sep = "\t",
                index= False)
        print("Preprocessing 2, Label Encoding and Scaling Success.")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()