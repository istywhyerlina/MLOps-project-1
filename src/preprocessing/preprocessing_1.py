import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
        data.to_csv(
                dataset_ready_path,
                sep = "\t",
                index= False)
        data.to_csv(
                "../../data/raw/data_prepros1.csv",
                sep = "\t",
                index= False)
        print("Preprocessing 1, delete columns Success.")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()