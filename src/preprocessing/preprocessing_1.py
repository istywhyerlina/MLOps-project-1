import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder_path = {
    "gender" : "../../model/ohe_gender.pkl",
    "geography" : "../../model/ohe_geography.pkl"
}
dataset_path = "../../data/processed/data.csv"
dataset_ready_path = "../../data/processed/dataset.pkl"

sep = "\t"
index_col = "index"

list_columns = ["gender", "geography"]
column_categories = {
    "gender" : np.array(["Female", "Male"]).reshape(-1, 1),
    "geography" : np.array(["Germany", "Spain", "France"]).reshape(-1, 1)
}

def main():
    data = pd.read_csv(dataset_path, sep = "\t")
    #Features 'RowNumber', 'CustomerId', and 'Surname' are specific to each customer and can be dropped
    data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    #EstimatedSalary' displays a uniform distribution for both types of customers and can be dropped.
    #The categories in 'Tenure' and 'HasCrCard' have a similar churn rate and are deemed redundant. This can be confirmed from a chi-square test 
    features_drop = ['Tenure', 'HasCrCard', 'EstimatedSalary']
    data.drop(features_drop, axis=1)
    data.columns = list_columns
    data.index.name = None

    for column in data.columns:
        if(column != "target"):
            ohe = OneHotEncoder(sparse_output = False)
            ohe.fit(column_categories[column])
            temp = pd.DataFrame(
                ohe.transform(data[column].to_numpy().reshape(-1, 1)),
                columns = [column + "_" + name for name in ohe.categories_[0].tolist()]
            )
            data = pd.concat([data, temp], axis = 1)
            data.drop(columns = column, inplace = True)
            joblib.dump(ohe, encoder_path[column])
            print(f"One Hot Encoding data {column} completed.")

    data.to_pickle(dataset_ready_path)

if __name__ == "__main__":
    main()