import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature
from sklearn.svm import SVC
'''from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
'''
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__": 
    mlflow.set_tracking_uri(uri = "http://localhost:5000") 
    mlflow.set_experiment("Churn Experiment") 
    data = pd.read_csv("../../data/processed/data.csv", sep = "\t")
    data['Geography'] = data.Geography.astype(float)
    data['Gender'] = data.Gender.astype(float)
    data['Tenure'] = data.Tenure.astype(float)
    data['NumOfProducts'] = data.NumOfProducts.astype(float)
    data['HasCrCard'] = data.HasCrCard.astype(float)
    data['IsActiveMember'] = data.IsActiveMember.astype(float)

    data['Exited'] = data.Exited.astype(str)
    y=data['Exited']
    X=data.drop('Exited',  axis='columns')
    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size = 0.2, random_state=17
    ) 
    
    #SMOTE (for imbalanced data)
    over = SMOTE(sampling_strategy='auto', random_state=33)
    X_train, y_train = over.fit_resample(X_train, y_train)
    params = {"C":2.0, "kernel":"rbf", "gamma":"scale"} 
    svc = SVC(**params)


    svc=svc.fit(X_train, y_train)


    y_pred = svc.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    
    with mlflow.start_run(run_name = "SVC run"): 
        mlflow.log_params(params) 
        mlflow.log_metric("accuracy", accuracy) 
        mlflow.set_tag("Training Info", "SVC model for churn data") 
        signature = infer_signature(X_train, svc.predict(X_train)) 
        model_info = mlflow.sklearn.log_model( 
            sk_model = svc, 
            artifact_path = "churn", 
            signature = signature, 
            input_example = X_train, 
            registered_model_name = "SVC best param", 
        )

