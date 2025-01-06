import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

'''from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

    
    params={"bootstrap": True,
      "criterion": "gini",
      "max_depth": 6,
      "min_samples_leaf": 5,
      "min_samples_split": 2,
      "n_estimators": 100}

    rf = RandomForestClassifier(**params)
    
    rf = rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    
    with mlflow.start_run(run_name = "RF run"): 
        mlflow.log_params(params) 
        mlflow.log_metric("accuracy", accuracy) 
        mlflow.set_tag("Training Info", "RF model for churn data") 
        signature = infer_signature(X_train, rf.predict(X_train)) 
        model_info = mlflow.sklearn.log_model( 
            sk_model = rf, 
            artifact_path = "churn", 
            signature = signature, 
            input_example = X_train, 
            registered_model_name = "tracking-quickstart-mlprojects-docker", 
        )

