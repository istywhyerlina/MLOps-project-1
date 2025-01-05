import mlflow
from sklearn import datasets
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__": 
    mlflow.set_tracking_uri(uri = "http://localhost:5000") 
    mlflow.set_experiment("Churn Experiment") 
    X, y = datasets.load_iris(return_X_y=True) 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size = 0.2
    ) 
    
    params = { "solver": "lbfgs",
               "max_iter": 10, 
               "multi_class": "auto", 
               "random_state": 2, 
    } 
    lr = LogisticRegression(**params) 
    lr.fit( 
        X_train, 
        y_train)
    
    y_pred = lr.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    
    with mlflow.start_run(run_name = "Initial run"): 
        mlflow.log_params(params) 
        mlflow.log_metric("accuracy", accuracy) 
        mlflow.set_tag("Training Info", "Basic LR model for iris data") 
        signature = infer_signature(X_train, lr.predict(X_train)) 
        model_info = mlflow.sklearn.log_model( 
            sk_model = lr, 
            artifact_path = "iris_model", 
            signature = signature, 
            input_example = X_train, 
            registered_model_name = "tracking-quickstart-mlprojects-docker", 
        )

