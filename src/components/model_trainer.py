import pandas as pd
import yaml
import pickle
import os
import argparse
import mlflow
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score

def train_model(input_path, output_path):
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv(os.path.join(input_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_path, "test.csv"))
    with open(os.path.join(input_path, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    X_train = vectorizer.transform(train_df['clean_comment'])
    y_train = train_df[params['base']['target_column']]
    X_test = vectorizer.transform(test_df['clean_comment'])
    y_test = test_df[params['base']['target_column']]
    
    smote = SMOTE(random_state=params['resampling']['params']['random_state'])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    mlflow.set_tracking_uri("http://34.123.157.45:5000") # Use your static IP
    mlflow.set_experiment("Sentiment Analysis - DVC Pipeline")

    with mlflow.start_run(run_name="LGBM_Training_Pipeline"):
        mlflow.log_params(params)
        
        model = lgb.LGBMClassifier(**params['model']['params'])
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", f1)
        
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        
        mlflow.sklearn.log_model(model, "lightgbm_model")
        print(f"Model trained and saved to {output_path}. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    train_model(args.input_path, args.output_path)
