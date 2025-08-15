import pandas as pd
import yaml
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def transform_data(input_path, output_path):
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    df = pd.read_csv(os.path.join(input_path, "raw_data.csv"))
    df.dropna(subset=['clean_comment'], inplace=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=params['base']['random_state'])
    
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(params['vectorizer']['params']['ngram_range']),
        max_features=params['vectorizer']['params']['max_features']
    )
    
    os.makedirs(output_path, exist_ok=True)
    
    vectorizer.fit(train_df['clean_comment'])
    with open(os.path.join(output_path, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)
    
    print(f"Data transformed and artifacts saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    transform_data(args.input_path, args.output_path)
