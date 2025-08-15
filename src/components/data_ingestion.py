import pandas as pd
import os
import argparse

def ingest_data(output_path):
    raw_data_path = os.path.join("data", "reddit_preprocessing.csv")
    df = pd.read_csv(raw_data_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    df.to_csv(os.path.join(output_path, "raw_data.csv"), index=False)
    print(f"Data ingested and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    ingest_data(args.output_path)
