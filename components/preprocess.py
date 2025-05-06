import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import json

def preprocess(data_path: str, output_dir: str):
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Format non supporté. Utilisez CSV, JSON ou Excel.")

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col]).fillna(df.mean(numeric_only=True))
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        metadata = {
            'target_column': target_col,
            'features': list(X.columns),
            'shape_original': df.shape,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        with open(f"{output_dir}/mlpipeline-ui-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Prétraitement terminé. Résultats : {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', default="./out")
    args = parser.parse_args()
    preprocess(args.data_path, args.output_dir)
