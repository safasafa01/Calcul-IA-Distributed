import pandas as pd
import json
import argparse
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def train(X_train_path: str, y_train_path: str, output_dir: str):
    try:
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).values.ravel()

        models = {
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            },
            'SVM': {
                'model': SVC(),
                'params': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1]}
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1]}
            },
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, solver='liblinear'),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
            }
        }

        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for name, config in models.items():
            print(f"\nEntraînement {name}...")
            grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            results[name] = {
                'accuracy': grid.best_score_,
                'best_params': grid.best_params_,
                'model_name': name
            }

        with open(f"{output_dir}/gridsearch_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nEntraînement terminé. Résultats dans : {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train_path', required=True)
    parser.add_argument('--y_train_path', required=True)
    parser.add_argument('--output_dir', default="./models")
    args = parser.parse_args()
    train(args.X_train_path, args.y_train_path, args.output_dir)
