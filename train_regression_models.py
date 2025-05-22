import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import json
import yaml
import numpy as np
# prétraitement des données
def preprocess_data(df):
    # Remplacer les valeurs manquantes dans les colonnes numériques par la médiane
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
# Encoder les colonnes catégorielles (textes) avec LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('missing')  
        df[col] = le.fit_transform(df[col])

    return df

df = pd.read_csv("data/iris_data.csv")

df = preprocess_data(df)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7]
        }
    }
}

best_model = None
best_model_name = None
best_rmse = float("inf")
results = {}
#Boucle (GridSearch)
for name, mp in models_params.items():
    print(f"GridSearch pour {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)  
    rmse = np.sqrt(mse) 

    results[name] = {
        'best_params': grid.best_params_,
        'test_rmse': rmse
    }
    print(f"{name} RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = grid.best_estimator_
        best_model_name = name
# Enregistrement des résultats dans un fichier JSON et YAML
with open("best_model_results.json", "w") as f_json:
    json.dump({"best_model": best_model_name, "best_rmse": best_rmse, "results": results}, f_json, indent=4)

with open("best_model_results.yaml", "w") as f_yaml:
    yaml.dump({
        "best_model": best_model_name,
        "best_rmse": round(best_rmse, 4),
        "results": results
    }, f_yaml)

print(f"\nMeilleur modèle: {best_model_name} avec RMSE = {best_rmse:.4f}")
