import subprocess
import os
import argparse

def run_pipeline(data_path, work_dir):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    # Chemins absolus
    data_path = os.path.abspath(data_path)
    work_dir = os.path.abspath(work_dir)
    preprocessed_dir = os.path.join(work_dir, "preprocessed")
    models_dir = os.path.join(work_dir, "models")

    # Chemin vers le dossier components
    components_dir = os.path.join(os.path.dirname(__file__), "components")

    # Prétraitement
    subprocess.run([
        "python", os.path.join(components_dir, "preprocess.py"),
        "--data_path", data_path,
        "--output_dir", preprocessed_dir
    ], check=True)

    # Entraînement
    subprocess.run([
        "python", os.path.join(components_dir, "train.py"),
        "--X_train_path", os.path.join(preprocessed_dir, "X_train.csv"),
        "--y_train_path", os.path.join(preprocessed_dir, "y_train.csv"),
        "--output_dir", models_dir
    ], check=True)

    # Sélection du meilleur modèle
    subprocess.run([
        "python", os.path.join(components_dir, "best_model.py"),
        "--results_path", os.path.join(models_dir, "gridsearch_results.json"),
        "--output_json_path", os.path.join(models_dir, "best_model.json")
    ], check=True)

    print("\nPipeline exécuté avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--work_dir', default="./pipeline_output")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    run_pipeline(args.data_path, args.work_dir)

#py pipeline_ML.py --data_path ./data/iris_data.csv