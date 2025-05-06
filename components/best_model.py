import argparse
import os
import json

def select_best_model(results_path: str, output_json_path: str):
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)

        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[1]

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(best_model, f, indent=2)

        print(f"Meilleur modèle : {best_model['model_name']} avec accuracy {best_model['accuracy']:.2%}")
        print(f"Détails sauvegardés dans : {output_json_path}")
    except Exception as e:
        print(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', required=True)
    parser.add_argument('--output_json_path', required=True)
    args = parser.parse_args()
    select_best_model(args.results_path, args.output_json_path)
