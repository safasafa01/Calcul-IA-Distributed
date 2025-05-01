import json

def select_best_model(evaluation_path, model_dir, output_dir):
    # Chargement des résultats d'évaluation
    with open(evaluation_path, 'r') as f:
        evaluation_results = json.load(f)
    
    # Initialisation des variables pour garder une trace du meilleur modèle
    best_model = None
    best_accuracy = -1

    # Parcourir chaque modèle dans le dictionnaire
    for model, metrics in evaluation_results.items():
        accuracy = metrics['accuracy']  # Accéder à la précision du modèle
        print(f"Modèle : {model} | Accuracy : {accuracy}%")
        
        # Comparer la précision pour trouver le meilleur modèle
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nLe meilleur modèle est : {best_model} avec une précision de {best_accuracy}%")

# Exemple d'appel à la fonction
select_best_model(
    evaluation_path='./evaluation/evaluation_results.json',
    model_dir='./models',
    output_dir='./best_model'
)
