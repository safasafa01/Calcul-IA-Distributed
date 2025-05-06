from kfp import dsl
from kfp.compiler import Compiler
from preprocess_DL import preprocess  
from train_cnn import train
from predict import predict

@dsl.pipeline(
    name='Image Processing Pipeline',
    description='Un pipeline pour le prétraitement des images, l\'entraînement du modèle et la prédiction.'
)
def image_pipeline():
    data_dir = 'chemin vers l image'  # L'utilisateur fournit ce chemin
    output_dir = '/chemin/vers/output'   # L'endroit où les images prétraitées seront stockées
    output_model = '/chemin/vers/sauvegarde_model'  # L'endroit où le modèle sera sauvegardé
    img_path = 'chemin vets l image'  # Image pour la prédiction

    # Étape 1 : Prétraitement des images
    preprocess_op = preprocess(data_dir=data_dir, output_dir=output_dir)

    # Étape 2 : Entraînement du modèle
    train_op = train(data_dir=output_dir, output_model=output_model)

    # Étape 3 : Prédiction avec le modèle
    predict_op = predict(model_path=output_model, img_path=img_path)

    # Lier les étapes
    predict_op.after(train_op)  

# Compilation du pipeline
pipeline_func = image_pipeline
pipeline_filename = 'pipe.yaml'  # Nom du fichier où tu veux sauvegarder le pipeline

# Compiler le pipeline
Compiler().compile(pipeline_func, pipeline_filename)

# Une fois compilé, tu peux le déployer sur un cluster ou exécuter des composants locaux.
print(f"Pipeline compilé et sauvegardé sous {pipeline_filename}")


# py components/pipeline_DL.py