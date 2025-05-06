import tensorflow as tf
from kfp import dsl

@dsl.component
def predict(model_path: str, img_path: str):
    """
    Cette fonction charge le modèle et effectue une prédiction sur une nouvelle image.
    Si la prédiction est supérieure ou égale à 0.5, on considère que c'est une fleur.
    """
    # Chargement du modèle entraîné
    model = tf.keras.models.load_model(model_path)

    # Prétraitement de l'image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    return "Fleur" if prediction >= 0.5 else "Pas une fleur"
