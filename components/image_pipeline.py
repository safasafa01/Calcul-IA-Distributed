def predict_image(image_path: str, model_path: str):
    """Prédit si l'image est une fleur ou non."""
    try:
        # Charger l'image et prétraiter
        img_array = preprocess_image(image_path)
        
        # Charger le modèle entraîné
        model = tf.keras.models.load_model(model_path)
        
        # Prédiction
        prediction = model.predict(img_array)
        
        # Afficher le résultat
        if prediction[0] > 0.5:
            print("L'image est une fleur.")
        else:
            print("L'image n'est pas une fleur.")
    
    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")
        raise
