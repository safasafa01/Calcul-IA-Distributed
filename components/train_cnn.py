import tensorflow as tf
from kfp import dsl

@dsl.component
def train(data_dir: str, output_model: str):
    
    # Chargement des données d'entraînement et de validation
    train_data = tf.data.experimental.load(f"{data_dir}/train")
    val_data = tf.data.experimental.load(f"{data_dir}/val")

    # Création du modèle CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sortie binaire (fleur ou pas)
    ])

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(train_data, validation_data=val_data, epochs=5)

    # Sauvegarde du modèle entraîné
    model.save(output_model)
