import os
import tensorflow as tf
from kfp import dsl

@dsl.component
def preprocess(data_dir: str, output_dir: str):
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Générateur pour les données d'entraînement
    train_generator = datagen.flow_from_directory(
        data_dir,  # Chemin dynamique
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    # Générateur pour les données de validation
    val_generator = datagen.flow_from_directory(
        data_dir,  
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Sauvegarde des images prétraitées
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    for i, (images, labels) in enumerate(train_generator):
        for j in range(images.shape[0]):
            image = images[j]
            image_name = f'train_image_{i * train_generator.batch_size + j}.png'
            tf.keras.preprocessing.image.save_img(os.path.join(output_dir, 'train', image_name), image)

        if i == train_generator.samples // train_generator.batch_size:
            break  # Stop après avoir traité tous les lots

    for i, (images, labels) in enumerate(val_generator):
        for j in range(images.shape[0]):
            image = images[j]
            image_name = f'val_image_{i * val_generator.batch_size + j}.png'
            tf.keras.preprocessing.image.save_img(os.path.join(output_dir, 'val', image_name), image)

        if i == val_generator.samples // val_generator.batch_size:
            break  # Stop après avoir traité tous les lots
