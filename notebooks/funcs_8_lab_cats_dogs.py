import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import random


def show_augmented_images(generator):
    plt.figure(figsize=(15, 10))
    
    images, labels = next(generator)
    
    for i in range(8):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        class_name = 'Собака' if labels[i] == 1 else 'Кошка'
        plt.title(f'{class_name}')
        plt.axis('off')
    
    plt.suptitle('Примеры аугментированных изображений', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Визуализация истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Точность на обучении')
    ax1.plot(history.history['val_accuracy'], label='Точность на валидации')
    ax1.set_title('Точность модели')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Потери на обучении')
    ax2.plot(history.history['val_loss'], label='Потери на валидации')
    ax2.set_title('Потери модели')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def predict_image(model, img_path, IMG_SIZE):
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return 'Собака', prediction
    else:
        return 'Кошка', 1 - prediction
    
def show_predictions(model, dirs, IMG_SIZE):
    cats_dir = dirs[0]
    dogs_dir = dirs[1]

    plt.figure(figsize=(20, 12))
    
    cat_images = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)[:5]]
    dog_images = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)[:5]]
    
    all_images = cat_images + dog_images
    true_labels = ['Кошка'] * 5 + ['Собака'] * 5
    
    for i, (img_path, true_label) in enumerate(zip(all_images, true_labels)):
        plt.subplot(2, 5, i + 1)
        
        img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        plt.imshow(img)
        
        predicted_class, confidence = predict_image(model, img_path, IMG_SIZE)
        
        color = 'green' if predicted_class == true_label else 'red'
        
        plt.title(f'Истина: {true_label}\nПредсказание: {predicted_class}\nУверенность: {confidence:.3f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Примеры работы модели классификации', fontsize=16)
    plt.tight_layout()
    plt.show()