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
from tensorflow.keras.applications import InceptionV3, VGG19
import pandas as pd


IMG_HEIGHT = 150
IMG_WIDTH = 150

def create_results_table(history1, history2, scores1, scores2):
    results = {
        'Модель': ['InceptionV3', 'VGG19'],
        'Финальная точность (валидация)': [scores1[1], scores2[1]],
        'Финальные потери (валидация)': [scores1[0], scores2[0]],
        'Максимальная точность на обучении': [max(history1.history['accuracy']), 
                                            max(history2.history['accuracy'])],
        'Максимальная точность на валидации': [max(history1.history['val_accuracy']), 
                                             max(history2.history['val_accuracy'])],
        'Минимальные потери на валидации': [min(history1.history['val_loss']), 
                                          min(history2.history['val_loss'])],
        'Количество эпох': [len(history1.history['accuracy']), 
                           len(history2.history['accuracy'])]
    }
    
    df = pd.DataFrame(results)
    return df

def plot_comparison(history1, history2, model1_name, model2_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history1.history['accuracy'], label=f'{model1_name} - обучение', linewidth=2)
    ax1.plot(history1.history['val_accuracy'], label=f'{model1_name} - валидация', linewidth=2)
    ax1.plot(history2.history['accuracy'], label=f'{model2_name} - обучение', linewidth=2)
    ax1.plot(history2.history['val_accuracy'], label=f'{model2_name} - валидация', linewidth=2)
    ax1.set_title('Сравнение точности моделей')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history1.history['loss'], label=f'{model1_name} - обучение', linewidth=2)
    ax2.plot(history1.history['val_loss'], label=f'{model1_name} - валидация', linewidth=2)
    ax2.plot(history2.history['loss'], label=f'{model2_name} - обучение', linewidth=2)
    ax2.plot(history2.history['val_loss'], label=f'{model2_name} - валидация', linewidth=2)
    ax2.set_title('Сравнение потерь моделей')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(history1.history['accuracy'], label='Обучение', linewidth=2)
    ax3.plot(history1.history['val_accuracy'], label='Валидация', linewidth=2)
    ax3.set_title(f'Точность {model1_name}')
    ax3.set_xlabel('Эпоха')
    ax3.set_ylabel('Точность')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(history2.history['accuracy'], label='Обучение', linewidth=2)
    ax4.plot(history2.history['val_accuracy'], label='Валидация', linewidth=2)
    ax4.set_title(f'Точность {model2_name}')
    ax4.set_xlabel('Эпоха')
    ax4.set_ylabel('Точность')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def combine_histories(history1, history2):
    combined = {}
    for key in history1.history.keys():
        combined[key] = history1.history[key] + history2.history[key]
    return type('History', (), {'history': combined})()

def show_augmented_images(generator):
    plt.figure(figsize=(15, 10))
    
    images, labels = next(generator)
    
    class_names = {v: k for k, v in generator.class_indices.items()}
    
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        
        if len(labels[i].shape) > 0 and len(labels[i]) > 1:
            class_idx = np.argmax(labels[i])
        else:
            class_idx = int(labels[i])
            
        class_name = class_names.get(class_idx, f"Класс {class_idx}")
        
        plt.title(f'{class_name}')
        plt.axis('off')
    
    plt.suptitle('Примеры аугментированных изображений', fontsize=16)
    plt.tight_layout()
    plt.show()

def predict_image_multiclass(model, img_path, IMG_SIZE, class_names):
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions

def show_predictions_multiclass(model, data_dir, IMG_SIZE, class_names, samples_per_class=3):
    plt.figure(figsize=(15, 12))
    
    all_images = []
    true_labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)[:samples_per_class]]
            all_images.extend(images)
            true_labels.extend([class_name] * len(images))
    
    combined = list(zip(all_images, true_labels))
    random.shuffle(combined)
    all_images, true_labels = zip(*combined)
    
    for i in range(min(9, len(all_images))):
        plt.subplot(3, 3, i + 1)
        
        img = keras.preprocessing.image.load_img(all_images[i], target_size=IMG_SIZE)
        plt.imshow(img)
        
        predicted_class, confidence, _ = predict_image_multiclass(
            model, all_images[i], IMG_SIZE, class_names
        )
        
        color = 'green' if predicted_class == true_labels[i] else 'red'
        
        plt.title(f'Истина: {true_labels[i]}\nПредсказание: {predicted_class}\nУверенность: {confidence:.3f}',
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Примеры работы модели многоклассовой классификации', fontsize=16)
    plt.tight_layout()
    plt.show()

def create_inception_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_vgg_model():
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def fine_tune_model(model, base_model, model_name):
    base_model.trainable = True
    
    if model_name == 'InceptionV3':
        fine_tune_at = len(base_model.layers) - 20
    else:
        fine_tune_at = len(base_model.layers) - 4
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Размораживаем слои начиная с {fine_tune_at} для {model_name}")
    print(f"Общее количество слоев: {len(base_model.layers)}")
    print(f"Количество размороженных слоев: {len(base_model.layers) - fine_tune_at}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model