import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def treinar_modelo(dataset_path, img_size=(224, 224), batch_size=32, epochs=5):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'modelo_mobilenetv2.h5'))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)

    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes

    report = classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys()), output_dict=True)
    matriz = confusion_matrix(y_true, y_pred_classes)

    return {
        'history': history.history,
        'report': report,
        'confusion_matrix': matriz
    }
