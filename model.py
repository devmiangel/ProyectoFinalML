from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#rutas
data_dir = os.path.join('data', 'chest-xray-pneumonia')  
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')  

#medidas
IMG_SIZE = 150  
BATCH_SIZE = 32 
EPOCHS = 10

#daatos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Modelo 
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')
])

# modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
import json  

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],  
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

#modelo y métricas
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'pneumonia_cnn.keras'))

training_metrics = {
    'train_accuracy': history.history['accuracy'][-1],
    'train_loss': history.history['loss'][-1],
    'val_accuracy': history.history['val_accuracy'][-1],
    'val_loss': history.history['val_loss'][-1]
}

with open('models/training_metrics.json', 'w') as f:
    json.dump(training_metrics, f)

def predict_pneumonia(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Neumonía' if prediction[0][0] > 0.5 else 'Normal'
