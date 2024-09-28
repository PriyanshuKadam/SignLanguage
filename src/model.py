import os
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from data import generate_data

# Define the label mapping
label_map = { 
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 
    5: '6', 6: '7', 7: '8', 8: '9', 9: 'A',
    10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 
    15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 
    20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 
    25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 
    30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'
}

def build_model(input_shape, no_of_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(no_of_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model(model, model_file):
    model.save(model_file)  # Save entire model to .h5 file

def load_model(model_file):
    return models.load_model(model_file)  # Load entire model from .h5 file

def train_and_predict(x_train, x_test, x_holdout, img_width, img_height, batch_size, no_of_classes, uploaded_file=None, model_file='model.h5'):
    input_shape = (img_width, img_height, 3)

    # Check if the model already exists
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        train_generator = generate_data(x_train, img_width, img_height, batch_size)
        validation_generator = generate_data(x_test, img_width, img_height, batch_size)

        model = build_model(input_shape, no_of_classes)
        model.fit(train_generator, epochs=4, validation_data=validation_generator)
        save_model(model, model_file)

    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(img_width, img_height))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
        predicted_label = label_map[predicted_class]  # Map to label
        return predicted_label
