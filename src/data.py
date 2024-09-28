import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(train_folder):
    all_data = []
    for folder in os.listdir(train_folder):
        label_folder = os.path.join(train_folder, folder)
        onlyfiles = [{'label': folder, 'path': os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        all_data += onlyfiles

    data_df = pd.DataFrame(all_data)
    x_train, x_holdout = train_test_split(data_df, test_size=0.10, random_state=42, stratify=data_df[['label']])
    x_train, x_test = train_test_split(x_train, test_size=0.20, random_state=42, stratify=x_train[['label']])

    img_width, img_height = 64, 64
    batch_size = 128
    no_of_classes = len(data_df['label'].unique())

    return x_train, x_test, x_holdout, no_of_classes, img_width, img_height, batch_size

def generate_data(dataframe, img_width, img_height, batch_size, x_col='path', y_col='label'):
    datagen = ImageDataGenerator(rescale=1/255.0)
    return datagen.flow_from_dataframe(
        dataframe=dataframe, x_col=x_col, y_col=y_col,
        target_size=(img_width, img_height), class_mode='categorical',
        batch_size=batch_size, shuffle=False
    )
