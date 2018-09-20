import os
import pandas as pd
import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

def import_images():
    filelist = os.listdir('training_images/')
    train_list = []
    for file in filelist:
        train_list.append(img_to_array(load_img('training_images/{}'.format(file), target_size=(48,48))))
    return train_list

def import_labels():
    df = pd.read_csv('training_labels.csv')
    labels = df['Label']
    labels_cat = to_categorical(labels)
    return labels_cat

def scale_images():
    train_imgs_scaled = []
    for img in train_list:
        train_imgs_scaled.append(img/255)
    return train_imgs_scaled

def train_model():


# Create and compile model
model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(48,48, 1)), padding='valid')
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model to images
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

# Validate model
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()

# Save model
