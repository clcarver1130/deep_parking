import os
import pandas as pd
import numpy as np
import pickle
import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

print('Importing images...')
def import_images():
    filelist = os.listdir('training_images/')
    train_list = []
    for file in filelist:
        train_list.append(img_to_array(load_img('training_images/{}'.format(file), target_size=(100,100), color_mode='grayscale')))
    return np.asarray(train_list)

print('Importing labels...')
def import_labels(CSV):
    df = pd.read_csv(CSV)
    labels = df['Label']
    labels_cat = to_categorical(labels)
    return labels_cat

print('Fitting model...')
def train_model(img_list, label_list):
    # Create and compile model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(100,100,1),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model to images
    model.fit(img_list, label_list, validation_split=0.2, epochs=10)

    # print('Saving model...')
    # # Save model
    # with open('cnn_model.pkl', 'wb') as output:
    #     pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    # print('Model saved in directory')

if __name__ == '__main__':
    imgs = import_images()
    labels = import_labels('training_labels.csv')
    train_model(imgs, labels)
