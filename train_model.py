import os
import pandas as pd
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
        train_list.append((img_to_array(load_img('training_images/{}'.format(file), target_size=(48,48))).reshape(-1, 48,48, 1))/255)
    return train_list

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
    model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(48,48, 1)), padding='valid')
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model to images
    training = model.fit(img_list, label_list, validation_split=0.2, epochs=3)

    # Validate model
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.show()

    print('Saving model...')
    # Save model
    with open('cnn_model.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    print('Model saved in directory')

if __name__ == '__main__':
    imgs = import_images('training_images')
    labels = import_labels('training_labels.csv')
    train_model(imgs, labels)
