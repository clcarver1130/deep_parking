import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

# Import images

# Preprocess images
train_data =
train_X = train_X.reshape(-1, 28,28, 1)
train_X = train_X.astype('float32')
train_X = train_X / 255.

train_labels =
train_Y_one_hot = to_categorical(train_Y)

# Create and compile model
batch_size = 64
epochs = 20
num_classes = 10

model = Sequential()
model.add(Conv2D(10, kernel_size=3, activation='relu', input_shape=(img_rows, img_cols, 1)), padding='valid')
# # Second convolutional layer
# model.add(Conv2D(10, kernel_size=2, activation='relu')
model.add(Flatten())
model.add(Dense(3, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model to images
training = model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

model.evaluate(test_data, test_labels, epochs=3)

# Validate model
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()

# Save model
