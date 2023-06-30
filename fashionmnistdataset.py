# -*- coding: utf-8 -*-
"""FashionMNISTDataSet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10-PYN1xJPC0fiIw5EcbJJ-05jKdcrC6O
"""

!pip install -U kaleido

"""### **Set Up the Notebook**"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""### **Import and Prepare the Data**"""

(trainImages, trainLabels), (testImages, testLabels) = keras.datasets.fashion_mnist.load_data()

classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boots']

# convert training and testing to one-hot encoded values (helps transfer categorical to numerical values)
trainLabels = to_categorical(trainLabels, 10)
testLabels = to_categorical(testLabels, 10)

pool_size = 2
model = keras.Sequential([
Conv2D(32, (3, 3), input_shape = (28, 28, 1), padding = 'same', activation = 'relu'),
BatchNormalization(),
MaxPooling2D(pool_size = pool_size),
Dropout(0.1),
Conv2D(64, (3,3), padding = 'same', activation = 'relu', use_bias = False),
MaxPooling2D(pool_size = pool_size),
Dropout(0.2),
Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
BatchNormalization(), # recently added
MaxPooling2D(pool_size = pool_size),
Dropout(0.25),
# Conv2D(128, (3,3), padding = 'same', activation = 'relu', use_bias = False), # recently added
# MaxPooling2D(pool_size = pool_size),
# Dropout(0.25), # recently added
# Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
# BatchNormalization(), # recently added
# MaxPooling2D(pool_size = pool_size),
# Dropout(0.25), # recently added
Flatten(),
Dense(512, activation = 'relu'),
BatchNormalization(),
Dense(10, activation = 'softmax', use_bias = False),

])

batch_size = 128
epochs = 10
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(trainImages, trainLabels, batch_size = batch_size, epochs = epochs, validation_split = (0.1), shuffle = True)
_, score = model.evaluate(testImages, testLabels, verbose = 1)
print(score)

"""### **Training and Testing Loss Graph**"""

training_loss = history.history['loss']
testing_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, color = 'g', label = 'Training Loss')
plt.plot(epoch_count, testing_loss, color = 'y', label = 'Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = "upper right")
plt.title("Training and Testing loss with different epochs")
plt.show()

"""### **Training and Testing Accuracy Graph**"""

# Training and Validation Accuracy Graph
training_accuracy = history.history['accuracy']
testing_accuracy = history.history['val_accuracy']

epoch_count = range(1, len(training_accuracy) + 1)
plt.plot(epoch_count, training_accuracy, color = 'g', label = 'Training Accuracy')
plt.plot(epoch_count, testing_accuracy, color = 'y', label = 'Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = "center right")
plt.title("Training and Testing accuracy with different epochs")
plt.show()

"""### **Confusion Matrix**"""

y_pred = model.predict(testImages)
# the problem that is happening here is that y_pred is not one-hot-encoded, since it is a categorical variable.
# confusion_matrix = confusion_matrix(y_true = testLabels, y_pred = y_pred)

y_pred_classes = np.argmax(y_pred, axis = 1) # max of each probability in axis = 1 for 10000 test images
y_test_classes = np.argmax(testLabels, axis = 1)
confusion_mtx = confusion_matrix(y_pred_classes, y_test_classes)
display = ConfusionMatrixDisplay(confusion_mtx, display_labels = classes)
display.plot(cmap = 'plasma', xticks_rotation = 'vertical')