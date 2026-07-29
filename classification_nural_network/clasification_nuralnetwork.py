import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

img_index = 9
image = x_train[img_index]
#print("ImageLabel: ", y_train[img_index])
#plt.imshow(image)
#plt.show()


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ]
)

#print(model.summary())

x_valid , x_train = x_train[:5000] / 255.0, x_train[5000:] / 255.0
y_valid, y_train = y_train[:5000], y_train[5000:]

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=30,
    validation_data=(x_valid, y_valid)
)

new = x_test[:5]/255.0
predictions = model.predict(new)
print(predictions)

classes = np.argmax(predictions, axis=1)
print("Predicted:", classes)
print("Actual   :", y_test[:5])