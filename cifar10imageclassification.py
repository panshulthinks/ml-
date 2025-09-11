import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import pandas as pd

# load the dataset
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}"   )
# preprocess the data
x_train = x_train/255.0
x_test = x_test/255.0
# lets reshape the data
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
print(f"Reshaped training data shape: {x_train.shape}")

df = pd.DataFrame(y_train, columns=['label'])
df.head()

#visualise some images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.xticks([]); plt.yticks([])
    plt.imshow(x_train[i])
    label_index = y_train[i]
    plt.xlabel(class_names[int(label_index)])
plt.show()


# build the CNN model
model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(
    x_train, y_train,
    epochs=1,
    validation_split=0.1,
    batch_size=64
)

# evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

#plotting training results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# make predictions
# Pick a random test image
index = 10
pred = model.predict(x_test[index].reshape(1,32,32,3))
print("Predicted:", class_names[np.argmax(pred)])
print("Actual:", class_names[int(y_test[index])])

