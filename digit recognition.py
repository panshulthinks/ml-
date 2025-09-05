import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}"   )
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

#preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

#reshaping to add channel dimension
#this is done because cnn requires 4d inputs
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"Reshaped training data shape: {x_train.shape}")


#build the CNN model
model = Sequential([
    # the layer conv2d detects patterns
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    #maxpooling2d reduces size, keeps imprtant features
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #converts 2d to 1d(flattening)
    Flatten(),
    #dense is fully connected layer
    Dense(128, activation='relu'),
    #dropout is used to prevent overfiting
    Dropout(0.5),
    Dense(10, activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


#train the model
history = model.fit(
    x_train, y_train,
    epochs=1,
    validation_split=0.1,
    batch_size=64
)


#evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

#visualize training
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#make predictions
predictions = model.predict(x_test[:5])

for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Prediction: {np.argmax(predictions[i])}, True: {y_test[i]}")
    plt.show()