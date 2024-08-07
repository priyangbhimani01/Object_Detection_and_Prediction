import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import  BatchNormalization, Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Set batch size
batch_size = 64

# Shuffle and batch the training data
train_dataset = train_dataset.shuffle(buffer_size=50000).batch(batch_size)

# Batch the test data
test_dataset = test_dataset.batch(batch_size)

# Verify the data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=20,
                    validation_data=test_dataset)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

y_pred = np.argmax(model.predict(x_test), axis=-1)

# Calculate accuracy for each class
class_accuracies = []
for i in range(10):
    class_indices = np.where(y_test == i)[0]
    class_accuracy = np.mean(y_pred[class_indices] == y_test[class_indices].flatten())
    class_accuracies.append(class_accuracy)
    print(f"Accuracy for class {class_names[i]}: {class_accuracy:.2f}")

plt.figure(figsize=(25, 10))
for i in range(250):
    plt.subplot(10, 25, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    plt.xlabel(class_names[y_pred[i]])
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

