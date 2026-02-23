import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input,Flatten 
import numpy as np
import matplotlib.pyplot as plt
import random
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
model = Sequential([
    Input(shape=(28, 28), name = 'input_layer'),
    Flatten(),
    Dense(25, activation = 'relu',name = 'hidden_layer_1'),
    Dense(15, activation = 'relu',name = 'hidden_layer_2'),
    Dense(10, activation = 'linear',name = 'output_layer')
])
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 10, validation_split = 0.2)
model.evaluate(X_test, y_test)
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) 
plt.plot(epochs, loss, 'b-', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, 'r--', linewidth=2, label='CV Loss')
plt.title('Training vs Cross-Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Cross Entropy)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.subplot(1, 2, 2) 
plt.plot(epochs, acc, 'b-', linewidth=2, label='Training Accuracy')
plt.plot(epochs, val_acc, 'r--', linewidth=2, label='CV Accuracy')
plt.title('Training vs Cross-Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
random_idx = random.randint(0, len(X_test) - 1)
test_image = X_test[random_idx]
true_label = y_test[random_idx]
input_image = np.expand_dims(test_image, axis=0)
logits = model.predict(input_image)
predicted_label = np.argmax(logits)
plt.figure(figsize=(4, 4))
plt.imshow(test_image, cmap='gray')
plt.title(f"Predict: {predicted_label} | True: {true_label}", 
          color='green' if predicted_label == true_label else 'red', 
          fontsize=14, fontweight='bold')
plt.axis('off') 
plt.show()
model.save('mnist_model.h5')