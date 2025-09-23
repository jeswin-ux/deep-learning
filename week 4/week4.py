import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], 28*28).astype("float32") / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype("float32") / 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential()
model.add(Dense(128, input_shape=(784,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax")) 


model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


model.fit(X_train, y_train,
          epochs=5,
          batch_size=128,
          validation_split=0.1,
          verbose=2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")
