import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


activations = ['sigmoid', 'tanh', 'relu']

results = {}

for act in activations:
    print(f"\nUsing activation: {act}")


    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=act),
        Dense(64, activation=act),
        Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1,
              validation_data=(x_test, y_test))


    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy with {act}: {accuracy:.4f}")

    results[act] = accuracy

print("\nFinal Results:", results)g
