import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# Load and prepare the MNIST dataset. Convert the sample data from integers to floating-point numbers
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Build a machine learning model

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


# The model returns a vector of log-odds scores, one for each class
predictions = model(x_train[:1]).numpy()
predictions

# The tf.nn.softmax function converts these log odds to probabilities for each class

tf.nn.softmax(predictions).numpy()

# Define a loss function for training.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This untrained model gives probabilities close to random 
loss_fn(y_train[:1], predictions).numpy()

# Configure and compile the model using Keras Model.compile
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train and evaluate the model - use Model.fit to adjust parameters and minimize loss
model.fit(x_train, y_train, epochs=5)

# Check model performance
model.evaluate(x_test,  y_test, verbose=2)

# Return a probability - wrap the trained model and attach softmax
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])


