import tensorflow as tf
import numpy as np

tf.config.optimizer.set_jit(True)
# Check if TensorFlow can access GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Generate large dummy data
x = np.random.random((100000, 100))  # 100,000 samples, 100 features each
y = np.random.random((100000, 1))    # 100,000 target values (single output)

# Define a deep and wide model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)  # Single output
])


# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(x, y, epochs=100, batch_size=64)
