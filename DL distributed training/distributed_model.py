import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# Horovod: Adjust the learning rate based on the number of workers
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Horovod: Use the Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer)

# Compile the model
model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Horovod: Broadcast initial variable states from rank 0 to all other workers
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# Horovod: Set up the distributed training using `fit` function
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1 if hvd.rank() == 0 else 0  # Only rank 0 (the master worker) should display verbose output
)

# Save the trained model
if hvd.rank() == 0:
    model.save("distributed_model.h5")
    
# Load the model
model = tf.keras.models.load_model("distributed_model.h5")