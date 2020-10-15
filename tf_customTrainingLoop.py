import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import keras
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

dataGen = ImageDataGenerator(rescale=1.0 / 255,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             fill_mode='nearest'
                             )

BATCH_SIZE = 32

train_generator = dataGen.flow_from_directory(
    'pokemon/dataset/train_images',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(50, 50),
)

val_generator = dataGen.flow_from_directory(

    directory='pokemon/dataset/validation_images',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(50, 50),
)


# As a list of BATCH_SIZE
def get_labels_from_batch(batch): return [i for i in batch[-1]]


# As a list of BATCH_SIZE
def get_values_from_batch(batch): return [i for i in batch[0]]


# Yields a sample of (x,y) from the batch
def xy_generator(batch):
    X = get_values_from_batch(batch)
    y = get_labels_from_batch(batch)

    for i in range(BATCH_SIZE):
        yield X[i], y[i]


# For testing the structure of the data
def identicality(l1, l2): return np.all(l1 == l2)


# A bit of test to see if everything is going okay with the generator.
# my_gen = xy_generator(batch)
# next_xy = next(my_gen)
#
# print(identicality(get_values_from_batch(batch)[0], next_xy[0]))
# print(identicality(get_labels_from_batch(batch)[0], next_xy[-1]))

model = keras.Sequential(

    [
        keras.Input((50, 50, 3)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(149, activation="softmax"),
    ]

)

epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.CategoricalAccuracy()

# Training Loop

for epoch in range(epochs):
    # Counter to end one epoch manually
    batch_number_t = 0
    print(f"\nStart of Training Epoch {epoch + 1}")
    # batch = train_generator.next()
    for batch_idx, (x_batch, y_batch) in enumerate(train_generator):
        # This is for recording all of the operations we gonna do in the fw prop
        with tf.GradientTape() as tape:
            # x_batch = np.expand_dims(BATCH_SIZE,x_batch, )
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)
        batch_number_t += 1
        if batch_number_t >= train_generator.samples / BATCH_SIZE:
            break

    train_acc = acc_metric.result()
    print(f"Accuracy over epoch {train_acc}")
    acc_metric.reset_states()

batch_number_v = 0
# Validation Loop
for batch_idx, (x_batch, y_batch) in enumerate(val_generator):
    y_pred = model(x_batch, training=True)
    acc_metric.update_state(y_batch, y_pred)
    batch_number_v += 1
    if batch_number_v >= val_generator.samples / BATCH_SIZE:
        break

train_acc = acc_metric.result()
print(f"Accuracy over Validation set:{train_acc}")
acc_metric.reset_states()
