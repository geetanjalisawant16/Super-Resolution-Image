# Reference link for code: https://keras.io/examples/vision/super_resolution_sub_pixel/
# Importing necessary libraries
import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.preprocessing as dir

# Loading training dataset(BSD500)
train_dataset= tf.keras.utils.get_file(
    "BSR",
    "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
     untar=True)
dir_root = os.path.join(train_dataset,"BSDS500/data/")

batch_size = 1
upscale_factor = 3
input_size = 128

# Training
train_ds = dir.image_dataset_from_directory(
  dir_root,
  validation_split=0.2,
  subset="training",
  seed=1337,
  image_size=(128, 128),
  batch_size=batch_size,
  label_mode=None,
)

# validation
val_ds = dir.image_dataset_from_directory(
  dir_root,
  validation_split=0.2,
  subset="validation",
  seed=1337,
  image_size=(128, 128),
  batch_size=batch_size,
  label_mode=None,
)

def scale_image(input_image):
    input_image = input_image / 255.0
    return input_image

train_ds = train_ds.map(scale_image)
val_ds = val_ds.map(scale_image)

def process_input(input, input_size, upscale_factor):
  input = tf.image.rgb_to_yuv(input)
  print(input)
  last_dimension_axis = len(input.shape) - 1
  print('dim:',last_dimension_axis)
  y, u, v = tf.split(input, 3, axis=last_dimension_axis)
  print(y)
  return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
  input = tf.image.rgb_to_yuv(input)
  last_dimension_axis = len(input.shape) - 1
  y, u, v = tf.split(input, 3, axis=last_dimension_axis)
  return y

train_ds = train_ds.map(
  lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
  )

val_ds = val_ds.map(
  lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)

# Building our model
def build_model():
  input = keras.Input(shape=(None, None, 1))

  conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input)
  pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)

  conv3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
  up2 = layers.UpSampling2D((2, 2))(conv3)
  output = layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(up2)

  return keras.Model(input, output)

# Call the model
model = build_model()
model.summary()

# Compile model
model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss=keras.losses.MeanSquaredError(),
  metrics=['accuracy']
)

# Fit the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100
)

# Save the model
model.save('OWN_MODEL.h5')
print("Saved model to disk")
