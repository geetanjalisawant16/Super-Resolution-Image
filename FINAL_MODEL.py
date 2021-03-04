import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array, load_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array, load_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
root_dir = os.path.join(data_dir,"BSDS500/data/")
batch_size = 8
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  root_dir,
  validation_split=0.2,
  subset="training",
  seed=1337,
  image_size=(crop_size, crop_size),
  batch_size=batch_size,
  label_mode=None,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  root_dir,
  validation_split=0.2,
  subset="validation",
  seed=1337,
  image_size=(crop_size, crop_size),
  batch_size=batch_size,
  label_mode=None,
)
# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   break
# Scale from (0, 255) to (0, 1)
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image
train_ds = train_ds.map(scaling)
val_ds = val_ds.map(scaling)
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
def get_model():
  inputs = keras.Input(shape=(None, None, 1))
  x = layers.Conv2D(128, 5, activation='tanh', padding='same')(inputs)
  x = layers.Conv2D(64, 5, activation='tanh', padding='same')(x)
  x = layers.Conv2D(64, 3, activation='tanh', padding='same')(x)
  x = layers.Conv2D(32, 3, activation='tanh', padding='same')(x)
  x = layers.Conv2D(9, 3, activation='sigmoid', padding='same')(x)
  print(x)
  outputs = tf.nn.depth_to_space(x, upscale_factor)
  return keras.Model(inputs, outputs)
model = get_model()
model.summary()
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=loss_fn
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)
model.save('own_model.h5')
print("Saved model to disk")
# load_model('own_model.h5')
# print("Loaded model from disk")