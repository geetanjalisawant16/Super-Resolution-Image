import matplotlib.pyplot as plt
from keras_preprocessing.image import img_to_array, load_img, save_img
import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

model = load_model('OWN_MODEL.h5')

upscale_factor = 3

test_images = sorted(
    [
        os.path.join('GSRM_LR_IMAGES', fname)
        for fname in os.listdir('GSRM_LR_IMAGES/')
        if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg") or fname.endswith(".gif")
    ]
)

def predicted_output_image(model, input_image):
    # Convert into ycbcr
    ycbcr = input_image.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0
    input = np.expand_dims(y, axis=0)

    # Predict the iamge based on input image
    out = model.predict(input)
    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

# To plot output image
def plot_output_image(image, image_prefix, title):
    image_array = img_to_array(image)
    image_array = image_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    fig, ax = plt.subplots()
    ax.imshow(image_array[::-1], origin="lower")

    # To save and show image
    save = os.path.join("GSRM_SR_IMAGES/")
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    plt.savefig(save + str(image_prefix) + "-" + title + ".png")
    # plt.title(title)
    # plt.show()

for index, image_test in enumerate(test_images):
    input_image = load_img(image_test) # Input image
    higher_img = predicted_output_image(model, input_image) # Predicted output Image
    plot_output_image(higher_img, index, "Higher_Resolution_Image") # To save and plot predicted output image
