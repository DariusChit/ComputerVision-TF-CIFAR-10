import tensorflow as tf
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

model = load_model('actually_model')

img_width, img_height = 32, 32
img = image.load_img('imgs/newcat.jpg', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

print(model.predict(img))

#airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
#order of class by index