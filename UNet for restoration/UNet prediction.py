import tensorflow as tf
import os
import numpy as np
import random
from models import Unet, ResUnet
from PIL import Image
from losses import UNet_loss

IMAGE_PATH_TEST = 'XXX/'
image_names_test = os.listdir(IMAGE_PATH_TEST)
test_file = []

#random.seed(1)
#random.shuffle(image_names_test)

for name in image_names_test:
	test_file.append(IMAGE_PATH_TEST+name)

def img_res(img):
 
	image_contents = tf.io.read_file(img)
	image_decoded = tf.image.decode_jpeg(image_contents)
	#image_decoded = tf.image.central_crop(image_decoded, 0.72388)
	image_converted = tf.cast(image_decoded, tf.bfloat16)
	image_scaled = tf.image.per_image_standardization(image_converted)
	paddings = tf.constant([[92,92],[92,92],[0,0]])
	image_padded = tf.pad(image_scaled, paddings, 'SYMMETRIC')
	image_padded = tf.reshape(image_padded,(1,572,572,3))

	model = ResUnet()
	model.load_weights(
		'XXX.h5')
	prediction = model.predict(image_padded)

	prediction = np.reshape(prediction,(388,388,3))
	R = Image.fromarray(np.uint8(prediction[:,:,0]))
	G = Image.fromarray(np.uint8(prediction[:,:,1]))
	B = Image.fromarray(np.uint8(prediction[:,:,2]))
	fixed_img = Image.merge('RGB', (R, G, B))

	fixed_img.save('XXX/Fixed {}'.format(img[53:]), quality=95)
	fixed_img.show()

for img in test_file:
	print(img[53:])
	img_resave(img)
