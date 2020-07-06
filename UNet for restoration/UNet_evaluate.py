import tensorflow as tf
import os
import numpy as np
from losses import *
from models import U_NET,Origin_U_NET

IMAGE_PATH_TEST = 'XXX/'
LABLE_PATH_TEST = 'XXX/'

image_names_test = os.listdir(IMAGE_PATH_TEST)
label_names_test = os.listdir(LABLE_PATH_TEST)

test_file = []
test_label = []

for name in image_names_test:
	test_file.append(IMAGE_PATH_TEST+name)

for name in label_names_test:
	test_label.append(LABLE_PATH_TEST+name)
		
def parse_function(filename, labelname):
	
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.bfloat16)
	image_scaled = tf.image.per_image_standardization(image_converted)
	paddings = tf.constant([[92,92],[92,92],[0,0]])
	image_padded = tf.pad(image_scaled, paddings, 'SYMMETRIC')
	
	label_contents = tf.io.read_file(labelname)
	label_decoded = tf.image.decode_jpeg(label_contents)
	label_converted = tf.cast(label_decoded, tf.bfloat16)
	
	return image_padded, label_converted

test_filenames = tf.constant(test_file)
test_labels = tf.constant(test_label)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(parse_function)
test_dataset = test_dataset.batch(8)

model = U_NET()
model.compile(loss=SSIM_loss, optimizer='Nadam', metrics=[SSIM])
model.load_weights(
	'XXX.h5')
model.evaluate(test_dataset)
