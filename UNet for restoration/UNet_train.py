import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from losses import *
from models import ResUnet, Unet

def training_save(hist, img_name):
	
	#This function saves the training history as .csv format
	loss = hist.history['loss']
	mae = hist.history['SSIM']
	val_loss = hist.history['val_loss']
	val_mae = hist.history['val_SSIM']
	
	list1 = np.array([loss,mae,val_loss,val_mae])
	name1 = range(1,41)
	name2 = ['loss','SSIM','val_loss','val_SSIM']
	test = pd.DataFrame(columns=name1,index=name2,data=list1)
	test.to_csv('XXX/trainhis {} ResUnet MaxPooling Nadam SSIM InsNorm 40ep.csv'.format(img_name),encoding='gbk')

def parse_function(filename, labelname):
	
	#This is the parse function for preprocessing of images and labels
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

def training(img, img_label, step):

	IMAGE_PATH_TRAIN = 'XXX/{}/train/'.format(img)
	IMAGE_PATH_VALIDATION = 'XXX/{}/validation/'.format(img)
	LABLE_PATH_TRAIN = 'XXX/{}/train/'.format(img_label)
	LABLE_PATH_VALIDATION = 'XXX/{}/validation/'.format(img_label)

	image_names_train = os.listdir(IMAGE_PATH_TRAIN)
	image_names_validation = os.listdir(IMAGE_PATH_VALIDATION)
	label_names_train = os.listdir(LABLE_PATH_TRAIN)
	label_names_validation = os.listdir(LABLE_PATH_VALIDATION)

	train_file = []
	validation_file = []

	train_labels = []
	validation_labels = []

	for name in image_names_train:
		train_file.append(IMAGE_PATH_TRAIN+name)

	for name in image_names_validation:
		validation_file.append(IMAGE_PATH_VALIDATION+name)

	for name in label_names_train:
		train_labels.append(LABLE_PATH_TRAIN+name)

	for name in label_names_validation:
		validation_labels.append(LABLE_PATH_VALIDATION+name)

	train_filenames = tf.constant(train_file)
	train_labels = tf.constant(train_labels)

	validation_filenames = tf.constant(validation_file)
	validation_labels = tf.constant(validation_labels)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
	train_dataset = train_dataset.shuffle(len(train_file))

	validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))

	train_dataset = train_dataset.map(parse_function)
	validation_dataset = validation_dataset.map(parse_function)

	train_dataset = train_dataset.batch(2).repeat()
	validation_dataset = validation_dataset.batch(2)

	model = ResUnet()
	model.compile(loss=SSIM_loss, optimizer='Nadam', metrics=[SSIM])
	filepath = 'XXX/{} ResUnet MaxPooling Nadam SSIM InsNorm 40ep.h5'.format(img)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_SSIM', 
			verbose=0, save_best_only=True, save_weights_only=True, 
			mode='max', period=1)
	callback = [checkpoint]
	hist = model.fit(train_dataset, validation_data=validation_dataset, 
			epochs=40, steps_per_epoch=step, callbacks=callback)

	return hist

#images of different focusing conditions were put separately in different file folders. The imglist contains folders names.
imglist=['10+10','10-10','10+20','10-20']
labelist=['10-0','10-0','10-0','10-0']
train_steps=[678,678,678,678]

for x,y,z in zip(imglist,labelist,train_steps):
	hist = training(x,y,z)
	training_vis(hist,x)
