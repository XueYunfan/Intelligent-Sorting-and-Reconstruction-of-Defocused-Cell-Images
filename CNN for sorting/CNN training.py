import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

#load image names
#======================================
TRAIN_PATH = 'XXX/'
VALIDATION_PATH = 'XXX/'

train_names = os.listdir(TRAIN_PATH)
validation_names = os.listdir(VALIDATION_PATH)

train_file = []
validation_file = []

train_labels = []
validation_labels = []

random.seed(1)
random.shuffle(train_names)

for name in train_names:
	if len(train_file) < 7560: #define training set size
		train_file.append(TRAIN_PATH+name)
		if name[-5] == '1':
			train_labels.append(1)
		elif name[-5] == '0':
			train_labels.append(0)
		
for name in validation_names:
	validation_file.append(VALIDATION_PATH+name)
	if name[-5] == '1':
		validation_labels.append(1)
	elif name[-5] == '0':
		validation_labels.append(0)
		
#define tensorflow dataset
#======================================
def parse_function(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_resized = tf.image.resize(image_decoded,(134,134),method='bilinear')
	image_converted = tf.cast(image_resized, tf.float32)
	image_scaled = tf.image.per_image_standardization(image_converted)
	return image_scaled, label

train_filenames = tf.constant(train_file)
train_labels = tf.keras.utils.to_categorical(train_labels, 2, dtype='int32')
train_labels = tf.constant(train_labels)

validation_filenames = tf.constant(validation_file)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 2, dtype='int32')
validation_labels = tf.constant(validation_labels)

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.shuffle(len(train_file))

validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))

train_dataset = train_dataset.map(parse_function)
validation_dataset = validation_dataset.map(parse_function)

train_dataset = train_dataset.batch(4).repeat()
validation_dataset = validation_dataset.batch(4)

#define models
#====================================================
def ResNet(input_shape=(134,134,3)):
	
	base_model = tf.keras.applications.ResNet50V2(
		weights=None, include_top=False, input_shape=input_shape, pooling='avg')
	output = base_model.output
	predictions = tf.keras.layers.Dense(2, activation=None)(output)
	predictions = tf.keras.activations.softmax(predictions)
	
	model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
	
	return model

def CNN(input_shape=(134,134,3)):

	input_layer = Input(input_shape)

	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid',
		kernel_initializer = 'he_normal')(input_layer)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'valid', 
		kernel_initializer = 'he_normal')(conv1)
	pooling1 = MaxPooling2D(pool_size=(3, 3), strides=None, 
		padding='valid', data_format=None)(conv2)

	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
		kernel_initializer = 'he_normal')(pooling1)
	conv4 = Conv2D(128, 3, activation = 'relu', padding = 'valid', 
		kernel_initializer = 'he_normal')(conv3)
	pooling2 = MaxPooling2D(pool_size=(3, 3), strides=None, 
		padding='valid', data_format=None)(conv4)

	conv5 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
		kernel_initializer = 'he_normal')(pooling2)
	conv6 = Conv2D(256, 3, activation = 'relu', padding = 'valid', 
		kernel_initializer = 'he_normal')(conv5)
	pooling3 = GlobalAveragePooling2D()(conv6)
	dense1 = Dense(512, activation='relu')(pooling3)
	dense2 = Dense(2, activation=None)(dense1)
	output_layer = tf.keras.activations.softmax(dense2)
	
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

	return model

def transfer_learning_model():
	
	model_path = 'XXX.hdf5'
	model = load_model(model_path)
	
	return model

model = CNN()

for layer in model.layers:
   layer.trainable = True

sgd = tf.keras.optimizers.SGD(lr=0.0, momentum=0.9, decay=0, nesterov=False)
model.compile(optimizer=sgd,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
		
model.summary()

#training
#======================================
def step_decay(epoch):
    init_lrate = 0.001
    drop = 0.5
    lrate = init_lrate * pow(drop, (epoch//5))
    print('lrate = '+str(lrate))
    return lrate

lrate = LearningRateScheduler(step_decay)

filepath = 'XXX.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
		verbose=0, save_best_only=True, save_weights_only=False, 
		mode='min', period=1)

callback = [lrate,checkpoint]

hist = model.fit(train_dataset, validation_data=validation_dataset, 
	epochs=50, steps_per_epoch=1890, callbacks=callback)

#save training history
#======================================
def training_vis(hist):
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	acc = hist.history['accuracy']
	val_acc = hist.history['val_accuracy']
	
	plt.rc('font',family='Arial') 

	fig = plt.figure(figsize=(8,4))

	ax1 = fig.add_subplot(121)
	ax1.plot(loss,label='train_loss')
	ax1.plot(val_loss,label='val_loss')
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')
	ax1.set_title('Loss on Training and Validation Data')
	ax1.legend()

	ax2 = fig.add_subplot(122)
	ax2.plot(acc,label='train_accuracy')
	ax2.plot(val_acc,label='val_accuracy')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Accuracy')
	ax2.set_title('Accuracy on Training and Validation Data')
	ax2.legend()
	plt.tight_layout()
	plt.savefig('XXX.png', dpi=300)
	
	list1 = np.array([loss,acc,val_loss,val_acc])
	name1 = range(1,51)
	name2 = ['loss','accuracy','val_loss','val_accuracy']
	test = pd.DataFrame(columns=name1,index=name2,data=list1)
	test.to_csv('XXX.csv',encoding='gbk')

training_vis(hist)
