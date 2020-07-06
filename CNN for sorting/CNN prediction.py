import tensorflow as tf
import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve,auc,confusion_matrix

TEST_PATH = 'XXX/'

test_names = os.listdir(TEST_PATH)
test_file = []
test_labels = []

random.seed(1)
random.shuffle(test_names)

for name in test_names:
	if len(test_file) < 7560:
		test_file.append(TEST_PATH+name)
		if name[-5] == '1':
			test_labels.append(1)
		elif name[-5] == '0':
			test_labels.append(0)
		
def parse_function_test(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_resized = tf.image.resize(image_decoded,(134,134),method='bilinear')
	image_converted = tf.cast(image_resized, tf.float32)
	image_scaled = tf.image.per_image_standardization(image_converted)
	return image_scaled, label

test_filenames = tf.constant(test_file)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, 2, dtype='int32')
test_labels_onehot = tf.constant(test_labels_onehot)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels_onehot))
test_dataset = test_dataset.map(parse_function_test)
test_dataset = test_dataset.batch(16)

model_path = 'XXX.hdf5'
model = load_model(model_path)
predictions = model.predict(test_dataset)

def save_ROC(predictions):
	
	fpr, tpr, thresholds  =  roc_curve(test_labels, predictions[:,1], pos_label=1)
	roc_auc = auc(fpr, tpr)
	roc_auc = np.array([roc_auc])

	data1 = pd.DataFrame(fpr)
	data2 = pd.DataFrame(tpr)
	data3 = pd.DataFrame(thresholds)
	data4 = pd.DataFrame(roc_auc)

	writer = pd.ExcelWriter('XXX.xlsx')
	data1.to_excel(writer,startcol=0,index=False)
	data2.to_excel(writer,startcol=1,index=False)
	data3.to_excel(writer,startcol=2,index=False)
	data4.to_excel(writer,startcol=3,index=False)
	writer.save()

save_ROC(predictions)

predictions = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, predictions)
classes = ['Unfocused','Focused']

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_yticks(range(2))
	ax.set_yticklabels(classes)
	ax.set_xticks(range(2))
	ax.set_xticklabels(classes)
	plt.tick_params(labelsize=10)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title, fontsize=13)
	plt.colorbar()
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
		horizontalalignment="center",
		color="white" if cm[i, j] > thresh else "black",
		fontsize = 10)

	plt.tight_layout()
	plt.ylabel('True label',fontsize=13)
	plt.xlabel('Predicted label',fontsize=13)
	plt.tight_layout()
	plt.savefig('XXX.png', dpi=300)

plot_confusion_matrix(cm,classes)
