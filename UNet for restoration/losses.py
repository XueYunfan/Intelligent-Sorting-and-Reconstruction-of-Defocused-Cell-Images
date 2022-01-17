import tensorflow.keras.backend as K
import tensorflow as tf

def SSIM_loss(y_true, y_pred):
	loss = 1-tf.image.ssim(y_true,y_pred,max_val=255)
	return loss
	
def SSIM(y_true, y_pred):
	loss = tf.image.ssim(y_true,y_pred,max_val=255)
	return loss

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))
