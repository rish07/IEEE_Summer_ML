import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.optimizers import RMSprop
from keras import backend as K

rms = RMSprop()


def contrastive_loss(y_true, y_pred): #Loss Function
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def init():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("weights_best_final.hdf5")
	print("Loaded Model from disk")


	#compile and evaluate loaded model
	loaded_model.compile(loss=contrastive_loss,optimizer=rms,metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph
