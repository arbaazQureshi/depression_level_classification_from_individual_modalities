from load_model import load_model
from load_data import load_training_data, load_development_data, load_test_data
import keras

import numpy as np
import os
from os import path
from sklearn.metrics import f1_score, accuracy_score

import random

#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"


from keras import backend as K

"""
def weighted_categorical_crossentropy(weights):
	'''
	A weighted version of keras.objectives.categorical_crossentropy
	
	Variables:
		weights: numpy array of shape (C,) where C is the number of classes
	
	Usage:
		weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
		loss = weighted_categorical_crossentropy(weights)
		model.compile(loss=loss,optimizer='adam')
	'''

	weights = K.variable(weights)

	def loss(y_true, y_pred):
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		# calc
		loss = y_true * K.log(y_pred) * weights
		loss = -K.sum(loss, -1)
		return loss
	
	return loss
"""

training_progress = []
development_progress = []
test_progress = []

model = load_model()

#weights = np.array([1/47, 1/28, 1/19, 1/7, 1/4])
model.compile(optimizer='adam', loss='categorical_crossentropy')

X_train, Y_train = load_training_data()
X_dev, Y_dev = load_development_data()
X_test, Y_test = load_test_data()

Y_dev_onehot = np.argmax(Y_dev, axis = 1)
Y_test_onehot = np.argmax(Y_test, axis = 1)

min_crossentropy_dev = 10000
max_fscore_dev = -1
max_accuracy_dev = -1

min_crossentropy_test = 10000
max_fscore_test = -1
max_accuracy_test = -1

current_epoch_number = 1
total_epoch_count = 300

m = X_train.shape[0]
batch_size_list = list(range(1, m))

print("\n\n")

while(current_epoch_number <= total_epoch_count):
	
	print((str(total_epoch_count - current_epoch_number)+' ')*20)

	#batch_size = random.choice(batch_size_list)
	batch_size = m
	print("Batch size is", batch_size)
	
	history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1)

	loss_dev = model.evaluate(X_dev, Y_dev, batch_size = X_dev.shape[0])
	loss_test = model.evaluate(X_test, Y_test, batch_size = X_test.shape[0])
	
	Y_dev_pred = model.predict(X_dev, batch_size = X_dev.shape[0])
	Y_test_pred = model.predict(X_test, batch_size = X_test.shape[0])

	Y_dev_pred = np.argmax(Y_dev_pred, axis = 1)
	Y_test_pred = np.argmax(Y_test_pred, axis = 1)

	f_score_dev = f1_score(Y_dev_onehot, Y_dev_pred, average='macro')
	f_score_test = f1_score(Y_test_onehot, Y_test_pred, average='macro')

	accuracy_dev = accuracy_score(Y_dev_onehot, Y_dev_pred)
	accuracy_test = accuracy_score(Y_test_onehot, Y_test_pred)

	loss_train = [history.history['loss'][0]]

	print("Test:\t\t", f_score_test, accuracy_test, loss_test)
	print("Development:\t", f_score_dev, accuracy_dev, loss_dev)
	print("Train:\t\t", loss_train[0])

	if(loss_dev < min_crossentropy_dev):
		
		min_crossentropy_dev = loss_dev
		model.save('opton_dev_crossentropy.h5')
		print("BEST DEV CROSSENTROPY MODEL!\n\n")

		with open('development_crossentropy_best.txt', 'w') as f:
			f.write('Min development crossentropy:\t\t' + str(loss_dev) + '\n')
			f.write('Corresponding development f1-score:\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development accuracy:\t' + str(accuracy_dev) + '\n\n')

			f.write('Corresponding test f1-score:\t\t' + str(f_score_test) + '\n')
			f.write('Corresponding test accuracy:\t' + str(accuracy_test) + '\n')
			f.write('Corresponding test crossentropy:\t\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(f_score_dev > max_fscore_dev):
		
		max_fscore_dev = f_score_dev
		model.save('opton_dev_f-score.h5')
		print("BEST DEV F-SCORE MODEL!\n\n")

		with open('development_f-score_best.txt', 'w') as f:
			f.write('Max development f-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Corresponding development crossentropy:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test f-score:\t\t' + str(f_score_test) + '\n')
			f.write('Corresponding test accuracy:\t\t' + str(accuracy_test) + '\n')
			f.write('Corresponding test crossentropy:\t\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(accuracy_dev > max_accuracy_dev):
		
		max_accuracy_dev = accuracy_dev
		model.save('opton_dev_accuracy.h5')
		print("BEST DEV ACCURACY MODEL!\n\n")

		with open('development_accuracy_best.txt', 'w') as f:
			f.write('Max development accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Corresponding development f-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development crossentropy:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test f-score:\t\t' + str(f_score_test) + '\n')
			f.write('Corresponding test accuracy:\t\t' + str(accuracy_test) + '\n')
			f.write('Corresponding test crossentropy:\t\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	



	if(loss_test < min_crossentropy_test):
		
		min_crossentropy_test = loss_test
		model.save('opton_test_crossentropy.h5')
		print("BEST TEST CROSSENTROPY MODEL!\n\n")

		with open('test_crossentropy_best.txt', 'w') as f:
			f.write('Min test crossentropy:\t\t' + str(loss_test) + '\n')
			f.write('Corresponding test f1-score:\t' + str(f_score_test) + '\n')
			f.write('Corresponding test accuracy:\t' + str(accuracy_test) + '\n\n')

			f.write('Corresponding development f1-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development accuracy:\t' + str(accuracy_dev) + '\n')
			f.write('Corresponding development crossentropy:\t\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	
	if(f_score_test > max_fscore_test):
		
		max_fscore_test = f_score_test
		model.save('opton_test_f-score.h5')
		print("BEST TEST F-SCORE MODEL!\n\n")

		with open('test_f-score_best.txt', 'w') as f:
			f.write('Max test f-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding test accuracy:\t\t' + str(accuracy_test) + '\n')
			f.write('Corresponding test crossentropy:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding development f-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Corresponding development crossentropy:\t\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(accuracy_test > max_accuracy_test):
		
		max_accuracy_test = accuracy_test
		model.save('opton_test_accuracy.h5')
		print("BEST TEST ACCURACY MODEL!\n\n")

		with open('test_accuracy_best.txt', 'w') as f:
			f.write('Max test accuracy:\t\t' + str(accuracy_test) + '\n')
			f.write('Corresponding test f-score:\t\t' + str(f_score_test) + '\n')
			f.write('Corresponding test crossentropy:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding development f-score:\t\t' + str(f_score_dev) + '\n')
			f.write('Corresponding development accuracy:\t\t' + str(accuracy_dev) + '\n')
			f.write('Corresponding development crossentropy:\t\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training crossentropy:\t\t' + str(loss_train[0]) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')


	training_progress.append([current_epoch_number, loss_train[0]])
	development_progress.append([current_epoch_number, loss_dev, f_score_dev, accuracy_dev])
	test_progress.append([current_epoch_number, loss_test, f_score_test, accuracy_test])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('development_progress.csv', np.array(development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('test_progress.csv', np.array(test_progress), fmt='%.4f', delimiter=',')

	current_epoch_number = current_epoch_number + 1
	print("\n\n")