#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Mohit Shridhar
# @Date:   2015-11-18 22:17:13
# @Last Modified by:   Mohit Shridhar
# @Last Modified time: 2015-12-03 20:45:28

from __future__ import absolute_import
from __future__ import division

from sklearn import cross_validation

import numpy as np
import math
import os, sys

import tensorflow.python.platform
import tensorflow as tf

# Load traning data:
print 'Loading Data'
n_skiprows = 1
data_train = np.loadtxt('training.csv', delimiter=',', skiprows=n_skiprows, converters={32: lambda x:int(x=='s'.encode('utf-8'))})

# Discard all phi features:
data_train = np.delete(data_train, 16, 1)
data_train = np.delete(data_train, 18, 1)
data_train = np.delete(data_train, 19, 1)
data_train = np.delete(data_train, 23, 1)
data_train = np.delete(data_train, 25, 1)

max_idx = data_train.shape[1] - 1;

# Pick a random seed for CV split:
np.random.seed(42)
r = np.random.rand(data_train.shape[0])

# Sorting data into Y(labels), X(input), W(weights)
print 'Assigning data to numpy arrays'
cv_ratio = 0.75
# First 75% for training
Y_train = data_train[:,max_idx][r<cv_ratio] > 0.
X_train = data_train[:,1:max_idx-1][r<cv_ratio]
W_train = data_train[:,max_idx-1][r<cv_ratio]
# Last 25% for validation
Y_valid = data_train[:,max_idx][r>=cv_ratio] > 0.
X_valid = data_train[:,1:max_idx-1][r>=cv_ratio]
W_valid = data_train[:,max_idx-1][r>=cv_ratio]

# Normalize Dataset
def find_mean_and_std(np_array_obj, n_rows, n_cols):
	colsum = np.zeros((1,n_cols))
	mean = np.zeros((1,n_cols))
	std = np.zeros((1,n_cols))

	try:
		colsum = sum(np_array_obj)
	
	except MemoryError:
		for k in range(n_rows):
			colsum += np_array_obj[k,:]

	mean = (1.0 / n_rows) * colsum

	for l in range(n_rows):
		std += (np_array_obj[l,:] - mean)**2

	std = (1.0 / n_rows) * std 

	return mean, std

def normalize_features(np_array_obj, mean, std):
	n_rows, n_cols = np_array_obj.shape
	norm_array_obj = np.zeros((n_rows, n_cols))
	norm_row = np.zeros((1,n_rows))

	for i in range(n_rows):
		norm_row = np_array_obj[i]
		norm_row -= mean
		norm_row = norm_row / std
		norm_array_obj[i] = norm_row

	return norm_array_obj

print 'Normalizing and Scaling Dataset'
mu_train, sigma_train = find_mean_and_std(X_train, X_train.shape[0], X_train.shape[1])
X_train = normalize_features(X_train, mu_train, sigma_train)

mu_valid, sigma_valid = find_mean_and_std(X_valid, X_valid.shape[0], X_valid.shape[1])
X_valid = normalize_features(X_valid, mu_valid, sigma_valid)


# ------------- TensorFlow Dropout Feed-Forward Neural Network ---------------

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):

	X = tf.nn.dropout(X, p_drop_input)
	h1 = tf.nn.relu(tf.matmul(X, w_h1))

	h1 = tf.nn.dropout(h1, p_drop_hidden)
	h2 = tf.nn.relu(tf.matmul(h1, w_h2))

	h2 = tf.nn.dropout(h2, p_drop_hidden)
	h3 = tf.nn.relu(tf.matmul(h2, w_h3))

	h3 = tf.nn.dropout(h3, p_drop_hidden)
	return tf.matmul(h3, w_o)


print 'Setting-up Neural Network'

n_features = X_train.shape[1]
n_samples = X_train.shape[0]

X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, 2])

w_h1 = init_weights([n_features, 400])
w_h2 = init_weights([400, 400]) 
w_h3 = init_weights([400, 400])
w_o = init_weights([400, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
dnn = model(X, w_h1, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(dnn, Y))
train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
predict_op = tf.nn.softmax(dnn) 

saver = tf.train.Saver()
restart_train = True
checkpoint_steps = 500
checkpoint_folder = 'checkpoints'
checkpoint_filename = checkpoint_folder + '/dnn_model.ckpt'

def convert_one_to_binary(np_array_obj):
	array_binary = np.zeros(shape=(np_array_obj.shape[0], 2))
	for k in range(np_array_obj.shape[0]):
		if (np_array_obj[k]):
			array_binary[k][1] = True
		else:
			array_binary[k][0] = True

	return array_binary


with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	def compute_ams_score(prob_predict_train, prob_predict_valid, pcut_percentile):

		# Take the top 15% of the output based on the highest probability
		pcut = np.percentile(prob_predict_train, pcut_percentile)

		# Make s, b predictions:
		Yhat_train = prob_predict_train > pcut
		Yhat_valid = prob_predict_valid > pcut

		# Apply given weights to get the real true positives and true negatives:
		true_positive_train = W_train * (Y_train == 1.0) * (1.0/cv_ratio)
		true_negative_train = W_train * (Y_train == 0.0) * (1.0/cv_ratio)
		true_positive_valid = W_valid * (Y_valid == 1.0) * (1.0/(1-cv_ratio))
		true_negative_valid = W_valid * (Y_valid == 0.0) * (1.0/(1-cv_ratio))

		# labels (s, b) for the training set:
		s_train = sum ( true_positive_train * (Yhat_train == 1.0) )
		b_train = sum ( true_negative_train * (Yhat_train == 1.0) )
		s_valid = sum ( true_positive_valid * (Yhat_valid == 1.0) )
		b_valid = sum ( true_negative_valid * (Yhat_valid == 1.0) )

		# Compute AMS score for training set:
		# print 'Computing AMS score'
		def AMSScore(s,b):
			return math.sqrt (2.*( (s + b + 10.) * math.log(1. + s / (b + 10.)) - s))

		ams_train = AMSScore(s_train, b_train)
		ams_valid = AMSScore(s_valid, b_valid)

		return ams_train, ams_valid

	Y_train_binary = convert_one_to_binary(Y_train)
	Y_valid_binary = convert_one_to_binary(Y_valid)

	for i in range(10000):
		
		sess.run(train_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 0.8, p_keep_hidden: 0.5})
		# sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})
		# print i, np.mean(Y_valid != tf.argmax(sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0}), 1))

		prob_predict_train = sess.run(predict_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]
		prob_predict_valid = sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]

		ams_train, ams_valid = compute_ams_score(prob_predict_train, prob_predict_valid, 85)
		print i, '- AMS Training:', ams_train, ' Test:', ams_valid

	# Load test data:
	print 'Loading test data'
	data_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)
	X_test = data_test[:,1:max_idx-1]
	ID_test = list(data_test[:,0])

	# Predict probability of test input using the trained model:
	print 'Making predictions for test data'
	prob_predict_test = sess.run(predict_op, feed_dict={X: X_test, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]
	pcut = np.percentile(prob_predict_test, 0.85)
	Yhat_test = list(prob_predict_test > pcut)
	prob_predict_test = list(prob_predict_test)

	# Format the output into the Kaggle specified format:
	print 'Organizing the prediction results'
	result_list = []
	for x in range(len(ID_test)):
		result_list.append([int(ID_test[x]), prob_predict_test[x], 's'*(Yhat_test[x]==1.0)+'b'*(Yhat_test[x]==0.0)])

	# Sort result in decending probability order:
	result_list = sorted(result_list, key=lambda a_entry: a_entry[1])

	# Replace probability prediction with integer ranking
	for y in range(len(result_list)):
		result_list[y][1] = y+1

	# Sort based on the integer ranking
	result_list = sorted(result_list, key=lambda a_entry: a_entry[0])

	# Write result to csv file:
	print 'Writing CSV file for Kaggle Submission'
	fcsv = open('higgsml_output.csv', 'w')
	fcsv.write('EventId,RankOrder,Class\n')
	for line in result_list:
		the_line = str(line[0]) + ',' + str(line[1]) + ',' + line[2] + '\n'
		fcsv.write(the_line);
	fcsv.close()
