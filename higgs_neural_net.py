#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Mohit Shridhar
# @Date:   2015-11-18 22:17:13
# @Last Modified by:   Mohit Shridhar
# @Last Modified time: 2015-12-04 16:16:26

from __future__ import absolute_import
from __future__ import division


import numpy as np
import math
import os, sys

import tensorflow.python.platform
import tensorflow as tf

import sklearn
from sklearn.cross_validation import StratifiedKFold

from matplotlib import pyplot as plt

# Load traning data:
print 'Loading Training Data'
n_skiprows = 1
data_train = np.loadtxt('training.csv', delimiter=',', skiprows=n_skiprows, converters={32: lambda x:int(x=='s'.encode('utf-8'))})

# Discard all phi features:
data_train = np.delete(data_train, 16, 1)
data_train = np.delete(data_train, 18, 1)
data_train = np.delete(data_train, 19, 1)
data_train = np.delete(data_train, 23, 1)
data_train = np.delete(data_train, 25, 1)

max_idx = data_train.shape[1] - 1;

# Load test data:
print 'Loading Test Data'
data_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)

# Discard all phi features:
data_test = np.delete(data_test, 16, 1)
data_test = np.delete(data_test, 18, 1)
data_test = np.delete(data_test, 19, 1)
data_test = np.delete(data_test, 23, 1)
data_test = np.delete(data_test, 25, 1)

max_idx_test = data_test.shape[1] - 1;

X_test = data_test[:,1:max_idx_test]
ID_test = list(data_test[:,0])

# Sorting data into Y(labels), X(input), W(weights)
print 'Assigning data to numpy arrays'
Y_data = data_train[:,max_idx] > 0.
X_data = data_train[:,1:max_idx-2]
W_data = data_train[:,max_idx-1]

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

# Normalize Data:
mu_feat, sigma_feat = find_mean_and_std(X_data, X_data.shape[0], X_data.shape[1])
X_data = normalize_features(X_data, mu_feat, sigma_feat)

mu_feat_test, sigma_feat_test = find_mean_and_std(X_test, X_test.shape[0], X_test.shape[1])
X_test = normalize_features(X_test, mu_feat_test, sigma_feat_test)

# Stratified K-fold Shuffling for Cross Validation:
skf = sklearn.cross_validation.StratifiedKFold(Y_data, n_folds=2, shuffle=True, random_state=None)

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

n_features = X_data.shape[1]

X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, 2])

w_h1 = init_weights([n_features, 450])
w_h2 = init_weights([450, 450]) 
w_h3 = init_weights([450, 450])
w_o = init_weights([450, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
dnn = model(X, w_h1, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(dnn, Y))
train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
predict_op = tf.nn.softmax(dnn) 

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


	def compute_ams_score(prob_predict_train, prob_predict_valid, W_train, W_valid, pcut_percentile, cv_ratio):

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

	# Create 7 neural-net bags
	n_bags = 7;
	prob_bags = np.zeros(shape=(n_bags, X_test.shape[0]))

	for bag in range(n_bags):

		print 'Neural Network Bag:', bag

		# Reinitialize weights
		w_h1 = init_weights([n_features, 450])
		w_h2 = init_weights([450, 450]) 
		w_h3 = init_weights([450, 450])
		w_o = init_weights([450, 2])

		sess.run(tf.initialize_all_variables())

		for i in range(10000):

			for train_index, valid_index in skf:

				X_train, X_valid = X_data[train_index], X_data[valid_index]
				Y_train, Y_valid = Y_data[train_index], Y_data[valid_index]
				W_train, W_valid = W_data[train_index], W_data[valid_index]

				Y_train_binary = convert_one_to_binary(Y_train)
				Y_valid_binary = convert_one_to_binary(Y_valid)
				
				sess.run(train_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 0.8, p_keep_hidden: 0.5})
				# sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})
				# print i, np.mean(Y_valid != tf.argmax(sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0}), 1))

				prob_predict_train = sess.run(predict_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]
				prob_predict_valid = sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]

				cv_ratio = (1.0 * X_train.shape[0]) / (X_train.shape[0] + X_valid.shape[0]);
				ams_train, ams_valid = compute_ams_score(prob_predict_train, prob_predict_valid, W_train, W_valid, 85, cv_ratio)
				print i, '\t', ams_train, '\t', ams_valid

		# Predict probability of test input using the trained model:
		print 'Making predictions for test data'
		prob_bags[bag,:] = sess.run(predict_op, feed_dict={X: X_test, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]

	prob_predict_test = np.mean(prob_bags, axis=0) 

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

	# # Plot signal-background distribution
	classifier_training_s = sess.run(predict_op, feed_dict={X: X_train[Y_train>0.5], p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()
	classifier_training_b = sess.run(predict_op, feed_dict={X: X_train[Y_train<0.5], p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()
	classifier_testing_a = sess.run(predict_op, feed_dict={X: X_test, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()

	c_max = max([classifier_training_s.max(), classifier_training_b.max(), classifier_testing_a.max()])
	c_min = max([classifier_training_s.min(), classifier_training_b.min(), classifier_testing_a.min()])

	# Histograms of classifiers
	print 'Creating probability histograms for s & b'
	histo_training_s = np.histogram(classifier_training_s, bins=50, range=(c_min,c_max))
	histo_training_b = np.histogram(classifier_training_b, bins=50, range=(c_min,c_max))
	histo_training_a = np.histogram(classifier_testing_a, bins=50, range=(c_min,c_max))

	# min/max of histograms
	all_histograms = [histo_training_s, histo_training_b]
	h_max = max([histo[0].max() for histo in all_histograms]) * 1.2
	h_min = 1.0

	# historgram properties
	bin_edges = histo_training_s[1]
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
	bin_widths = (bin_edges[1:] - bin_edges[:-1])

	# Error bar plots, poisson uncertainty sqrt(N)
	errorbar_testing_a = np.sqrt(histo_training_a[0])

	# Draw objects
	ax1 = plt.subplot(111)

	# Draw historgrams for training data
	ax1.bar(bin_centers - bin_widths/2., histo_training_b[0], facecolor='red', linewidth=0, width=bin_widths, label='Background (Train)', alpha=0.5)
	ax1.bar(bin_centers - bin_widths/2., histo_training_s[0], bottom=histo_training_b[0], facecolor='blue', linewidth=0, width=bin_widths, label='Signal (Train)', alpha=0.5)

	ff = (1.0 * (sum(histo_training_s[0]) + sum(histo_training_b[0]))) / (1.0 * sum(histo_training_a[0]))

	# Draw error-bars for the testing data
	ax1.errorbar(bin_centers, ff * histo_training_a[0], yerr=ff*errorbar_testing_a, xerr=None, ecolor='black', c='black', fmt='.', label='Test (reweighted)')

	# Backdrop coloring
	ax1.axvspan(pcut, c_max, color='blue', alpha=0.08)
	ax1.axvspan(c_min, pcut, color='red', alpha=0.08)

	# Title and labels
	plt.title("Higgs Boson Signal-Background Distribution")
	plt.xlabel("Probability Output (Dropout Neural-Nets)")
	plt.ylabel("Counts/Bin")

	# Legend
	legend = ax1.legend(loc='upper center', shadow=True, ncol=2)
	for alabel in legend.get_texts():
		alabel.set_fontsize('small')

	# Save graph as png
	print 'Saving histrograms as PNG'
	plt.savefig('higgs_xgb.png')
