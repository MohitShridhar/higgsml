#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Mohit Shridhar
# @Date:   2015-11-18 22:17:13
# @Last Modified by:   Mohit Shridhar
# @Last Modified time: 2015-11-18 23:31:41

import numpy as np
import math
import os

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

# Load traning data:
print 'Loading Data'
n_skiprows = 1 #244500
data_train = np.loadtxt('training.csv', delimiter=',', skiprows=n_skiprows, converters={32: lambda x:int(x=='s'.encode('utf-8'))})

# Pick a random seed for CV split:
np.random.seed(42)
r = np.random.rand(data_train.shape[0])

# Sorting data into Y(labels), X(input), W(weights)
print 'Assigning data to numpy arrays'
cv_ratio = 0.75
# First 75% for training
Y_train = data_train[:,32][r<cv_ratio] > 0.
X_train = data_train[:,1:31][r<cv_ratio]
# Last 25% for validation
Y_valid = data_train[:,32][r>=cv_ratio] > 0.
X_valid = data_train[:,1:31][r>=cv_ratio]

# Build Dataset
print 'Arranging Dataset'

ds_train = ClassificationDataSet(X_train.shape[1], 1, nb_classes=2)
for k in range(len(X_train)):
	ds_train.addSample(X_train[k].ravel(), Y_train[k])

ds_test = ClassificationDataSet(X_valid.shape[1], 1, nb_classes=2)
for k in range(len(X_valid)):
	ds_test.addSample(X_valid[k].ravel(), Y_valid[k])

ds_train._convertToOneOfMany()
ds_test._convertToOneOfMany()

# Train Feed-Forward Neural Network
print 'Training Feed-Forward Network using Back-propagation'

# Save nn parameters locally so we don't have retrain everytime
nn_filename = 'higgs_fnn.xml'

if os.path.isfile(nn_filename):
	fnn = NetworkReader.readFrom(nn_filename)
else:
	fnn = buildNetwork(ds_train.indim, 600, 600, 600, ds_train.outdim, outclass=SoftmaxLayer)

trainer = BackpropTrainer(fnn, dataset=ds_train, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)

# CV using test data
for x in range(0, 30):
	trainer.trainEpochs(1)
	print 'Percent Error on Test Dataset: ', percentError(trainer.testOnClassData(dataset=ds_test), ds_test['class'])

# Write nn to local file
NetworkWriter.writeToFile(fnn, nn_filename)


