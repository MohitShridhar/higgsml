import numpy as np
import xgboost as xgb
import math

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sknn import mlp

from matplotlib import pyplot as plt

# Load traning data:
print 'Loading Data'
n_skiprows = 1
data_train = np.loadtxt('training.csv', delimiter=',', skiprows=n_skiprows, converters={32: lambda x:int(x=='s'.encode('utf-8'))})

# Pick a random seed for CV split:
np.random.seed(42)
r = np.random.rand(data_train.shape[0])

# Sorting data into Y(labels), X(input), W(weights)
print 'Assigning data to numpy arrays'
cv_ratio = 0.75
# First 90% for training
Y_train = data_train[:,32][r<cv_ratio] > 0.
X_train = data_train[:,1:31][r<cv_ratio]
W_train = data_train[:,31][r<cv_ratio]
# Last 10% for validation
Y_valid = data_train[:,32][r>=cv_ratio] > 0.
X_valid = data_train[:,1:31][r>=cv_ratio]
W_valid = data_train[:,31][r>=cv_ratio]

print 'Training classifier ...'

# ------- Gradient Boost Classifier -----------
# gbc = GBC(n_estimators=50, max_depth=5, min_samples_leaf=200, max_features=10, verbose=1)
# gbc.fit(X_train, Y_train)

# prob_predict_train = gbc.predict_proba(X_train)[:,1]
# prob_predict_valid = gbc.predict_proba(X_valid)[:,1]


# ------- X Gradient Boost Classifier ---------
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=3, silent=False, nthread=8)
xgb_model.fit(X_train, Y_train)

prob_predict_train = xgb_model.predict_proba(X_train)[:,1]
prob_predict_valid = xgb_model.predict_proba(X_valid)[:,1]

# ------- Neural Network -----------------
# n_feat = data_train.shape[1]
# n_targets = Y_train.max() + 1

# # Train neural net
# nn = mlp.Classifier(
#         layers=[
#             mlp.Layer("Tanh", units=n_feat/8),
#             mlp.Layer("Sigmoid", units=n_feat/16),
#             mlp.Layer("Softmax", units=n_targets)],
#         n_iter=50,
#         n_stable=5,
#         learning_rate=0.002,
#         learning_rule="momentum",
#         valid_size=0.1,
#         verbose=1)

# nn.fit(X_train, Y_train)
# prob_predict_train = nn.predict_proba(X_train)[:,1]
# prob_predict_valid = nn.predict_proba(X_valid)[:,1]

# Take the top 15% of the output based on the highest probability:
pcut = np.percentile(prob_predict_train, 85)

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
print 'Computing AMS score'
def AMSScore(s,b):
	return math.sqrt (2.*( (s + b + 10.) * math.log(1. + s / (b + 10.)) - s))

ams_train = AMSScore(s_train, b_train)
ams_valid = AMSScore(s_valid, b_valid)

print 'AMS of training set (90%): ', ams_train
print 'AMS of validation set (10%): ', ams_valid

# Compute Percent Error training and validation set:
print 'Computing Percentage Error'
predict_train = xgb_model.predict(X_train)
predict_valid = xgb_model.predict(X_valid)

print 'Percentage Error (Train): ', np.mean(np.abs((Y_train - predict_train) / Y_train)) * 100
print 'Percentage Error (Valid): ', np.mean(np.abs((Y_valid - predict_valid) / Y_valid)) * 100

# Terminate if the generalization error is very bad:
epsilon_error = 0.45
if abs(ams_train - ams_valid) > epsilon_error:
	print 'Generalization Error Too Big'
	# quit()

# Load test data:
print 'Loading test data'
data_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)
X_test = data_test[:,1:31]
ID_test = list(data_test[:,0])

# Predict probability of test input using the trained model:
print 'Making predictions for test data'
prob_predict_test = xgb_model.predict_proba(X_test)[:,1]
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


# # ------- Plot s-b distribution ---------

# # Plot signal-background distribution
# classifier_training_s = xgb_model.predict_proba(X_train[Y_train>0.5])[:,1].ravel()
# classifier_training_b = xgb_model.predict_proba(X_train[Y_train<0.5])[:,1].ravel()
# classifier_testing_a = xgb_model.predict_proba(X_test)[:,1].ravel()

# c_max = max([classifier_training_s.max(), classifier_training_b.max(), classifier_testing_a.max()])
# c_min = max([classifier_training_s.min(), classifier_training_b.min(), classifier_testing_a.min()])

# # Histograms of classifiers
# print 'Creating probability histograms for s & b'
# histo_training_s = np.histogram(classifier_training_s, bins=50, range=(c_min,c_max))
# histo_training_b = np.histogram(classifier_training_b, bins=50, range=(c_min,c_max))
# histo_training_a = np.histogram(classifier_testing_a, bins=50, range=(c_min,c_max))

# # min/max of histograms
# all_histograms = [histo_training_s, histo_training_b]
# h_max = max([histo[0].max() for histo in all_histograms]) * 1.2
# h_min = 1.0

# # historgram properties
# bin_edges = histo_training_s[1]
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
# bin_widths = (bin_edges[1:] - bin_edges[:-1])

# # Error bar plots, poisson uncertainty sqrt(N)
# errorbar_testing_a = np.sqrt(histo_training_a[0])

# # Draw objects
# ax1 = plt.subplot(111)

# # Draw historgrams for training data
# ax1.bar(bin_centers - bin_widths/2., histo_training_b[0], facecolor='red', linewidth=0, width=bin_widths, label='Background (Train)', alpha=0.5)
# ax1.bar(bin_centers - bin_widths/2., histo_training_s[0], bottom=histo_training_b[0], facecolor='blue', linewidth=0, width=bin_widths, label='Signal (Train)', alpha=0.5)

# ff = (1.0 * (sum(histo_training_s[0]) + sum(histo_training_b[0]))) / (1.0 * sum(histo_training_a[0]))

# # Draw error-bars for the testing data
# ax1.errorbar(bin_centers, ff * histo_training_a[0], yerr=ff*errorbar_testing_a, xerr=None, ecolor='black', c='black', fmt='.', label='Test (reweighted)')

# # Backdrop coloring
# ax1.axvspan(pcut, c_max, color='blue', alpha=0.08)
# ax1.axvspan(c_min, pcut, color='red', alpha=0.08)

# # Title and labels
# plt.title("Higgs Boson Signal-Background Distribution")
# plt.xlabel("Probability Output (XGBoost)")
# plt.ylabel("Counts/Bin")

# # Legend
# legend = ax1.legend(loc='upper center', shadow=True, ncol=2)
# for alabel in legend.get_texts():
# 	alabel.set_fontsize('small')

# # Save graph as png
# print 'Saving histrograms as PNG'
# plt.savefig('higgs_xgb.png')



