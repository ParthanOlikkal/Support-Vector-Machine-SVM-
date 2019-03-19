import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib.pyplot as plt

#Load Dataset
url = "https://raw.githubusercontent.com/ParthanOlikkal/Support-Vector-Machine-SVM-/master/cell_samples.csv"
cell_df = pd.read_csv(url)
cell_df.head()


#Data visualization by plotting Clump thickness and Uniformity of Cell Size
ax = cell_df[cell_dfp['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#Data preprocessing and selection
cell_df.dtypes

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

#converting pd to np
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNuc1', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

#Since class can have only 2 values : 2-benign, 4-malignant, the measurement level should be changed for this
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]

#Train/Test the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4) 
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#For Kernelling Radial Basis Function is used
from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma='auto')
clf.fit(X_train, y_train)

#Predict
yhat = clf.predict(X_test)
yhat [0:5]

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import itertools #for iteration

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
	if normalize;
		cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotate=45)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1]))
	plt.text(j,i,format(cm[i,j],fmt), horzontalaligment='center', color='white' if cm[i,j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

#Compute confusion matrix
cnf_matrix = confusion_matrix(y_label, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_label, yhat))

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')

#f1_score
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average = 'weighted')

#jaccard index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

