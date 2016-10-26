import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import preprocessing, cross_validation
import pickle

df = pd.read_csv('train.csv')
df.fillna(0, inplace=True)
print(df.head())

X = np.array(df.drop(['label'], 1))
# X = preprocessing.scale(X)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

# clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
# clf.fit(X_train, Y_train)
#
# with open('DigitRecognizer.pickle', 'wb') as f:
#     pickle.dump(clf, f)
pickle_in = open('DigitRecognizer.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)
# print(accuracy)