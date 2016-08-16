import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = Quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# replacing all the nan data with a dummy data 'cause removing the col will get a lot of data missed and nan data can't be used for ML
df.fillna(-9999, inplace = True)

# we are trying to make Adj. Close as a label and predicting Close few days ahead from the presnt date
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

# X is our features so everything except label is our features
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# here we are keeping the last 31 data to predict the next month data and rest will be shown on the graph as in the data
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
Y = np.array(df['label'])

# here we are randomly classifying data as training set and testing set using cross validation
# testing set size is 20% of the total data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

# using linear regression function with argument n_jobs which means no. of threads or processors to be used while computation of the data
# clf = LinearRegression(n_jobs=-1)
# fit is synonymous to train
# clf.fit(X_train, Y_train)
# using pickle which is used to save the trained data in a file
# so that we can use it later and we can save our time to train the algo again and again
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
# loading the data from the pickle file
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# score is synonymous to test
accuracy = clf.score(X_test, Y_test)
# predicting the the nexxt 31 days data from the last 31 days available data
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

# creating the tabular form of the new data formed corresonding to their respective dates
last_date = df.iloc[-1].name
one_day = 86400
next_day = last_date + datetime.timedelta(seconds=one_day)

# puttin nan in all columns of the new dates except for the forecast whre we put the predicted data
for i in forecast_set:
    df.loc[next_day] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_day += datetime.timedelta(seconds=one_day)

# plotting the graph
# plotting Adj. Colse - this plots all the non-nan data, forecast is nan in the old data
df['Adj. Close'].plot()
# here forecast is non-nan and other data are nan in case of new data
df['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# info - the available data is till 28-6-2016 and we are using the lasr month data to predict the data till 29-7-2016
# first we randomly take some training data from the available data and train the machine
# then we check the accuracy from the rest of the random data left by applying the algo
# then we apply the best algo to predict the future data