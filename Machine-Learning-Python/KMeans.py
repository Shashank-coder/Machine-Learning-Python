import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')
# print(df.head())

df.drop(['name', 'body'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())

# this function converts all the string data to number data
def handle_non_numerical_data(df):
    # here we extract all the column headers in a list(.values) and (.columns) gives us all the column objects
    columns = df.columns.values

    # here we iterate on all the values of each columns which require change
    for column in columns:
        # we will store all the unique data of a single column in this dictionary with a number value corresponding to it
        text_digit_val = {}

        # use this function to map the string to their corresponding numbers from the above dictionary
        def convert_to_int(val):
            return text_digit_val[val]

        # if the column data is not a number then only perform these else skkip
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # extracting all rows of the column in a list
            column_contents = df[column].values
            # extracting all the unique rows of the column in a list
            unique_elements = set(column_contents)
            x = 0
            # putting these unique rows in the dictionary with unique numbers
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1

            # using pandas map function to map all the column rows with the new data obtained after passing it to the function
            # syntax = map(func, arg)
            # NOTE- list() is optional here just to show that the output is a list
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
# print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

# checking the predictions of the trained machine with the actual data
correct = 0.0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))