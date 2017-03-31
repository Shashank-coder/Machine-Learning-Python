import numpy as np
import pandas as pd
from sknn.mlp import Classifier, Layer

df = pd.read_csv('newformat.csv')
# df = pd.get_dummies(df)
df_train = df[:28710]
df_test = df[28710:]
train_num_examples = 28710

X_train = np.array(df_train.drop(['emotion'], 1))
X_train = np.stack(X_train)
Y_train = np.array(df_train['emotion'])
Y_train = np.stack(Y_train)


X_test = np.array(df_test.drop(['emotion'], 1))
X_test = np.stack(X_test)
Y_test = np.array(df_test['emotion'])
Y_test = np.stack(Y_test)


#   Build the classifier
nn = Classifier(
    layers=[
        Layer("Rectifier", units=50),
        Layer("Softmax", name="OutputLayer", units=7)
    ],
    learning_rate=0.01,
    n_iter=100
)

# print(y)

#   Fit the classifier with training data set
nn.fit(X_train, Y_train)

# parameters = nn.get_parameters()


#   Predict answers for test data set
Y_pred = nn.predict(X_test)

correctPredCount = 0
for i in range(0, X_test.__len__()):
    if Y_pred[i] == Y_test[i]:
        correctPredCount += 1


accuracy = (float(correctPredCount) / X_test.__len__()) * 100

#print(accuracy)

print("Accuracy of prediction is : " + str(accuracy))
