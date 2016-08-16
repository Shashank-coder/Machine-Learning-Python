import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# creating random dataset where variance is used to set the range of y and correlation for positive or negative slope
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# function to calculate slope and intercept of the best fit line
# m = (xbar*ybar - (x*y)bar)/(xbar)^2 - (x^2)bar)
# b = ybar - m*xbar
def best_fit_slope_and_intercept(xs, ys):
    m = ((np.mean(xs)*np.mean(ys)) - (np.mean(xs*ys))) / ((np.mean(xs)**2) - (np.mean(xs**2)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b

# formula = summation((yline - y)^2)
def squared_error(y_orig, y_line):
    return sum((y_line - y_orig)**2)

# formula = 1 - sq(y)/sq(ymean)
def coefficient_of_determination(y_orig, y_line):
    y_mean_line = [np.mean(y_orig) for y in y_orig]
    squared_error_regr = squared_error(y_orig, y_line)
    squared_error_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_mean)

xs, ys = create_dataset(40, 100, 2, correlation='neg')


m, b = best_fit_slope_and_intercept(xs, ys)
# one line for loop to get all the regression line points
regression_line = [(m*x + b) for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# to plot points on the graph
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100)
# to plot a curve on the graph
plt.plot(xs, regression_line)
plt.show()