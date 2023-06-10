import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def get_lr(x,y):
    lr = LinearRegression()
    lr.fit(x,y)
    return lr.coef_ , lr.intercept_

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.0002

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
    
    return m_curr,b_curr

if __name__ == "__main__":
    df = pd.read_csv("test_Scores.csv")
    print(df.head())

    #because equals is so faster with np array type
    x = np.array(df.math)
    y = np.array(df.cs)

    m , b = get_lr(df[["math"]],df.cs)
    m_curr , b_curr=gradient_descent(x,y)

    print("\nIf u used Sklearn linear regression model, your m: {} and b: {}".format(m,b))
    print("With your Gradient descent, your m: {} and b: {}".format(m_curr,b_curr))
