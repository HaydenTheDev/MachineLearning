#  simple regression fromula // y= b0 + b1 * x1
#  example                      salary = b0 + b1 * experience


#  Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#  Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=1)

#  Training the Simple Linear Regression model on the Training set

#  Predicting the Test set results

#  Visualizing the Training set results

#  Visualizing the Test set results
