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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("x train set: ")
print(x_train)
print("y train set: ")
print(y_train)

#  Predicting the Test set results

y_pred = regressor.predict(x_test)

#  Visualizing the Training set results

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
#  Visualizing the Test set results
