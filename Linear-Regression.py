


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = {
    'Experience': [1, 2, 3, 4, 5],
    'Salary': [30000, 35000, 40000, 45000, 50000]
}


df = pd.DataFrame(data)


X = df[['Experience']]
y = df['Salary']


model = LinearRegression()


model.fit(X, y)


prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])
