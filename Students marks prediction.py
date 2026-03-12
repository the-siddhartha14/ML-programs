
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Marks': [35, 40, 50, 55, 65]
}

df = pd.DataFrame(data)


X = df[['Hours_Studied']]
y = df['Marks']


model = LinearRegression()


model.fit(X, y) 

prediction = model.predict([[6]])

print("Predicted Marks for 6 hours study:", prediction[0])
