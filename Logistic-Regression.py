
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7],
    'Pass': [0, 0, 0, 1, 1, 1, 1]  
}


df = pd.DataFrame(data)


X = df[['Hours_Studied']]
y = df['Pass']


model = LogisticRegression()


model.fit(X, y)


prediction = model.predict([[3.5]])

if prediction[0] == 1:
    print("Student will Pass")
else:
    print("Student will Fail")
