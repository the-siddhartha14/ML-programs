
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi'],
    'Area': [1000, 1500, 1800, 2000, 2500],
    'Price': [2000000, 3000000, 3500000, 4000000, 5000000]
}


df = pd.DataFrame(data)


dummies = pd.get_dummies(df['City'])


df = pd.concat([df, dummies], axis=1)


df = df.drop('City', axis=1)


X = df[['Area', 'Delhi', 'Mumbai']]


y = df['Price']


model = LinearRegression()


model.fit(X, y)


prediction = model.predict([[2200, 1, 0]])

print("Predicted Price:", prediction[0])
