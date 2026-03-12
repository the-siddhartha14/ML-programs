
import pandas as pd
from sklearn.linear_model import LinearRegression


data = {
    'Area': [1000, 1500, 1800, 2000, 2500],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Age': [10, 8, 6, 5, 3],
    'Price': [2000000, 3000000, 3600000, 4000000, 5000000]
}


df = pd.DataFrame(data)


X = df[['Area', 'Bedrooms', 'Age']]


y = df['Price']


model = LinearRegression()


model.fit(X, y)rice for a ne


prediction = model.predict([[2200, 4, 4]])

print("Predicted House Price:", prediction[0])
