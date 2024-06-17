# Not suitable for current database as it gives different result than One Hot Encoding, So don't push on GitHub.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

# Load Dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\One_Hot_Coding.csv")
print(df)

# Label Encoding
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
print(dfle)

# Making X and Y parameter for model
X = dfle.drop(['price'], axis=1)
Y = dfle.price

# Model
reg = linear_model.LinearRegression()
reg.fit(X.values, Y)
print(reg.predict([[2, 2800]]))
print(reg.score(X.values, Y))
