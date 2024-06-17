import pandas as pd
from sklearn import linear_model

df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\One_Hot_Coding.csv")
print(df)

# One Hot Encoding
df1 = pd.get_dummies(df, columns=["town"])      # same as One Hot Encoding
print(df1)

df2 = df1.drop(["town_west windsor"], axis=1)
print(df2)

# Parameter for model
x = df2.drop(["price"], axis=1)
y = df2.price

# Linear Regression Model
reg = linear_model.LinearRegression()
reg.fit(x.values, y)
print(reg.predict([[2800, 0, 1]]))
print(reg.score(x.values, y))
