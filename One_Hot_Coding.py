import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

# Load Dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\One_Hot_Coding.csv")
print(df)

# One Hot Encoding
ohe = OneHotEncoder()
ohe_df = ohe.fit_transform(df[["town"]]).toarray()
print(ohe.categories_)

# Concat One Hot Encoded Columns
df1 = pd.DataFrame(ohe_df, columns=["town_" + str(i) for i in range(ohe_df.shape[1])])
df2 = pd.concat([df, df1], axis=1)
df2 = df2.drop(["town", "town_2"], axis=1)
print(df2)

# Making X and Y parameter for model
X = df2.drop(['price'], axis=1)
Y = df2.price

# Model
reg = linear_model.LinearRegression()
reg.fit(X.values, Y)
print(reg.predict([[2800, 0, 1]]))
print(reg.score(X.values, Y))
