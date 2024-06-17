# Importing Libraries
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Read CSV/Excel or load dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\Car.csv")
print(df)

# Parameters X & Y for model
X = df[['Mileage', 'Age(yrs)']]
Y = df[['Sell Price($)']]
print(X)
print(Y)

# Train and Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# X_train = Part of dataset used for training the model
# X_test = Part of dataset used for testing the model
# Y_train = Part of dataset used for training the model
# Y_test = Part of dataset used for testing the model

# Model
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)

# Prediction
print(reg.predict(X_test))

print(X_test)
print(Y_test)
print(reg.score(X_test, Y_test))
