# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\insurance.csv")

# Parameters X & Y for model
X = df[['age']]
Y = df[['bought_insurance']]

# Train and Split
# Here values.ravel() is used to flatten the array as logistics regression takes flattened array as input.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y.values.ravel(), test_size=0.1)

# Model (Logistics Regression)
reg = LogisticRegression()
reg.fit(X_train, Y_train)

# Prediction
print(reg.predict(X_test))

print(X_test)
print(Y_test)
print(reg.score(X_test, Y_test))
