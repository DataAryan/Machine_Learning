# Importing Libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns


# Loading Dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\New folder\salaries.csv")
print(df)


# Database differentiation
inp = df.drop(["Salary_more_than_100k"],axis=1)
target = df['Salary_more_than_100k']


# Label Encoding
le = LabelEncoder()
inp['company_label'] = le.fit_transform(inp['Company'])
inp['job_label'] = le.fit_transform(inp['Job'])
inp['degree_label'] = le.fit_transform(inp['Degree'])


# Parameters X & Y for model
X = inp.drop(['Company','Job','Degree'],axis=1)
Y = target

print(X)


# Train and Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)


# Model : Decision Tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)


# Model Score
print(X_test)
print(Y_test)
print(model.score(X_test,Y_test))


# Prediction
print(model.predict(X_test))