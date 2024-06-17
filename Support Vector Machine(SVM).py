# Importing Libraries

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Creating separate dataframes on the basis of flower_names
df0 = df[df.target == 0]
df1 = df[df.target == 0]
df2 = df[df.target == 0]

# Parameters X & Y for train and split
X = df.drop(['target'], axis=1)
Y = df.target

# Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Model Support Vector Machine(SVM)
model = SVC()
model.fit(X_train, Y_train)

# Model Score
print(model.score(X_test, Y_test))

# Prediction
print(model.predict(X_test))
print(iris.target_names[model.predict(X_test)])
