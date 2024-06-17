# Importing libraries

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

digits = load_digits()

# Load Dataframe
df = pd.DataFrame(digits.data)
df['target'] = digits.target

# Parameters X & Y for training
X = df.drop(['target'], axis=1)
Y = df.target

# Train and Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Model : Random Forest
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, Y_train)

# Model Score
print(model.score(X_test, Y_test))

# Predicting
print(X_test)
print(Y_test)
print(model.predict(X_test))

# Confusion Matrix
# To get idea where or for which inputs do the model does not work well.
# Sklearn provide a way to know this in way of confusion matrix
cm = confusion_matrix(Y_test, model.predict(X_test))
print(cm)

# Heatmap
# It is visualization of confusion matrix for better understanding using seaborn library,
# It is used to understand the accuracy of the model.
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
