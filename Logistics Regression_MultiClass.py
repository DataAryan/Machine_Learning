# Importing Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# Loading Dataset
digits = load_digits()

# Train and Split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Model(Logistics Regression)
reg = LogisticRegression(solver='lbfgs', max_iter=1000)
reg.fit(x_train, y_train)

# Prediction
print(reg.predict(x_test))
print(x_test)
print(y_test)

# Score of Model
print(reg.score(x_test, y_test))

# Confusion Matrix
# To get idea where or for which inputs do the model does not work well.
# Sklearn provide a way to know this in way of confusion matrix
cm = confusion_matrix(y_test, reg.predict(x_test))
print(cm)

# Heatmap
# It is visualization of confusion matrix for better understanding using seaborn library,
# It is used to understand the accuracy of the model.
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
