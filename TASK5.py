import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data from a CSV file
data = pd.read_csv('DECISION TREE IRIS.csv')

# Split data into features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the data
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(10, 7))
plot_tree(clf, filled=True, rounded=True, feature_names=X.columns, class_names=y.unique())
plt.show()

# Predict the class of a new data point
new_data = pd.DataFrame([[1, 0, 0, 0]], columns=X.columns)
prediction = clf.predict(new_data)
print(prediction)

