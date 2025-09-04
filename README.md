# Decision-Tree-On-Iris-Dataset
Assignment 3 for machine learning course includes decision tree on iris dataset along with the graph for information gain.


📦 Imports Explained :
[

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

]

matplotlib.pyplot as plt → Used for plotting graphs and visualizations.

load_iris → Loads the Iris dataset (flower measurements and species labels).

train_test_split → Splits the dataset into training and testing sets.

DecisionTreeClassifier → Machine learning model for classification using a decision tree.

plot_tree → Function to visualize the trained decision tree.

pandas as pd → Handles dataset in a tabular (DataFrame) format.

📊 Loading the Dataset:
[

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

]

iris.data → Feature values (flower measurements).

iris.feature_names → Column names (sepal length, sepal width, petal length, petal width).

iris.target → Encoded labels: 0 = setosa, 1 = versicolor, 2 = virginica.

 ### X contains the input features, and y contains the output labels.

Splitting Data:
[

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

]

70% training data (105 samples).

30% testing data (45 samples).

random_state=42 ensures reproducibility.

🌳 Building the Decision Tree:
[

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

]

criterion="entropy" → Uses Information Gain (entropy) for splitting.

random_state=42 → Fixes randomness.

.fit() → Trains the model using training data.

🖼️ Visualizing the Decision Tree:
[

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree on Iris Dataset")
plt.show()

]

Draws a decision tree with features and class labels.

filled=True colors the nodes by predicted class.

📈 Feature Importance (Information Gain):
[

importances = clf.feature_importances_
features = iris.feature_names

]

clf.feature_importances_ → Importance of each feature based on information gain.

Example (Iris dataset):

[0.01, 0.02, 0.56, 0.41]


→ Petal length & petal width are most important.

📊 Plotting Feature Importance:

[

plt.figure(figsize=(8,5))
plt.bar(features, importances, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Information Gain (Feature Importance)")
plt.title("Information Gain of Features in Iris Dataset")
plt.show()

]

Creates a bar chart of feature importance.

Shows which features contribute most to classification.

Summary:

Loaded the Iris dataset.

Split data into training (70%) and testing (30%).

Built a Decision Tree Classifier using entropy (information gain).

Visualized the decision tree.

Calculated and plotted feature importance.

## Result: Petal length and petal width are the most important features for classifying iris species.
