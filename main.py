import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

print("Python version: ", sys.version)
print("scikit-learn version: ", sklearn.__version__)
print("pandas version: ", pd.__version__)
print("numpy version: ", np.__version__)
print("plotly version: ", plotly.__version__)
print("\n\n")

iris = pd.read_csv("./data/IRIS.csv")

print(iris.head())

print(iris.describe())

print("Target Labels", iris["species"].unique())

fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()

x = iris.drop("species", axis=1)
y = iris["species"]

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train.values, y_train)

print("\n\n\n")

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)

print("Values: {}".format(x_new))
print("Prediction: {}".format(prediction))
