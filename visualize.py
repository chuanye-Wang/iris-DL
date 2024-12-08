import pandas as pd
import matplotlib.pyplot as plt

__path__ = "iris\iris.csv"
df = pd.read_csv(__path__, names=["sepal length", "sepal width", "petal length", "petal width", "cat"])
# print(df)

X = df[["sepal length", "sepal width", "petal length"]].values

mapping = {
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2    
}

df["cat"] = df["cat"].map(mapping)
y = df["cat"]

figure = plt.figure()
ax = figure.add_subplot(111, projection="3d")

ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='r', label="Iris-setosa")
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='b', label="Iris-versicolor")
ax.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], c='g', label="Iris-virginica")

ax.set_xlabel("sepal length(cm)")
ax.set_ylabel("sepal width(cm)")
ax.set_zlabel("petal length(cm)")

plt.legend()
plt.show()