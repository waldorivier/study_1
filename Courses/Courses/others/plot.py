import os
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns

# closure pour encapsuler la fonction

def fParam(a) :
    def f(x) :
        if x != 0 :
            return a * np.sin(x) / x
        else : 
            return 0

    return f

# Series est un enumerable (car un tableau n'en est pas un)

x = pd.Series(np.arange(-15,15))
y1 = x.apply(fParam(1))
y2 = x.apply(fParam(2))
y3 = x.apply(fParam(3))

fig, ax = plt.subplots()

ax.plot(x, y1, color='red', linewidth=2.0, linestyle='--', label='f1')
ax.plot(x, y2, color='blue', linewidth=2.0)
ax.plot(x, y3, color='green', linewidth=2.0)

ax.set_xlabel('x')
ax.set_ylabel('a*sin(x)')
# pour afficher les labels...
ax.legend()
plt.show()

# --------------------------------------------------------

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], 
columns=iris['feature_names'])
iris_df['species'] = iris['target']

iris_df.rename(index=str, 
               columns={"sepal length (cm)": "sepal_length", 
                        "sepal width (cm)" : "sepal_width",
                        "petal length (cm)": "petal_length",
                        "petal width (cm)" : "petal_width",
                        }, inplace=True)

fig, ax = plt.subplots()

for spec in iris_df.species.unique() :
    spec_ = iris_df[iris_df.species==spec]
    ax.scatter(spec_.petal_length, spec_.sepal_length, label=iris.target_names[spec])

ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(iris_df.columns[0])
plt.ylabel(iris_df.columns[1])
plt.show()

# --------------------------------------------------------

sns.pairplot(iris_df)
plt.show()



