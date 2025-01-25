import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["label"] = iris.target
df = df.iloc[:100, [0, 1, -1]]
df.columns = ["s_l", "s_w", "label"]

data = np.array(df)
X, y = data[:, :-1], data[:, -1]
y[y == 0] = -1


def perceptron(X, y, eta, max_iter=1000):
    w = np.zeros(X.shape[1])
    w[0] = random.randint(0, 10)
    w[1] = random.randint(0, 10)
    b = random.randint(0, 10)
    cnt = 0
    while True:
        wrong_count = 0
        for i in range(len(X)):
            x = X[i]
            yi = y[i]
            if yi * (np.dot(x, w) + b) <= 0:
                w = w + eta * yi * x
                b = b + eta * yi
                wrong_count += 1
        yield w, b, cnt, wrong_count
        if wrong_count == 0 or cnt > max_iter:
            break
        cnt += 1


n = 100
c = []

fig = plt.figure()
for i in range(n):
    x_ticks = np.linspace(0, 10, 10)
    ax = plt.subplot(1, 1, 1)
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("s_l")
    ax.set_ylabel("s_w")
    # for i in X[:50]:
    #     plt.scatter(i[0], i[1], label="0")
    # for i in X[50:]:
    #     plt.scatter(i[0], i[1], label="1")
    ax.scatter(X[:50, 0], X[:50, 1], color="blue")
    ax.scatter(X[50:100, 0], X[50:100, 1], color="yellow")
    ax.grid()
    ew, eb, ecnt, ewt = 0, 0, 0, 0
    for w, b, cnt, wt in perceptron(X, y, 0.721, 200):
        ew, eb, ecnt, ewt = w, b, cnt, wt
        try:
            plt.gca().lines[0].remove()
        except Exception:
            ...
        # np.linspace(0,10,100)
        ax.plot(df["s_l"], (w[0] * df["s_l"] + b) / (-w[1]), color="darkviolet")
        ax.set_title("%d time, wrong: %d" % (cnt, wt))
        plt.pause(0.001)
        # if cnt % 4: break
    ecrt = 0
    for j in X[:50]:
        crr = ew[0] * j[0] + ew[1] * j[1] + eb
        if crr > 0:
            ecrt += 1
    for j in X[50:]:
        crr = ew[0] * j[0] + ew[1] * j[1] + eb
        if crr < 0:
            ecrt += 1
    print("%d time, wrong: %d" % (i, ecrt))
    c.append((ew, eb, ecnt, ewt, ecrt))
    plt.clf()
    for j in X:
        j[0] += random.uniform(-0.1, 0.1)
        j[1] += random.uniform(-0.1, 0.1)
        j[0] = abs(j[0])
        j[1] = abs(j[1])
        if j[0] > 10:
            j[0] = 10
        if j[1] > 10:
            j[1] = 10

print(c)
