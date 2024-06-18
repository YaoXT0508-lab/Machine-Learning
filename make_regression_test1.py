from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 直接使用 make_regression 而不是 datasets.make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=1)

# 注意：n_target 参数已被移除，如果你需要多个目标变量，可以设置 n_targets > 1
plt.scatter(X, y)
plt.show()

