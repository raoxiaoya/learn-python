import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

'''
随机小批量梯度下降法
'''

# 生成一个大的合成数据集
X, y = make_regression(n_samples=100000, n_features=10, noise=0.1)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class SGDRegressor:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.bias = 0
        test_sample_size = X.shape[0]  # 训练集个数

        self.losses = []

        for i in range(self.n_iterations):
            # 生成 [0, test_sample_size) 序列，并随机打乱
            indices = np.random.permutation(test_sample_size)
            X_shuffled = X[indices]  # 打乱X
            y_shuffled = y[indices]  # 打乱y

            for start in range(0, test_sample_size, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = np.dot(X_batch, self.theta) + self.bias
                error = y_pred - y_batch

                gradient_theta = (1 / self.batch_size) * \
                    np.dot(X_batch.T, error)
                gradient_bias = (1 / self.batch_size) * np.sum(error)

                self.theta -= self.learning_rate * gradient_theta
                self.bias -= self.learning_rate * gradient_bias

            # 计算损失并存储
            y_train_pred = np.dot(X, self.theta) + self.bias
            loss = mean_squared_error(y, y_train_pred)
            self.losses.append(loss)

        return self

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias


sgd_regressor = SGDRegressor(
    learning_rate=0.01, n_iterations=100, batch_size=64)
sgd_regressor.fit(X_train, y_train)

# 预测
y_train_pred = sgd_regressor.predict(X_train)
y_test_pred = sgd_regressor.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")


def showLost():
    # 损失函数的变化
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sgd_regressor.losses)),
             sgd_regressor.losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Loss Function over Iterations')
    plt.legend()
    plt.show()


def showResult():
    # 预测结果的分布
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.3, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.show()


def showDistribute():
    # 测试数据和预测数据的误差分布情况
    residuals = y_test - y_test_pred
    sns.histplot(residuals, kde=True, color='purple')  # 绘制直方图
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


showDistribute()
