import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def generate_mixed_gaussian(mu1, sigma1, mu2, sigma2, p, size=10000,seed=None):
    if seed is not None:
        np.random.seed(seed)
    eta = np.random.choice([0, 1], size=size, p=[1-p, p])
    X = np.random.normal(mu1, np.sqrt(sigma1), size)
    Y = np.random.normal(mu2, np.sqrt(sigma2), size)
    Z = X + eta * Y
    return Z

def plot_histogram(data, title, filename):
    # 直方图绘制
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g')

    # 使用Seaborn的kdeplot绘制核密度估计轮廓线
    sns.kdeplot(data, bw_adjust=0.5, color='k', linewidth=2)

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.show()

# 参数
mu1, sigma1 = 0, 1
mu2, sigma2 = 5, 2
p = 0.4
seed = 52

# 生成混合高斯分布的随机数
data = generate_mixed_gaussian(mu1, sigma1, mu2, sigma2, p, seed=seed)

# 画出频率分布直方图
plot_histogram(data, f'Mixed Gaussian Distribution\nmu1={mu1}, sigma1^2={sigma1}, mu2={mu2}, sigma2^2={sigma2}, p={p}', 'mixed_gaussian.png')