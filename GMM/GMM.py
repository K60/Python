import numpy as np
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def data_generate():
    # 生成待聚类的数据子集
    mean1 = [1, 1]
    cov1 = [[3, 1], [1, 5]]
    data1 = np.random.multivariate_normal(mean1, cov1, 500)

    mean2 = [6, 2]
    cov2 = [[1, 0], [0, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 500)

    mean3 = [1, 10]
    cov3 = [[3, -2], [-2, 4]]
    data3 = np.random.multivariate_normal(mean3, cov3, 500)

    mean4 = [10, 10]
    cov4 = [[3, -1], [-1, 3]]
    data4 = np.random.multivariate_normal(mean4, cov4, 500)
    # 拼接数据
    data = np.concatenate((data1, data2, data3,data4))
    # 构建数据标签
    label = np.hstack([np.zeros(500, dtype=np.intp),
                       np.ones(500, dtype=np.intp),
                       np.ones(500, dtype=np.intp) * 2,
                       np.ones(500, dtype=np.intp) * 3]).T
    # 结果保留4位小数, 注意数据维度
    return np.round(data, 4), label


def normpdf(x, mu, sigma):
    '''
    多元正态（高斯）分布概率密度函数
    其中x为D维的向量，mu为均值，sigma为协方差
    '''
    # 这里的n对应的是多元高斯分布公式中的D
    n = len(x)
    # 计算下边俩个被除的系数
    # 加一个很小的单位阵是应对随机参数导致的协方差矩阵无法求逆的情况
    div = (2 * np.pi) ** (n / 2) * (abs(np.linalg.det(sigma)) ** 0.5)
    # 计算指数系数
    expOn = -0.5 * (np.dot((x - mu).T, np.dot(np.linalg.inv(sigma), (x - mu))))
    # 计算当前给定的参数下的概率
    return np.exp(expOn) / div


def init_parameters(data, k):
    '''
    用于生成K个高斯分布的初始化参数和选择每个高斯分布类别的概率
    :param data: 待分类的数据
    :param k: 数据的类别
    :return: 返回高斯分布均值，协方差矩阵和类别先验概率
    '''
    _, n = data.shape

    model = k_means(data, n_clusters=k)
    mus = model[0]
    # mus = np.random.rand(k, n)

    sigmas = np.zeros([k,n,n])
    for i in range(len(sigmas)):
        sigmas[i]=np.eye(n)*np.random.rand()*10

    pis = np.random.rand(k)
    pis /= np.sum(pis)
    return mus, sigmas, pis


def EM_algorithm(data, k, steps):
    # 随机初始化k个高斯分布的初始参数（mu，sigma）和每个类别的概率pi
    mus, sigmas, pis = init_parameters(data, k)
    # 构建响应度数组
    gamaArray = np.zeros((len(data), k))
    for s in range(steps):  # 步长循环
        # E-step
        for j, x in enumerate(data):  # 通过枚举遍历数据集中的每一个点计算响应度
            temp, tempP = 0, 0
            for i in range(k):
                tempP = normpdf(x, mus[i], sigmas[i])
                gamaArray[j][i] = pis[i] * tempP
                temp += tempP
            gamaArray[j] /= temp
        # M-setp
        for i in range(k):
            # 更新mus
            mus[i] = np.dot(gamaArray[:, i].T, data) / sum(gamaArray[:, i])
            # 更新sigmas
            temp = np.zeros(sigmas[0].shape)
            for j in range(len(data)):
                ds = (data[j] - mus[i]).reshape(len(mus[i]), 1)
                temp += gamaArray[j][i] * np.dot(ds, ds.T)
            temp /= sum(gamaArray[:, i])
            sigmas[i] = temp
            # 更新pis
            pis[i] = sum(gamaArray[:, i]) / len(data)
    return mus, sigmas, pis


def draw_ellipse(mu, cov, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = plt.gca()
    # Convert covariance to principal axes
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(mu, nsig * width, nsig * height,
                             angle, **kwargs))


k = 4
data, label = data_generate()
mus, sigmas, pis = EM_algorithm(data, k, 10)
plt.scatter(data[:, 0], data[:, 1], c=label, s=7)
for i in range(k):
    draw_ellipse(mus[i], sigmas[i], alpha=0.4, color='red', fill=False, linewidth=2)
plt.show()
