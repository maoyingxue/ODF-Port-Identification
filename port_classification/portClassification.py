import numpy as np
import pickle
from collections import Counter
from utils import calFeature
from sklearn.svm import SVC
from sklearn import neighbors

def find_most(al):  # 找列表的众数
    c = Counter(al)
    dd = c.most_common(len(al))  # 转为列表
    # print("转换后：", dd)  #[(7, 14), (6, 5), (8, 2), (15, 1)]
    # 把出现次数最大的都找到 要下标和个数
    hh = [t for i, t in dd]
    nmax = hh.count(max(hh))  # 最大次数的个数
    ii = c.most_common(nmax)
    for i in ii:
        return i[0]

class Classification():
    def __init__(self, gallery_path, method="knn", feature="hist"):
        """
        # 加载训练数据以及模型
        :param gallery_path: 训练数据路径
        :param method: 提取特征的方式
        :param feature: 提取特征的方式
        """
        print("gallery path:", gallery_path)
        data = open(gallery_path + "\\complete_dbase.txt", 'rb')
        data = pickle.load(data)
        self.feature = feature

        if method == "knn":
            self.traindata_feature = data
            self.nparray = np.array([[]])
            for i, feat in enumerate(data):
                if i == 0:
                    self.nparray = feat[0].T
                else:
                    self.nparray = np.concatenate((self.nparray, feat[0].T), axis=0)
        elif method == "knn2":
            Xarray = np.array([[]])
            Yarray = np.array([])
            for i, feat in enumerate(data):
                Yarray = np.append(Yarray, feat[1])
                if i == 0:
                    Xarray = feat[0].T
                else:
                    Xarray = np.concatenate((Xarray, feat[0].T), axis=0)
            self.clf = neighbors.KNeighborsClassifier(4, algorithm="auto")
            self.clf.fit(Xarray, Yarray)
        elif method == "svm":
            Xarray = np.array([[]])
            Yarray = np.array([[]])
            for i, feat in enumerate(data):
                Yarray = np.append(Yarray, feat[1])
                if i == 0:
                    Xarray = feat[0].T
                else:
                    Xarray = np.concatenate((Xarray, feat[0].T), axis=0)
            self.clf = SVC(gamma='auto', kernel='rbf')
            self.clf.fit(Xarray, Yarray)

    def knn(self, testdata, k=4):
        """
        # 利用KNN进行分类
        :param testdata:   测试图片
        :param k: 近邻
        :return: 分类结果
        """
        if self.feature == "hist":
            test_feature = calFeature.createHistFeature(testdata)
        similar = np.linalg.norm(self.nparray - np.squeeze(test_feature), axis=1)
        index = np.argsort(similar)
        select = []
        for i in range(k):
            select.append(int(self.traindata_feature[index[i]][1]))
        return find_most(select)  # 投票

    def knn2(self, testdata):
        """
        # 利用sklearn中的KNN进行分类
        :param testdata:   测试图片
        :return: 分类结果
        """
        if self.feature == "hist":
            test_feature = calFeature.createHistFeature(testdata)
        return self.clf.predict(test_feature.T)

    def svm(self, testdata):
        """
        # 利用SVM进行分类
        :param testdata:   测试图片
        :return: 分类结果
        """
        if self.feature == "hist":
            test_feature = calFeature.createHistFeature(testdata)
        return self.clf.predict(test_feature.T)



