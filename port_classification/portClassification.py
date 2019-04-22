import numpy as np
import pickle
from collections import Counter
from utils import calFeature

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
    def __init__(self, data_path):   #加载训练数据
        """
        :param data: 训练数据
        """
        print("gallery path:", data_path)
        data = open(data_path + "\\complete_dbase.txt", 'rb')
        data = pickle.load(data)
        self.traindata_feature = data

    def knn(self, testdata, k):
        """
        :param testdata:   测试图片
        :param k: 近邻
        :return: 分类结果
        """
        test_feature = calFeature.createHistFeature(testdata)
        nparray = np.array([[]])
        for i, feat in enumerate(self.traindata_feature):
            if (i == 0):
                nparray = feat[0].T
            else:
                nparray = np.concatenate((nparray, feat[0].T), axis=0)

        similar = np.linalg.norm(nparray - np.squeeze(test_feature), axis=1)
        index = np.argsort(similar)
        select = []
        for i in range(k):
            select.append(int(self.traindata_feature[index[i]][1]))
        return find_most(select)  # 投票




