import numpy as np
import cv2
import pickle
from collections import Counter

class Knn():
    def __init__(self, data_path):   #加载训练数据
        """
        :param data: 训练数据
        """
        print("gallery path:", data_path)
        data = open(data_path + "\\complete_dbase.txt", 'rb')
        data = pickle.load(data)
        self.traindata_feature = data

    def create_feature(self, img):
        """
        #生成特征
        :param img: 数据
        :return: 数据的特征
        """
        small_grid = 3
        hist_mask = np.array([])
        colnum = int(img.shape[1] / small_grid)
        rownum = int(img.shape[0] / small_grid)
        for i in range(small_grid):
            for j in range(small_grid):
                image = img[i * colnum:(i + 1) * colnum, j * rownum:(j + 1) * rownum, :]

                hist_mask0 = cv2.calcHist([image], [0], None, [16], [0, 255])
                hist_mask1 = cv2.calcHist([image], [1], None, [16], [0, 255])
                hist_mask2 = cv2.calcHist([image], [2], None, [16], [0, 255])
                hist_mask_small = np.concatenate((hist_mask0, hist_mask1, hist_mask2), axis=0)
                if (len(hist_mask) == 0):
                    hist_mask = hist_mask_small
                else:
                    hist_mask = np.concatenate((hist_mask, hist_mask_small), axis=0)
        return hist_mask


    def knn(self, testdata, k):
        """
        :param testdata:   测试图片
        :param k: 近邻
        :return: 分类结果
        """
        test_feature = self.create_feature(testdata)
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
        return self.find_most(select)  # 投票

    def find_most(self, al):  # 找列表的众数
        c = Counter(al)
        dd = c.most_common(len(al))  # 转为列表
        # print("转换后：", dd)  #[(7, 14), (6, 5), (8, 2), (15, 1)]
        # 把出现次数最大的都找到 要下标和个数
        hh = [t for i, t in dd]
        nmax = hh.count(max(hh))  # 最大次数的个数
        ii = c.most_common(nmax)
        for i in ii:
            return i[0]


