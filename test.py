from port_classification import portClassification
import cv2
import numpy as np
from utils import segment
import os
import xlrd
import math
import time
from classify import odfclassify
from get_points import type1,type2,type3,type4
def get_high(points_list):
    """
    # 用于计算透射变换的高
    :param points_list: 四个顶点的列表
    """
    h1 = math.sqrt((points_list[0][0] - points_list[3][0]) * (points_list[0][0] - points_list[3][0]) +
                   (points_list[0][1] - points_list[3][1]) * (points_list[0][1] - points_list[3][1]))
    h2 = math.sqrt((points_list[1][0] - points_list[2][0]) * (points_list[1][0] - points_list[2][0]) +
                   (points_list[1][1] - points_list[2][1]) * (points_list[1][1] - points_list[2][1]))
    return int(max(h1, h2))

def get_width(points_list):
    """
    # 用于计算透射变换的宽
    :param points_list: 四个顶点的列表
    """
    w1 = math.sqrt((points_list[0][0] - points_list[1][0]) * (points_list[0][0] - points_list[1][0]) +
                   (points_list[0][1] - points_list[1][1]) * (points_list[0][1] - points_list[1][1]))
    w2 = math.sqrt((points_list[2][0] - points_list[3][0]) * (points_list[2][0] - points_list[3][0]) +
                   (points_list[2][1] - points_list[3][1]) * (points_list[2][1] - points_list[3][1]))
    return int(max(w1, w2))

def transform(image, points):
    """
    # 图像扶正
    :param image: 待扶正的图片
    :return: 扶正后的图片
    """
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [get_width(points), 0], [get_width(points), get_high(points)], [0, get_high(points)]])
    p = cv2.getPerspectiveTransform(pts1, pts2)
    image_change = cv2.warpPerspective(image, p, (get_width(points), get_high(points)))
    return image_change

def drawPoints(event, x, y, flags, param):
    """
    # 手工标定四个顶点，参数为默认
    """
    if event == cv2.EVENT_LBUTTONDOWN and len(points_hand) < 4:
        cv2.circle(img_hand_copy, (x, y), 5, (0, 0, 255), 1)
        cv2.imshow("points", img_hand_copy)
        cv2.waitKey(0)
        points_hand.append(np.array([x, y]))
        print(points_hand)

def predictOdfType(image, method="input"):
    """
    # 预测机架的类型
    :param image: 原始图片
    :param method: 分类算法
    :return: 机架的类型
    """
    # 固定值
    if method == "hand":
        odf_type = "3"

    # 调用分类算法计算
    elif method == "algorithm":
        odf_type=odfclassify(image)
    # 人工输入
    elif method == "input":
        odf_type = str(input("请输入机架的类型："))

    return odf_type

def calculateNum(image, method="input"):
    """
    # 计算机架的行数和列数
    :param image: 扶正后的图片
    :param method: 计算方法
    :return: 机架的行数与列数
    """
    # 固定值
    if method == "hand":
        row = 24
        col = 12

    # 调用segment.py计算
    elif method == "segment":
        col, row = segment.Segmentation(image)

    # 人工输入
    elif method == "input":
        row = int(input("请输入行数："))
        col = int(input("请输入列数："))

    return row, col

def getPoints(image,odf_type, method="hand"):
    """
    # 得到机架的四个顶点
    :param image: 原始图片
    :param method: 计算方法
    :return: 四个顶点
    """
    if method == "hand":
        global img_hand, img_hand_copy, points_hand
        img_hand = image
        img_hand_copy = img_hand.copy()
        points_hand = []
        cv2.namedWindow("origin")
        cv2.imshow("origin", img_hand)
        cv2.setMouseCallback('origin', drawPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return points_hand
    if method=='auto':
        if odf_type=='1':
            points_hand=type1.getpoint(image)
        elif odf_type=='2':
            points_hand=type2.getpoint(image,'res/type2_6.txt')
        elif odf_type=='3':
            points_hand=type3.getpoint(image)
        elif odf_type=='4':
            points_hand=type4.getpoint(image)
        return points_hand

def predictPortType(image_change, x_num, y_num, gallery_path, method="knn", feature="hist", if_show=True):
    """
    # 得到各个端口的分类结果
    :param image_change: 扶正后的图片
    :param x_num, y_num: 机架的列数与行数
    :param method: 分类算法
    :param if_show: 是否可视化
    :return: 结果矩阵
    """

    k = portClassification.Classification(gallery_path, method, feature)
    split_x = image_change.shape[1] / x_num
    split_y = image_change.shape[0] / y_num
    result = np.zeros((y_num, x_num))
    c = 0
    for i in range(y_num):
        for j in range(x_num):
            c += 1
            img = image_change[round(i * split_y): round((i + 1) * split_y),
                  round(j * split_x): round((j + 1) * split_x)]

            start = time.clock()
            if method == "knn":
                result[i, j] = k.knn(img, 4)
            elif method == "svm":
                result[i, j] = k.svm(img)
            elif method == "knn2":
                result[i, j] = k.knn2(img)
            elapsed = (time.clock() - start)
            # print("已预测", c, "个端口, 用时：", elapsed)
            if if_show:
                tl = (round(j * split_x), round(i * split_y))
                br = (round((j + 1) * split_x), round((i + 1) * split_y))
                cv2.rectangle(image_change, tl, br, (178, 34, 34), 1)
                cv2.putText(image_change, str(int(result[i, j])), (int(tl[0]), int(tl[1] + split_x)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    if if_show:
        cv2.imshow("dsa", image_change)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result

def calculateAccuracy(results, excel_path, flag=False):
    """
    # 计算每张图上端口的分类精度
    :param results: 分类结果矩阵
    :param excel_path: ground truth存放的路径
    :param flag: 是否计算
    :return: 准确率
    """
    if flag == True:
        data = xlrd.open_workbook(excel_path)
        table = data.sheet_by_index(0)
        nrows = table.nrows
        ncols = table.ncols
        number_list = []

        # 列的数量以最小值为准，去除多余的列
        for i in range(nrows):
            number = 0
            for j in range(ncols):
                if (table.row(i)[j].value != ''):
                    number += 1
            number_list.append(number)
        ncols = min(number_list)

        # 若行列数目不匹配
        if nrows != len(results) or ncols != len(results[0]):
            return 0

        correct_count = 0
        unclear_count = 0
        labels = np.zeros([nrows, ncols], dtype=int)
        for i in range(nrows):
            for j in range(ncols):
                labels[i][j] = int(table.row(i)[j].value)
                if labels[i][j] == -1:
                    unclear_count += 1
                    continue
                if labels[i][j] == results[i][j]:
                    correct_count += 1
        print("ground truth:")
        print(labels)
        return correct_count / (nrows * ncols - unclear_count)
    else:
        return 0

if __name__ == '__main__':

    img_path = "origin_data\\test"
    excel_path = "origin_data\\label"
    port_cls_method = "knn2"
    feature_method = "hist"

    images = os.listdir(img_path)
    n = 0
    accuracy_list = []
    for image in images:

        if os.path.splitext(image)[1] != ".jpg" and os.path.splitext(image)[1] != ".png":
            continue
        n += 1
        print('-' * 20, n, '-' * 20)
        name = os.path.splitext(image)[0]
        print(name)

        img = cv2.imread(img_path + '\\' + image)

        # 图片缩放
        if (min(img.shape[0], img.shape[1]) > 3000):
            width = int(img.shape[0] / 4)
            high = int(img.shape[1] / 4)
        else:
            width = int(img.shape[0] / 1.2)
            high = int(img.shape[1] / 1.2)
        print(width, high)
        img = cv2.resize(img, (high, width))

        # 预测机架的类型
        odf_tpye = predictOdfType(img, "input")
        print("odf type:", odf_tpye)
        gallery_path = "gallery" + "\\" + "type" + odf_tpye

        # 得到机架的四个顶点
        points = getPoints(img,odf_tpye)

        # 图像扶正
        image_change = transform(img, points)

        # 得到机架的行列信息
        y_num, x_num = calculateNum(image, "input")

        # 预测端口的分类结果
        print("开始预测，请耐心等待......")
        results = predictPortType(image_change, x_num, y_num, gallery_path, port_cls_method, feature_method)
        print("predict results:")
        print(results)

        # 测试准确率
        excel_file = excel_path + '\\' + name + '.xlsx'
        accuracy = calculateAccuracy(results, excel_file, True)
        accuracy_list.append(accuracy)
        print("端口分类算法：", port_cls_method)
        print("特征提取方法：", feature_method)
        print("准确率：{}".format(accuracy))

    print("平均准确率：{}".format(np.mean(accuracy_list)))