import cv2
import os
import math
import xlrd
import numpy as np

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

def transform(image):
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

def cutTemplate(image_change, image_change_copy, grid_labels, path):
    """
    # 剪切端口训练集并保存
    :param image_change: 扶正后的图片
    :param image_change_copy: image_change的副本，用于可视化
    :param grid_labels: 训练集的标签
    :param path: 训练集存储路径
    """
    split_x = image_change.shape[1] / ncols
    split_y = image_change.shape[0] / nrows
    for i in range(nrows):
        for j in range(ncols):
            grid = image_change[round(i * split_y): round((i + 1) * split_y), round(j * split_x): round((j + 1) * split_x)]
            tl = (round(j * split_x), round(i * split_y))
            br = (round((j + 1) * split_x), round((i + 1) * split_y))
            cv2.rectangle(image_change_copy, tl, br, (178, 34, 34), 1)
            if(grid_labels[i][j] == -1):
                continue
            label = grid_labels[i][j]
            saved_name = path + "\\" + name + "_" + str(i+1) + "_" + str(j+1) + "_" + str(label) + '.jpg'
            if os.path.exists(saved_name):
                os.remove(saved_name)
            cv2.imencode('.jpg', grid)[1].tofile(saved_name)
    return image_change_copy

def findPoints(event, x, y, flags, param):
    """
    # 手工标定四个顶点，参数为默认
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), 1)
        cv2.imshow("points", img_copy)
        cv2.waitKey(0)

        points.append(np.array([x, y]))
        print(points)
    if len(points) == 4:
        correct_img = transform(img)
        correct_img_copy = transform(img_copy)
        correct_img_copy = cutTemplate(correct_img, correct_img_copy, labels, gallery_path)

        cv2.imshow("grid", correct_img_copy)

if __name__ == '__main__':
    img_path = "origin_data\\sp"

    images = os.listdir(img_path)
    n = 0
    for image in images:
        global img, img_copy, points, width, high, nrows, ncols, labels, name, gallery_path

        if os.path.splitext(image)[1] != ".jpg" and os.path.splitext(image)[1] != ".png":
            continue
        n += 1
        print('-' * 20, n, '-' * 20)
        name = os.path.splitext(image)[0]
        print(name)

        gallery_path = "gallery\\" + name[0:5]
        if not os.path.exists(gallery_path):
            os.mkdir(gallery_path)
        gallery_path = "gallery\\" + name[0:5] + "\\images"
        if not os.path.exists(gallery_path):
            os.mkdir(gallery_path)

        excel_file = "origin_data\\label\\" + name + '.xlsx'
        print(excel_file)
        data = xlrd.open_workbook(excel_file)
        table = data.sheet_by_index(0)
        nrows = table.nrows
        ncols = table.ncols
        number_list = []

        # 列的数量以最小值为准，去除多余的列
        for i in range(nrows):
            number = 0
            for j in range(ncols):
                if(table.row(i)[j].value != ''):
                    number += 1
            number_list.append(number)
        ncols = min(number_list)
        labels = np.zeros([nrows, ncols], dtype=int)
        for i in range(nrows):
            for j in range(ncols):
                labels[i][j] = int(table.row(i)[j].value)
        print(labels)

        img = cv2.imread(img_path + '\\' + image)

        # 图片缩放
        if(min(img.shape[0], img.shape[1]) > 3000):
            width = int(img.shape[0] / 4)
            high = int(img.shape[1] / 4)
        else:
            width = int(img.shape[0] / 1.2)
            high = int(img.shape[1] / 1.2)
        print(width, high)
        img = cv2.resize(img, (high, width))
        img_copy = img.copy()

        # 边界标定，进行训练集的剪切并保存
        points = []
        cv2.namedWindow("origin")
        cv2.imshow("origin", img)
        cv2.setMouseCallback('origin', findPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()