import cv2
import numpy as np
from collections import Counter

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

def Segmentation(img):
    """
        :param img: 找到四个点的图片
        :return: (x_num,y_num)
        """
    high, width = img.shape[:2]
    # img=cv2.resize(img,(width,high))
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_low = np.array([156, 43, 46])
    red_high = np.array([180, 255, 255])
    mask = cv2.inRange(image_hsv, red_low, red_high)
    # cv2.imshow("we", mask)
    image, contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    idx = 0
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i]) > max_area):
            max_area = cv2.contourArea(contours[i])
            idx = i
    rect = cv2.boundingRect(contours[idx])
    # cv2.circle(image_change, (242,771), 1, (0, 255, 0), -1)
    # 选取最大区域的宽和高
    x_length = rect[2]
    y_length = rect[3]
    # print("x", x_length)
    # print("y", y_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("sda", image_change)

    # x轴和y轴投影
    x_axis = np.zeros(mask.shape[1])
    y_axis = np.zeros(mask.shape[0])
    for x in range(len(x_axis)):
        for y in range(len(y_axis)):
            if (mask[y][x] != 0):
                x_axis[x] = 1
                break
    for y in range(len(y_axis)):
        for x in range(len(x_axis)):
            if (mask[y][x] != 0):
                y_axis[y] = 1
                break
    # 计算x轴每个小区域的间距
    start = 0
    flag = False
    x_list = []
    for i in range(len(x_axis)):
        if (x_axis[i] == 1 and start == 0):
            start = i
            continue
        if (x_axis[i] == 1 and start != 0 and flag == False):
            continue
        if (x_axis[i] == 0 and start != 0):
            flag = True
            continue
        if (x_axis[i] == 1 and flag == True):
            flag = False
            space = i - start
            start = 0
            x_list.append(space)
    x_list.sort()
    x_list = [x for x in x_list if x > x_length * 0.8]  # 剔除间距过小的区域
    x_num = 0
    divisor = x_list[0]
    averge_list = []  # 用来计算单个区域的平均间隔
    most_num = find_most(x_list)
    for num in x_list:
        n = round(num / divisor)
        if (n == 1):
            if (num >= most_num):
                averge_list.append(num)
            divisor = sum(averge_list) / len(averge_list) if len(averge_list) != 0 else divisor
        x_num += n
    if (width - divisor * x_num - x_length) < divisor:  # 判断总的长度 减去 提取出的区域长度和  余下的部分是否小于但各区域的长度
        x_num += 1
    else:
        x_num += round((width - divisor * x_num - y_length) / divisor)
    # print(x_num)
    # print("-------------------------------------------------------------------------------------")
    # 计算y轴每个小区域的间距
    start = 0
    flag = False
    y_list = []
    for i in range(len(y_axis)):
        if (y_axis[i] == 1 and start == 0):
            start = i
            continue
        if (y_axis[i] == 1 and start != 0 and flag == False):
            continue
        if (y_axis[i] == 0 and start != 0):
            flag = True
            continue
        if (y_axis[i] == 1 and flag == True):
            flag = False
            space = i - start
            start = 0
            y_list.append(space)
    y_list.sort()
    y_list = [y for y in y_list if y > y_length * 0.8]  # 剔除间距过小的区域

    y_num = 0
    divisor = y_list[0]
    averge_list = []  # 用来计算单个区域的平均间隔
    most_num = find_most(y_list)  # 找列表的
    for num in y_list:
        n = round(num / divisor)
        if (n == 1):
            if (num >= most_num):
                averge_list.append(num)
            divisor = sum(averge_list) / len(averge_list) if len(averge_list) != 0 else divisor
        y_num += n

    if (high - sum(y_list) - y_length) < divisor:  # 判断总的长度 减去 提取出的区域长度和  余下的部分是否小于但各区域的长度
        y_num += 1
    else:
        y_num += round((high - sum(y_list) - y_length) / divisor)
    # print(y_num)
    return x_num, y_num
    # cv2.waitKey()





