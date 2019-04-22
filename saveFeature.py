import numpy as np
import cv2
import os
import pickle

def createHistFeature(grid, small_grid=3):
    """
    # 生成颜色直方图特征
    :param img: 数据
    :return: 数据的特征
    """
    hist_mask = np.array([])
    colnum = int(grid.shape[1] / small_grid)
    rownum = int(grid.shape[0] / small_grid)
    for i in range(small_grid):
        for j in range(small_grid):
            image = grid[i * colnum:(i + 1) * colnum, j * rownum:(j + 1) * rownum, :]
            hist_mask0 = cv2.calcHist([image], [0], None, [16], [0, 255])
            hist_mask1 = cv2.calcHist([image], [1], None, [16], [0, 255])
            hist_mask2 = cv2.calcHist([image], [2], None, [16], [0, 255])
            hist_mask_small = np.concatenate((hist_mask0, hist_mask1, hist_mask2), axis=0)
            if (len(hist_mask) == 0):
                hist_mask = hist_mask_small
            else:
                hist_mask = np.concatenate((hist_mask, hist_mask_small), axis=0)
    return hist_mask

def saveFeature(gallery_path, txt_path):
    """
    # 将gallery中的端口进行特征提取并保存
    :param gallery_path: 图片路径
    :param txt_path: 保存路径
    """
    print(gallery_path)
    images = os.listdir(gallery_path)
    print(images)
    feature = []
    for image in images:
        if os.path.splitext(image)[1] != ".jpg":
            continue
        img = cv2.imread(gallery_path + "\\" + image)
        image_name = os.path.splitext(image)[0]
        data = []
        data.append(createHistFeature(img))
        data.append(int(image_name[-1]))
        feature.append(data)
    fp = open(txt_path, 'wb')
    pickle.dump(feature, fp)
    fp.close()

if __name__ == '__main__':
    # 计算训练集的特征并保存
    gallery_path_total = "gallery\\"
    gallery_list = os.listdir(gallery_path_total)
    for gallery in gallery_list:
        gallery_path = gallery_path_total + gallery + "\\images"
        txt_path = gallery_path_total + gallery + "\\complete_dbase.txt"
        saveFeature(gallery_path, txt_path)
