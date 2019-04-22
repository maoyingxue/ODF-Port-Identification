import numpy as np
import cv2

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