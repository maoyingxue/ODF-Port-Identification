from utils import calFeature
import cv2
import os
import pickle

def saveFeature(gallery_path, txt_path, method="hist"):
    """
    # 将gallery中的端口进行特征提取并保存
    :param gallery_path: 图片路径
    :param txt_path: 保存路径
    :param method: 提取特征的方式
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
        if method == "hist":
            data.append(calFeature.createHistFeature(img))
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
