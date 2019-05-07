# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:05:33 2019

@author: maoyingxue
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import operator
import os
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
def get_types(path):
    files=os.listdir(path)
    trainMats=[]
    labels=[]
    for file in files:
        #print(file)
        label=file.split('_')[0][-1]
        trainMat=[]
        image = cv2.imread(path+file)
        image=cv2.resize(image,(500,700))
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        trainMat.append(np.mean(H))
        trainMat.append(np.mean(S))
        trainMat.append(np.mean(V))
        trainMats.append(trainMat)
        labels.append(label)
    return trainMats,labels
trainDataPath='odf_test/'
trainMats,labels=get_types(trainDataPath)
def odfclassify(testImg):
    global trainMats
    testMat=[]
    HSV = cv2.cvtColor(testImg, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    testMat.append(np.mean(H))
    testMat.append(np.mean(S))
    testMat.append(np.mean(V))
    trainMats=np.array(trainMats)
    result=classify0(testMat,trainMats,labels,1)
    #print("预测机架类型：",result)
    return result
if __name__ == '__main__':
    testImg=cv2.imread('odf_test/type2_1.jpg')
    testImg=cv2.resize(testImg,(500,700))
    odfclassify(testImg)