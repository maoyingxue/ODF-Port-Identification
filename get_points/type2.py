# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:28:01 2019

@author: maoyingxue
"""

import numpy as np
import cv2
def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16
    return new_h / img_size[0], new_w / img_size[1]
def getpoint(image,box_addr):
    #image = cv2.imread('odf_test/type2_6.jpg')
    rh,rw=resize_image(image)
    #ratio=0.15
    #image=cv2.resize(image,(0,0),fx=ratio,fy=ratio)
    f=open(box_addr)
    image4text=image.copy()
    x=image.shape[1]//2
    y=image.shape[0]//2
    rects=[]
    rect=[]
    for line in f.readlines():
        tmp=[int(t) for t in line.strip().split(',')[:-1]]
#        bp1=np.array([tmp[0]/rh*ratio,tmp[1]/rw*ratio,],dtype=int)
#        bp3=np.array([tmp[4]/rh*ratio,tmp[5]/rw*ratio],dtype=int)
        bp1=np.array([tmp[0]/rh,tmp[1]/rw],dtype=int)
        bp3=np.array([tmp[4]/rh,tmp[5]/rw],dtype=int)   
        if((bp3[0]-bp1[0])/(bp3[1]-bp1[1])>3):
            rect.append(bp1)
            rect.append(bp3)
            rect.append((bp3[0]-bp1[0])/(bp3[1]-bp1[1]))
            cv2.circle(image4text,tuple(bp1),5,(155,0,0),-1)
            cv2.circle(image4text,tuple(bp3),5,(155,0,0),-1)
            rects.append(rect)
        rect=[]
    if len(rects)>=2:
        forsort=[rect[2] for rect in rects]
        forsort=np.array(forsort)
        forsort=np.argsort(-forsort)
        forsort=forsort[:2]
        dis_ratio=((rects[forsort[0]][0][0]-rects[forsort[1]][0][0])**2+(rects[forsort[0]][0][1]-rects[forsort[1]][0][1])**2)**0.5/y
        print(dis_ratio)    
        if dis_ratio<0.6:
             forsort=forsort[:1]
        rects=[rects[i] for i in forsort]
    print(rects)
    miny,maxy=0,0
    if len(rects)==1:
        h1=(rects[0][0][1]+rects[0][1][1])/2
        if h1<y:
            miny=int(h1)
        else:
            maxy=int(h1)
    else:
        h1=(rects[0][0][1]+rects[0][1][1])/2
        h2=(rects[1][0][1]+rects[1][1][1])/2
        miny=int(min(h1,h2))
        maxy=int(max(h1,h2))
    print(miny,maxy)
    cv2.imshow("image4text", image4text)
    
    Img=image.copy()
    HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    #red
    lower = np.array([156, 43, 46], dtype="uint8")
    upper = np.array([180, 255, 255], dtype="uint8")
    Mask1 = cv2.inRange(HSV, lower, upper)
    lower = np.array([0, 43, 46], dtype="uint8")
    upper = np.array([10, 255, 255], dtype="uint8")
    Mask2 = cv2.inRange(HSV, lower, upper)
    Mask=cv2.bitwise_or(Mask1,Mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    Mask = cv2.erode(Mask, kernel)
    Mask = cv2.dilate(Mask, kernel)
    vis=False
    if vis==True:
        BlueThings = cv2.bitwise_and(Img, Img, mask=Mask)
        cv2.imshow("images2", np.hstack([Img, BlueThings]))
    #cv2.imwrite("type/type2_1.jpg",np.hstack([Img, BlueThings]))
    _, Contours, Hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = sorted(Contours, key=lambda c: c.shape[0], reverse=True)
    Contours = [c for c in Contours if len(c) > 5 ]
    #cv2.drawContours(image,Contours,-1,(0,255,0),3)
    distance1=0
    distance2=0
    distance3=0
    distance4=0
    
    P1,P2,P3,P4=0,0,0,0
    for c in Contours:
        for p in c:
            dis=abs(p[0][0]-x)+abs(p[0][1]-y)
            #dis=(p[0][0]-x)**2+(p[0][1]-y)**2
            if dis>distance1 and p[0][0]<x and p[0][1]<y:
                distance1=dis;P1=p
            if dis>distance2 and p[0][0]>x and p[0][1]<y:
                distance2=dis;P2=p
            if dis>distance3 and p[0][0]<x and p[0][1]>y:
                distance3=dis;P3=p
            if dis>distance4 and p[0][0]>x and p[0][1]>y:
                distance4=dis;P4=p
    distance1,distance2=0,0
    #print(P1,P2,P3,P4)
    if P1 is 0:
        minx=P3[0][0]
    if P2 is 0:
        maxx=P4[0][0]
    if P3 is 0:
        minx=P1[0][0]
    if P4 is 0:
        maxx=P2[0][0]
    if P1 is not 0 and P2 is not 0 and P3 is not 0 and P4 is not 0:
        minx=min(P1[0][0],P3[0][0])
        maxx=max(P2[0][0],P4[0][0])
    if P1 is not 0:
        if P1[0][0]-minx>20:
            P1[0][0]=minx
        if miny is not 0 and abs(P1[0][1]-miny)>20:
            P1[0][1]=miny+20
    elif miny is not 0:
        P1=[[0,0]]
        P1[0][0]=minx
        P1[0][1]=miny+20
    if P2 is not 0:
        if maxx-P2[0][0]>20:
            P2[0][0]=maxx
        if miny is not 0 and abs(P2[0][1]-miny)>20:
            P2[0][1]=miny+20
    elif miny is not 0:
        P2=[[0,0]]
        P2[0][0]=maxx
        P2[0][1]=miny+20
    if P3 is not 0:
        if P3[0][0]-minx>20:
            P3[0][0]=minx
        if maxy is not 0 and abs(maxy-P3[0][1])>20:
            P3[0][1]=maxy-20
    elif maxy is not 0:
        P3=[[0,0]]
        P3[0][0]=minx
        P3[0][1]=maxy-20
    if P4 is not 0:
        if maxx-P4[0][0]>20:
            P4[0][0]=maxx
        if maxy is not 0 and abs(maxy-P4[0][1])>20:
            P4[0][1]=maxy-20
    elif maxy is not 0:
        P4=[[0,0]]
        P4[0][0]=maxx
        P4[0][1]=maxy-20
    if vis==True:
        cv2.circle(image,tuple(P1[0]),5,(0,155,0),-1)
        cv2.circle(image,tuple(P2[0]),5,(0,155,0),-1)   
        cv2.circle(image,tuple(P3[0]),5,(0,155,0),-1)
        cv2.circle(image,tuple(P4[0]),5,(0,155,0),-1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return [P1[0],P2[0],P3[0],P4[0]]
