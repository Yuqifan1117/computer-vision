'''
Created on 2018年11月19日

@author: coderwangson
'''
"#codeing=utf-8"
import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
from numpy.core.fromnumeric import argsort
IMAGE_SIZE =(112,92)
def createDatabase():
    # 查看路径下所有文件
    TrainFiles = 'homework2/att-face/'
    T = []
    # 把所有图片转为1-D并存入T中
    for i in range(1,41):
        for j in range(1,7):
            image = cv.imread(TrainFiles+'s'+str(i)+'/'+str(j)+'.pgm',cv.IMREAD_GRAYSCALE)
            image=cv.resize(image,IMAGE_SIZE)
            # 转为1-D
            image = image.reshape(image.size,1)
            T.append(image)        
    T = np.array(T)
    # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    T = T.reshape(T.shape[0],T.shape[1])
    return np.mat(T).T
 
def eigenfaceCore(T):
    # 把均值变为0 axis = 1代表对各行求均值
    m = T.mean(axis = 1)
    A = T-m
    L = (A.T)*(A)
#     L = np.cov(A,rowvar = 0)
    # 计算AT *A的 特征向量和特征值V是特征值，D是特征向量
    V, D = np.linalg.eig(L)
    print(V)
    sortIndex = argsort(-V)
    V = V[sortIndex]
    D = D[:, sortIndex]
    L_eig = []
    for i in range(30):
        L_eig.append(D[:,i])
    L_eig = np.mat(np.reshape(np.array(L_eig),(-1,len(L_eig))))
    print(L_eig.shape)
    # 计算 A *AT的特征向量
    eigenface = A * L_eig
    print(eigenface.shape)
    return eigenface,m,A  
 
def recognize(testImage, eigenface,m,A):
    _,trainNumber = np.shape(eigenface)
    # 投影到特征脸后的
    projectedImage = eigenface.T*(A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage,dtype=np.uint8),cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray=cv.resize(testImageArray,IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size,1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    projectedTestImage = eigenface.T*(differenceTestImage)
    distance = []
    print(trainNumber)
    for i in range(0, trainNumber):
        q = projectedImage[:,i]
        temp = np.linalg.norm(projectedTestImage - q)
        distance.append(temp)
  
    minDistance = min(distance)
    index = distance.index(minDistance)
    print(index)
    cv.imshow("recognize result",cv.imread('./att-face/s1/'+str(index+1 )+'.pgm',cv.IMREAD_GRAYSCALE))
    cv.waitKey()
    return index+1
# 进行人脸识别主程序
def example(filename):
    T = createDatabase()
    eigenface,m,A = eigenfaceCore(T)
    testimage = filename
    print(testimage)
    print(recognize(testimage, eigenface,m,A))


# 构建可视化界面
def gui():
    root = tk.Tk()
    root.title("pca face")
    #点击选择图片时调用
    def select():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            s=filename # jpg图片文件名 和 路径。
            im=Image.open(s)
            tkimg=ImageTk.PhotoImage(im) # 执行此函数之前， Tk() 必须已经实例化。
            l.config(image=tkimg)
            btn1.config(command=lambda : example(filename))
            btn1.config(text = "开始识别")
            btn1.pack()
            # 重新绘制
            root.mainloop()
    # 显示图片的位置
    l = tk.Label(root)
    l.pack()
    
    btn = tk.Button(root,text="选择识别的图片",command=select)
    btn.pack()
    
    btn1 = tk.Button(root) # 开始识别按钮，刚开始不显示
    root.mainloop()
if __name__ == "__main__":
    example('homework2/att-face/s1/10.pgm')