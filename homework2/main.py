from numpy import * 
import numpy as np
from numpy import linalg as la
import cv2
import os
import random
from numpy.linalg.linalg import eig
 

  
def loadImg(file_dir,m,n,l,energy): 
    '''
        载入图像，灰度化处理，统一尺寸，直方图均衡化 
        :param fileName: 图像文件名 
        :param dsize: 统一尺寸大小。元组形式 
        :return: 图像矩阵 
    '''
    # L的每一行表示一个训练图片
    picture = np.empty(shape=(n, m, l), dtype='float64')
    L = np.empty(shape=(m*n, l), dtype='float64')                  
    cur_img = 0
    training_ids = []
    for face_id in range(1, 40 + 1):
        # 随机选取训练图片
        ids = random.sample(range(1, 11), 6)  
        training_ids.append(ids)                            
        for training_id in ids:
            path_to_img = os.path.join(file_dir,
                        's' + str(face_id), str(training_id) + '.pgm')         
            # 读取灰度图
            img = cv2.imread(path_to_img, 0)                            
            img_col = np.array(img, dtype='float64').flatten()
            image = np.array(img, dtype='float64')     
            # 将图片存储到训练集中
            L[:, cur_img] = img_col[:]   
            picture[:,:, cur_img] = image[:]                              
            cur_img += 1
    mean_img_col = np.mean(L, axis=1) 
    for j in range(l):
        # 对每个图片减去平均脸，得到偏差矩阵
        L[:, j] -= mean_img_col
    # 求解协方差矩阵
    C = np.dot(L.T, L)
    eigvals,eigVects = linalg.eig(C)
    eigSortIndex = argsort(-eigvals)

    eigvals = eigvals[eigSortIndex]
    eigVects = eigVects[:, eigSortIndex]

    eigvals_sum = sum(eigvals[:])                                      
    evalues_count = 0   
    # 根据能量百分比决定取多少个特征脸                                                    
    evalues_energy = 0.0
    for evalue in eigvals:
        evalues_count += 1
        evalues_energy += evalue / eigvals_sum
        if evalues_energy >= energy:
            break
    eigvals = eigvals[0:evalues_count]
    eigVects = eigVects[:,0:evalues_count]
    eigpic = np.dot(picture, eigVects) 
    eigVects = np.dot(L, eigVects)      
    for i in range(10):
        img =  eigpic[:,:,i]
        cv2.imshow('img',img)     
        cv2.waitKey()                               
    # 将每一个特征向量继续归一化
    norms = np.linalg.norm(eigVects, axis=0)                           
    eigVects = eigVects / norms                                   
    # 求得每一个特征向量对导入人脸的权重向量
    W = np.dot(eigVects.transpose(), L)          
    return W
def classify(self, path_to_img, W, mean_img_col):
    img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
    img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
    img_col -= mean_img_col                                            # subract the mean column
    img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

    S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

    diff = W - S                                                       # finding the min ||W_j - S||
    norms = np.linalg.norm(diff, axis=0)

    closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
    return int(closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)
if __name__ == '__main__':
    file_list = 'D:/ZJU-learn/computer-vision/homework2/att_faces'
    faces_count = 40        # 40个人脸
    train_faces_count = 6   # 60%用于训练集
    test_faces_count = 4    # 40%用于测试集
    l = train_faces_count * faces_count
    m = 92
    n = 112
    mn = m*n
    mat = loadImg(file_list,m,n,l,energy=0.85)
    print(mat.shape)
    # avgImg,covVects,diffTrain = ReconginitionVector(file_list+'/s1/')
    # print(covVects)
    # cv2.imshow('img',avgImg)
    # cv2.waitKey()