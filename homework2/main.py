from numpy import * 
import numpy as np
from numpy import linalg as la
import cv2
import os
import random
from numpy.linalg.linalg import eig
import matplotlib.pyplot as plt 

training_ids = []      
faces_count = 40        # 40个人脸
train_faces_count = 6   # 60%用于训练集
test_faces_count = 4    # 40%用于测试集
l = train_faces_count * faces_count
m = 92
n = 112
mn = m*n
picture = np.empty(shape=(n, m, l), dtype='float64')
L = np.empty(shape=(m*n, l), dtype='float64')  
def loadImg(file_dir,energy): 
    # L的每一行表示一个训练图片
                
    cur_img = 0
    
    for face_id in range(1, 40 + 1):
        # 随机选取训练图片
        ids = [1,2,3,4,5,6]
        training_ids.append(ids)                            
        for training_id in ids:
            path_to_img = os.path.join(file_dir,
                        's' + str(face_id), str(training_id) + '.pgm')         
            # 读取灰度图
            img = cv2.imread(path_to_img, cv2.COLOR_BGR2GRAY)                      
            img_col = np.array(img).flatten()
            image = np.array(img)     
            # 将图片存储到训练集中
            L[:, cur_img] = img_col[:]   
            picture[:,:, cur_img] = image[:]                              
            cur_img += 1
    mean_img_col = np.mean(L, axis=1) 
    mean_img = np.mean(picture, axis=2)
    mean_img = np.array(mean_img, dtype=np.uint8)
    cv2.imwrite("homework2/avg_face.bmp", mean_img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])

    for j in range(l):
        # 对每个图片减去平均脸，得到偏差矩阵
        L[:, j] -= mean_img_col
    # 求解协方差矩阵

    C = np.matmul(L.T, L)
    eigvals,eigVects = linalg.eig(C)
    eigSortIndex = argsort(-eigvals)

    eigvals = eigvals[eigSortIndex]
    eigVects = eigVects[:, eigSortIndex]
    # 直接给定需要的PCs的数量                                   
    evalues_count = energy  
    # 根据能量百分比决定取多少个特征脸                                                    
    # evalues_energy = 0.0
    # for evalue in eigvals:
    #     evalues_count += 1
    #     evalues_energy += evalue / eigvals_sum
    #     if evalues_energy >= energy:
    #         break
    eigvals = eigvals[0:evalues_count]
    eigVects = eigVects[:,0:evalues_count]    
    # 将每一个特征向量继续归一化
    norms = np.linalg.norm(eigVects, axis=0)                           
    eigVects = eigVects / norms 
    eigenface = np.matmul(L, eigVects)     
    # 100个特征脸
    # 创建画布和子图对象
    # fig, axes = plt.subplots(1,10
    #                    ,figsize=(15,15)
    #                    ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
    #                    )
    # #填充图像
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(eigenface[:,i].reshape(112,92),cmap="gray") #reshape规定图片的大小，选择色彩的模式                   
    # plt.savefig("homework2/eigenface.png") 
    # plt.show()                            
    # 求得每一个特征向量对导入人脸的权重向量
    return eigenface, mean_img_col, eigVects
def classify(path_to_img, i):
    eigenface, avg_face, eigVects = loadImg('./homework2/att-face', i)
    img = cv2.imread(path_to_img, 0)                                       
    img_col = np.array(img, dtype='float64').flatten()                    
    img_col -= avg_face     
    S = np.matmul(eigenface.T, img_col)  
    resVal = inf
    res = 0
    for i in range(40):  
        for m in range(6):
            j = i*6 + m                       
            TrainVec = np.matmul(eigenface.T,L[:, j])
            if (array(S-TrainVec)**2).sum() < resVal:
                res =  i
                resVal = (array(S-TrainVec)**2).sum()                                                                            # Eigenspace, to find out the weights

    return res+1           # return the faceid (1..40)

def construct(j):
    # 表示各个特征脸的线性组合
    # TrainVec = np.matmul(eigenface.T, L[:, j])
    # 输出重构的近似人脸
    
    fig, axes = plt.subplots(1,10
                       ,figsize=(20,10)
                       ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
                       )
    i = 10

    for h, ax in enumerate(axes.flat):
        eigenface, avg_face, eigVects = loadImg(file_list,i)
        print(eigVects[j].shape)
        # pic_pca = np.matmul(eigVects[j],pic)
        ax.imshow(avg_face.reshape(112,92) + np.matmul(eigenface, eigVects[j]).reshape(112, 92), cmap="gray") # np.dot()矩阵乘法
        i = i + 20
    plt.savefig('homework2/face_reconstruct.png')
    plt.show()
def evaluate(faces_dir):
    print ('> Evaluating AT&T faces started')
    results_file = os.path.join('homework2/results', 'att_results.txt')               
    f = open(results_file, 'w')                                             

    test_count = 4 * 40               
        
    total_correct = []
    PCs = []
    # 记录随着PCs的增加,识别能力的增加
    for i in range(200):
        test_correct = 0
        for face_id in range(1, 40 + 1):
            for test_id in range(7, 11):
                            
                path_to_img = os.path.join(faces_dir,
                        's' + str(face_id), str(test_id) + '.pgm')          
                # predict label
                result_id = classify(path_to_img, i)
                result = (result_id == face_id)

                if result == True:
                    test_correct += 1
                    f.write('image: %s\nresult: correct\n\n' % path_to_img)
                else:
                    f.write('image: %s\nresult: wrong, got %2d\n\n' %
                            (path_to_img, result_id))

        print ('> Evaluating AT&T faces ended')
        accuracy = float(100. * test_correct / test_count)
        print ('Correct: ' + str(accuracy) + '%')
        f.write('Correct: %.2f\n' % (accuracy))
        total_correct.append(accuracy)
        PCs.append(i+1)
    f.close() 
    
    plt.plot(PCs,total_correct)
    plt.savefig('homework2/PCs-total_correct.png')  
    plt.show()                               

if __name__ == '__main__':
    file_list = './homework2/att-face'
    # 评估eigenface训练结果，完成人脸识别
    evaluate(file_list)
    # print(classify(file_list,10))
    # 得到特征脸
    # loadImg(file_list,10)
    # 重构人脸
    # construct(2)