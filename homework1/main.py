import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
# empty image
def img_HW(imgpath):
    image = cv2.imread(imgpath)
    height, width = image.shape[0], image.shape[1]
    return height, width
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
        "font/STXINWEI.TTF", textSize)
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
img = np.zeros((640, 800, 3), np.uint8)
fps = 24
size = (800,640)
video = cv2.VideoWriter("VideoTest1.avi", 
                cv2.VideoWriter_fourcc(*'XVID'), fps, size)
cv2.namedWindow('test video',0)
cv2.resizeWindow("test video", 800, 640);                                                      
item = './1.png'
item2 = './2.png'
item3 = './3.png'                           
item4 = './4.png'
img1 = cv2.imread(item)
img2 = cv2.imread(item2)
img3 = cv2.imread(item3)
img4 = cv2.imread(item4)

img1=cv2.resize(img1,(800,640))
img2=cv2.resize(img2,(800,640))
img3=cv2.resize(img3,(200,160))
img4=cv2.resize(img4,(800,640))

for it in range(100+1):
    weight = it / 100
    # print(weight)
    img = cv2.addWeighted(img1, 1-weight,img2, weight, 3)
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)

    video.write(img)   
                                                               
for it in range(200):
    x_offset=y_offset=it*2
    img = img2.copy()
    img[y_offset:y_offset+img3.shape[0], x_offset:x_offset+img3.shape[1]] = img3 
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img)
    
    
temp = img
for i in range(50):

    img = cv2ImgAddText(temp, "12121049 於其樊", 50+i*3, 10, (0, 0 , 0), 50)
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img) 
img = img4
# draw a line
cv2.line(img, (100,0), (41,181), (255, 0, 0), 10)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)
cv2.line(img, (100,0), (159,181), (255, 0, 0), 10)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)
cv2.line(img, (5,69), (195,69), (255, 0, 0), 10)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)
cv2.line(img, (5,69), (159,181), (255, 0, 0), 10)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)
cv2.line(img, (195,69), (41,181), (255, 0, 0), 10)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)

# draw a rectangle
cv2.rectangle(img, (0,0), (255,255), (0, 255, 0), 5)
cv2.imshow("test video",img)
if(cv2.waitKey(100)==32):
    cv2.waitKey(0)
video.write(img)
# draw a circle
for i in range(0, 100): 
    radius = np.random.randint(15, high=100) 
    color = np.random.randint(0, 255, 3, dtype=np.int32)

    pt = (300,300) 
    
    cv2.circle(img, tuple(pt), radius, (int(color[0]), int(color[1]), int(color[2])), -1) 
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img)


def Srotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
    return sRotatex,sRotatey
temp = img.copy()
for i in range(50):
    radius = 50
    color = (30,60,80)
    sRotatex,sRotatey = Srotate(i, 400, 200, 600, 300)
    pt = (int(sRotatex),int(sRotatey))
    img = temp.copy()
    cv2.circle(img, tuple(pt), radius, color, -1) 
    cv2.imshow("test video",img)
    if(cv2.waitKey(100)==32):
        cv2.waitKey(0)
    video.write(img)
# draw a ellipse
temp = img.copy()
for i in range(360):
    
    img = temp.copy()
    cv2.ellipse(img, (450,450), (150, 75), i, 0, 360, (0, 255, 255), -1)
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img)
# add text
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,600), font, 6,(0,0,0),2)
cv2.imshow("test video",img)
if(cv2.waitKey(10)==32):
    cv2.waitKey(0)
video.write(img)

filelist = './frame/'
fps = 24
size = (640,480)


for item in range(1,742):
    
    item = filelist + "frame_%d.jpg" % item
    img = cv2.imread(item)
    img = cv2.resize(img,(800,640))
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img)
imgend = np.zeros((640, 800, 3), np.uint8)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgend,'END',(200,400), font, 8,(255,255,255),2)
for it in range(50+1):
    weight = it / 50
    img = cv2.addWeighted(img, 1-weight,imgend, weight, 3)
    cv2.imshow("test video",img)
    cv2.waitKey(100)
    video.write(img)  
temp = img
for i in range(100):
    img = cv2ImgAddText(temp, "by qifanyu", 50+i*3, 500, (255, 255 , 255), 50)
    cv2.imshow("test video",img)
    if(cv2.waitKey(10)==32):
        cv2.waitKey(0)
    video.write(img)  
video.release()
cv2.destroyAllWindows()