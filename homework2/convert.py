"""
    完成对rgb图片从jpg格式到pgm格式的转储
"""
from PIL import Image
import matplotlib.pyplot as plt 

print(Image.open('homework2/att-face/s1/1.pgm').size)
for i in range(1,11):
    pic = Image.open('homework2/own_face/%i.jpg'  % i).convert('L')#如果是rgb图，要转为单通道的灰度图；如果是灰度图，那么去掉convert，保持灰度图
    
    pic = pic.crop((800,550,3200,4700))  #坐标从左上开始
    plt.figure()
    plt.imshow(pic)
    plt.show()
    pic = pic.resize((92,112), Image.ANTIALIAS)
    print(pic.size)
    pic.save('homework2/own_face/%i.pgm'  % i)