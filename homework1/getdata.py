import cv2
import os

path=r'./frame/'
cap = cv2.VideoCapture('./1.mp4')
i=1
if not os.path.exists(path):
	os.makedirs(path)
while(cap.isOpened()):
	ret, frame = cap.read()#ret是bool型，当读完最后一帧就是False，frame是ndarray型
	if ret == False:#
		break
	
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转灰度
	#cv2.imwrote(' ',gray) #路径无中文存图
	cv2.imencode('.jpg', frame)[1].tofile(path+'frame_'+str(i)+'.jpg')#路径含中文存图
	cv2.imshow('frame',frame)
	i+=1
	
	#不加这一句，由于计算速度极快，直接会显示到最后一帧
	if cv2.waitKey(1) & 0xFF == ord('q'):#检测到按下q，就break。waitKey(1)相当于1ms延时
		break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
