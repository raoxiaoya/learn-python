import cv2

# 读取一张图像
# image = cv2.imread('20240910144915375.jpg')
image = cv2.imread('ee0f844b67e1349aa706cace7e36c313.jpeg')

# 1.将图像转换为灰度图像，因为Haar级联分类器需要处理灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2.加载模块自带的人脸检测器分类器
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3.使用分类器检测图像中的人脸，返回一个包含人脸位置信息的列表
faces = faceCascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 4.打印检测到的人脸数量和坐标
print(f'发现了{len(faces)}张人脸，他们的位置分别是：', faces)

# 5.在原始图像上绘制矩形框来标记检测到的人脸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 6.显示标注后的图像
cv2.imshow('Faces found', image)

# 7.等待用户按键退出，然后关闭图像窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
