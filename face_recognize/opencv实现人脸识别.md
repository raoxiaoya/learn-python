opencv实现人脸识别



#### 一、检测图片中的人脸



**人脸检测：** 通常被认为是在图像中找到面部（位置和大小）并可能提取它们以供人脸检测算法使用。

**人脸识别：** 人脸识别算法用于查找图像中唯一描述的特征。面部图像已经被提取、裁剪、调整大小，并通常转换为灰度。



**cv2.CascadeClassifier**

是OpenCV中用于物体检测的一个重要类，通常用于人脸和特征检测，特别适用于基于 Haar 特征的级联分类器。它可以用来检测图像中的特定对象，如人脸、眼睛、汽车等。

要使用 cv2.CascadeClassifier，首先需要加载一个预先训练好的 XML 文件，这个文件包含了用于检测特定对象的级联分类器信息。

参数说明：xml_file_path：这是一个字符串，指定了包含级联分类器信息的 XML 文件的路径。目录在`D:\ProgramData\Anaconda3\envs\tensorflow2.4\Lib\site-packages\cv2\data`。

![image-20240910115928111](D:\dev\php\magook\trunk\server\md\img\image-20240910115928111.png)

XML中存放的是训练后的特征池，特征size大小根据训练时的参数而定，检测的时候可以简单理解为就是将每个固定size特征（检测窗口）与输入图像的同样大小区域比较，如果匹配那么就记录这个矩形区域的位置，然后滑动窗口，检测图像的另一个区域，重复操作。由于输入的图像中特征大小不定，比如在输入图像中眼睛是50x50的区域，而训练时的是25x25，那么只有当输入图像缩小到一半的时候，才能匹配上，所以这里还有一个逐步缩小图像，也就是制作图像金字塔的流程。



**detectMultiScale** 

image：要检测对象的灰度图像。

scaleFactor：默认值1.1，每次缩小图像的比例，可以调整来适应不同尺寸的人脸。

minNeighbors：默认值3，控制每个检测到的矩形周围需要有多少邻近矩形才能保留。较大的值可以减少误报的数量，但可能会漏掉一些真实的目标。

minSize 和 maxSize：定义了检测对象的最小和最大尺寸。这对于过滤掉过小或过大的目标非常有用。



![image-20240910151228191](D:\dev\php\magook\trunk\server\md\img\image-20240910151228191.png)



关于参数`minNeighbors`，全网没有一个清晰的解释，不知所云。



`cv2`需要安装`opencv-python`

```bash
pip install opencv-python
opencv-python: 4.10.0.84
Python 3.8.0
```



```python
import cv2


def detect_faces_in_picture():
    # 读取一张图像
    # image = cv2.imread('20240910144915375.jpg')
    image = cv2.imread('20240911093212799.jpg')

    # 1.将图像转换为灰度图像，因为Haar级联分类器需要处理灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2.加载模块自带的人脸检测器分类器
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3.使用分类器检测图像中的人脸，返回一个包含人脸位置信息的列表
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

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


def detect_faces_in_video():
    # 1.加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2.打开摄像头，检测实时摄像头流
    # camera = cv2.VideoCapture(0)
    # 0 表示默认摄像头，如果有多个摄像头，可以尝试不同的索引
    # 检测视频文件，参数为视频路径
    camera = cv2.VideoCapture('ee0f844b67e1349aa706cace7e36c313.mp4')
    while True:
        # 3.读取摄像头的一帧图像
        ret, frame = camera.read()
        if not ret:
            print("无法获取帧")
            break

        # 4.将图像转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5.进行人脸检测
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

        # 6.在检测到的人脸上画矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 7.显示结果
        cv2.imshow('Face Detection', frame)

        # 8.按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9.释放摄像头资源并关闭所有窗口
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_faces_in_picture()

```

```bash
发现了4张人脸，他们的位置分别是： [[300 272  50  50]
 [556 298  57  57]
 [464 220  57  57]
 [401 305  53  53]]
```

![image-20240911093051188](D:\dev\php\magook\trunk\server\md\img\image-20240911093051188.png)

#### 二、实时检测视频中的人脸

数据源可以是视频文件，也可以是摄像头的视频流。在循环中一帧一帧读取图片并识别，并实时将标记的图片展示出来。



#### 三、识别出人脸是谁

这里需要先上传素材进行训练，然后才能识别出图片对应的标签，即需要训练数据和测试数据。将人脸图像先录入系统，每一个人录一张即可，通过训练，使标签和图片对应上，即 Y=f(x)，其中x为图像，y为标签。

```bash
pip install opencv-contrib-python
否则会报错
AttributeError: module 'cv2' has no attribute 'face'
```

```python
import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def trainedModel():
    '''
    训练模型
    '''
    face_list_path = "./face_list/"
    c = 0
    X, y = [], []
    for filename in os.listdir(face_list_path):
        filepath = os.path.join(face_list_path, filename)
        gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        a, b, w, h = faces[0]
        img_face = cv2.rectangle(gray, (a, b), (a + w, b + h), (255, 0, 0), 2)
        img_face = cv2.resize(img_face[b:b+h, a:a+w], (200, 200))

        # cv2.imshow('Faces found', img_face)
        # cv2.waitKey(0)

        # X.append(np.asarray(img_face, dtype=np.uint8))
        X.append(img_face)
        y.append(c)

        c = c + 1

    # 训练
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    model.save('recognize_face.xml')


def recognizeFace(model, img):
    '''
    识别
    '''
    face_list_name = ['huge', 'liudehua', 'zhangyi']
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    for (a, b, w, h) in faces:
        img_face = cv2.rectangle(gray, (a, b), (a + w, b + h), (255, 0, 0), 2)
        img_face = cv2.resize(img_face[b:b+h, a:a+w], (200, 200))

        # cv2.imshow('Faces found', img_face)
        # cv2.waitKey(0)

        result = model.predict(img_face)

        print("Label: %s, Name: %s, Confidence: %.2f" %
              (result[0], face_list_name[result[0]], result[1]))

        # 展示
        img = cv2.rectangle(img, (a, b), (a + w, b + h), (255, 0, 0), 2)
        cv2.putText(img, face_list_name[result[0]], (a, b - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    cv2.imshow("face", img)


def recognizeFaceFromPicture(model):
    '''
    从图片中识别
    '''
    recognizeFace(model, cv2.imread('huge2.jpg'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recognizeFaceFromVideo(model):
    '''
    从视频中识别
    '''
    camera = cv2.VideoCapture(0)
    while (True):
        ret, img = camera.read()
        if not ret:
            print("无法获取帧")
            break
        recognizeFace(model, img)

    camera.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 训练模型
    # trainedModel()

    # 加载已经训练好的模型
    model = cv2.face.EigenFaceRecognizer_create()
    model.read('recognize_face.xml')

    recognizeFaceFromPicture(model)

```

face_list 目录下有三个文件用来训练：huge.jpg, liudehua.jpg, zhangyi.jpg。

huge2.jpg 是胡歌的另一个图片。



![image-20240911155628534](D:\dev\php\magook\trunk\server\md\img\image-20240911155628534.png)

