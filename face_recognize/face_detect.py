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