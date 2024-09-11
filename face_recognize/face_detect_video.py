import cv2


def detect_faces_in_video():
    # 1.加载预训练的人脸检测模型
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2.打开摄像头，检测实时摄像头流
    # cap = cv2.VideoCapture(0)
    # 0 表示默认摄像头，如果有多个摄像头，可以尝试不同的索引
    # 检测视频文件，参数为视频路径
    cap = cv2.VideoCapture('quanchanhong.mp4')
    while True:
        # 3.读取摄像头的一帧图像
        ret, frame = cap.read()
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
    cap.release()
    cv2.destroyAllWindows()


# 运行函数
detect_faces_in_video()
