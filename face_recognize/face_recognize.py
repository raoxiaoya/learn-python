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
