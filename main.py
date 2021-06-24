import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, QTime
from functools import partial
import numpy as np
import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
my_video = cv2.VideoCapture(0)


class mainwindow(QWidget):
    def __init__(self):
        super(mainwindow, self).__init__()

        loader = QUiLoader()
        self.ui = loader.load("form.ui")
        self.ui.btn_webcam.clicked.connect(self.webcam)
        self.ui.btn_femoji.clicked.connect(self.filter_emoji)
        self.ui.btn_fsticker.clicked.connect(self.filter_sticker)
        self.ui.btn_fvertical.clicked.connect(self.filter_vertical)
        self.ui.btn_fanonymize.clicked.connect(self.filter_anonymize)

        self.ui.show()

    def webcam(self):
        while True:
            valdation, frame = my_video.read()
            if valdation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3)
            for i, face in enumerate(faces):
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def filter_emoji(self):
        sticker1 = cv2.imread('7777825.png')
        while True:
            valdation, frame = my_video.read()
            if valdation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3)
            for i, face in enumerate(faces):
                x, y, w, h = face
                resized_sticker = cv2.resize(sticker1, (w, h))
                frame[y:y+h, x:x+w] = resized_sticker
                cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def filter_sticker(self):
        eye = cv2.imread('eye1.png')
        lip = cv2.imread('lip1.png')
        while (True):
            ret, frame = my_video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = frame_gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_detector.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye = cv2.resize(eye, (ew, eh))
                    roi_color[ey:ey+ew, ex:ex+eh] = eye
                lips = smile_detector.detectMultiScale(roi_gray, 1.3, 20)
                for(lx, ly, lw, lh) in lips:
                    lip = cv2.resize(lip, (lw, lh))
                    roi_color[ly:ly+lh, lx:lx+lw] = lip

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def filter_anonymize(self):
        while True:
            valdation, frame = my_video.read()
            if valdation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, 1.3)
            for i, face in enumerate(faces):
                x, y, w, h = face
                fr_gray = frame_gray[y:y + h, x:x + w]
                fr_color = frame[y:y + h, x:x + w]
                fr_color = cv2.resize(fr_color, (16, 6))
                fr_color = cv2.resize(fr_color, (w, h))
                frame[y:y+w, x:x+h] = fr_color
            cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def filter_vertical(self):
        while True:
            valdation, frame = my_video.read()
            if valdation is not True:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows, cols = frame_gray.shape
            half_frame = frame_gray[0:rows//2, :]
            flipped_half_frame = cv2.flip(half_frame, 1)
            frame_gray[rows//2:, :] = flipped_half_frame
            cv2.imshow('output', frame_gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    app = QApplication([])
    window = mainwindow()
    sys.exit(app.exec_())
